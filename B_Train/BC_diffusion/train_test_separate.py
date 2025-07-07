import numpy as np
import U_Chem.visualizer as vis
from U_Chem.analyze import analyze_stability_for_molecules
from U_Chem.sampling import sample_chain, sample, sample_sweep_conditional, sample_sweep_conditional_separate, \
    sample_separate, sample_chain_separate
from U_Chem.dataset_info import anion_data_info, cation_data_info
import B_Train.BC_diffusion.utils as utils
import U_Chem.utils as chemutils
from U_Chem import losses
import time
import torch

from U_Sub_Structure.EDM_model.utils import remove_mean_with_mask, assert_mean_zero_with_mask, assert_correctly_masked
from U_Sub_Structure.EDM_model.utils import sample_center_gravity_zero_gaussian_with_mask


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, optim,
                nodes_dist, gradnorm_queue, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        if args.ion == "anion":
            x = data['anion_atom_position_list'].to(device, dtype)
            node_mask = data['anion_node_mask_list'].to(device, dtype).unsqueeze(2)
            edge_mask = data['anion_edge_mask_list'].to(device, dtype)
            one_hot = data['anion_one_hot_list'].to(device, dtype)
            charges = (data['anion_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)

            x = remove_mean_with_mask(x, node_mask)
        else:
            x = data['cation_atom_position_list'].to(device, dtype)
            node_mask = data['cation_node_mask_list'].to(device, dtype).unsqueeze(2)
            edge_mask = data['cation_edge_mask_list'].to(device, dtype)
            one_hot = data['cation_one_hot_list'].to(device, dtype)
            charges = (data['cation_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)

            x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(
                x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        # check_mask_correct([x, one_hot, charges], node_mask)
        # assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}
        context = chemutils.prepare_context(data, args.ion)
        context = context.to(device, dtype)
        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_separate(
            args, model_dp, nodes_dist, x, h, node_mask, edge_mask, context)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f},"
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            save_and_sample_conditional(args, device, model_ema, prop_dist, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, prop_dist=prop_dist, epoch=epoch, batch_id=str(i))
            sample_different_sizes_and_save(
                model_ema, nodes_dist=nodes_dist,
                args=args, device=device, prop_dist=prop_dist, epoch=epoch, batch_id=str(i))
            save_reconstruct_chain(model_ema, args, device, prop_dist, loader, dtype, epoch=0, id_from=0, batch_id='')
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{i}")
            vis.ion_visualize_chain(
                f"ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{i}/chain/")
            vis.ion_visualize_chain(
                f"ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{i}/recon_chain/")
            vis.ion_visualize_chain(
                'ZDataD_Molecules/ZDA_process_mol/%s/%s/epoch_%d/conditional/' % (args.exp_name, args.ion, epoch),
                mode='conditional')


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        n_iterations = len(loader)

        for i, data in enumerate(loader):
            if i + 1 == args.stop_test:
                break
            x = data['anion_atom_position_list'].to(device, dtype)
            node_mask = data['anion_node_mask_list'].to(device, dtype).unsqueeze(2)
            edge_mask = data['anion_edge_mask_list'].to(device, dtype)
            one_hot = data['anion_one_hot_list'].to(device, dtype)
            charges = (data['anion_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)
            batch_size = x.size(0)
            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            context = chemutils.prepare_context(data, args.ion)
            context = context.to(device, dtype)
            assert_correctly_masked(context, node_mask)

            # transform batch through flow
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_separate(
                args, eval_model,
                nodes_dist, x, h, node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size

            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch / n_samples:.2f}")

    return nll_epoch / n_samples


def save_and_sample_chain(model, args, device, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain_separate(args=args, device=device, flow=model, n_tries=1, prop_dist=prop_dist)
    vis.save_xyz_file(
        f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{batch_id}/chain/',
        one_hot, charges, x, anion_data_info if args.ion == "anion" else cation_data_info,
        id_from, name='chain')

    return one_hot, charges, x


def save_reconstruct_chain(model, args, device, prop_dist, loader, dtype,
                           epoch=0, id_from=0, batch_id=''):
    for i, data in enumerate(loader):
        if args.ion == "anion":
            x = data['anion_atom_position_list'].to(device, dtype)
            node_mask = data['anion_node_mask_list'].to(device, dtype).unsqueeze(2)
            edge_mask = data['anion_edge_mask_list'].to(device, dtype)
            one_hot = data['anion_one_hot_list'].to(device, dtype)
            charges = (data['anion_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)
            n_nodes = data['anion_n_nodes_list'][0]
        else:
            x = data['cation_atom_position_list'].to(device, dtype)
            node_mask = data['cation_node_mask_list'].to(device, dtype).unsqueeze(2)
            edge_mask = data['cation_edge_mask_list'].to(device, dtype)
            one_hot = data['cation_one_hot_list'].to(device, dtype)
            charges = (data['cation_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)
            n_nodes = data['cation_n_nodes_list'][0]
        x = remove_mean_with_mask(x, node_mask)
        context = chemutils.prepare_context(data, args.ion)
        context = context[0, :n_nodes].unsqueeze(0)
        context = context.to(device, dtype)
        break
    one_hot, charges, x = sample_chain_separate(args=args, device=device, flow=model, n_tries=1, prop_dist=prop_dist,
                                                context=context, n_nodes=n_nodes)
    vis.save_xyz_file(
        f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{batch_id}/recon_chain/',
        one_hot, charges, x, anion_data_info if args.ion == "anion" else cation_data_info,
        id_from, name='chain')
    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample_separate(
            args, device, model, prop_dist=prop_dist,
            nodesxsample=nodesxsample
        )
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/{args.ion}/epoch_{epoch}_{batch_id}/',
                          one_hot, charges, x, anion_data_info if args.ion == "anion" else cation_data_info,
                          batch_size * counter, name=args.ion)


def analyze_and_save(epoch, model_sample, args, device, prop_dist, context=None,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        # anion_nodesxsample = nodes_dist.sample(batch_size)
        # cation_nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample_separate(args, device, model_sample, prop_dist, context=context)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, cation_data_info)

    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, epoch=0, id_from=0):
    (one_hot, charges, x, node_mask) = sample_sweep_conditional_separate(
        args, device, model, prop_dist, n_nodes=10, n_frames=100)

    vis.save_xyz_file(
        path='ZDataD_Molecules/ZDA_process_mol/%s/%s/epoch_%d/conditional/' % (args.exp_name, args.ion, epoch),
        one_hot=one_hot,
        charges=charges,
        positions=x,
        dataset_info=anion_data_info if args.ion == "anion" else cation_data_info,
        id_from=id_from, name=args.ion, node_mask=node_mask)

    return one_hot, charges, x
