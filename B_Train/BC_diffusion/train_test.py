import wandb
import numpy as np
import U_Chem.visualizer as vis
from U_Chem.analyze import analyze_stability_for_molecules
from U_Chem.sampling import sample_chain, sample, sample_sweep_conditional
from U_Chem.dataset_info import anion_data_info, cation_data_info
import utils
import U_Chem.utils as chemutils
from U_Chem import losses
import time
import torch

from U_Sub_Structure.EDM_model.utils import remove_mean_with_mask, assert_mean_zero_with_mask, assert_correctly_masked
from U_Sub_Structure.EDM_model.utils import sample_center_gravity_zero_gaussian_with_mask


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, optim,
                anion_nodes_dist, cation_nodes_dist, gradnorm_queue, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        anion_x = data['anion_atom_position_list'].to(device, dtype)
        anion_node_mask = data['anion_node_mask_list'].to(device, dtype).unsqueeze(2)
        anion_edge_mask = data['anion_edge_mask_list'].to(device, dtype)
        anion_one_hot = data['anion_one_hot_list'].to(device, dtype)
        anion_charges = (data['anion_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
            device, dtype)

        anion_x = remove_mean_with_mask(anion_x, anion_node_mask)

        cation_x = data['cation_atom_position_list'].to(device, dtype)
        cation_node_mask = data['cation_node_mask_list'].to(device, dtype).unsqueeze(2)
        cation_edge_mask = data['cation_edge_mask_list'].to(device, dtype)
        cation_one_hot = data['cation_one_hot_list'].to(device, dtype)
        cation_charges = (data['cation_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
            device, dtype)

        cation_x = remove_mean_with_mask(cation_x, cation_node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            anion_eps = sample_center_gravity_zero_gaussian_with_mask(
                anion_x.size(), anion_x.device, anion_node_mask)
            anion_x = anion_x + anion_eps * args.augment_noise
            cation_eps = sample_center_gravity_zero_gaussian_with_mask(
                cation_x.size(), cation_x.device, cation_node_mask)
            cation_x = cation_x + cation_eps * args.augment_noise

        anion_x = remove_mean_with_mask(anion_x, anion_node_mask)
        cation_x = remove_mean_with_mask(cation_x, cation_node_mask)
        if args.data_augmentation:
            anion_x = utils.random_rotation(anion_x).detach()
            cation_x = utils.random_rotation(cation_x).detach()

        check_mask_correct([anion_x, anion_one_hot, anion_charges], anion_node_mask)
        assert_mean_zero_with_mask(anion_x, anion_node_mask)
        check_mask_correct([cation_x, cation_one_hot, cation_charges], cation_node_mask)
        assert_mean_zero_with_mask(cation_x, cation_node_mask)

        anion_h = {'categorical': anion_one_hot, 'integer': anion_charges}
        cation_h = {'categorical': cation_one_hot, 'integer': cation_charges}
        anion_context, cation_context = chemutils.prepare_context(data)
        anion_context = anion_context.to(device, dtype)
        cation_context = cation_context.to(device, dtype)
        optim.zero_grad()

        # transform batch through flow
        anion_nll, cation_nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
            args, model_dp,
            anion_nodes_dist, anion_x, anion_h, anion_node_mask, anion_edge_mask,
            cation_nodes_dist, cation_x, cation_h, cation_node_mask, cation_edge_mask,
            anion_context, cation_context)
        # standard nll from forward KL
        loss = anion_nll + cation_nll + args.ode_regularization * reg_term
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
                  f"Loss {loss.item():.2f}, Anion NLL: {anion_nll.item():.2f}, Cation NLL: {cation_nll.item():.2f},"
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(anion_nll.item() + cation_nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            save_and_sample_conditional(args, device, model_ema, prop_dist, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, prop_dist=prop_dist, epoch=epoch, batch_id=str(i))
            sample_different_sizes_and_save(
                model_ema, anion_nodes_dist=anion_nodes_dist, cation_nodes_dist=cation_nodes_dist,
                args=args, device=device, prop_dist=prop_dist, epoch=epoch, batch_id=str(i))
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{i}", wandb=wandb)
            vis.visualize_chain(f"ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{i}/chain/",
                                wandb=wandb)

            vis.visualize_chain(
                "ZDataD_Molecules/ZDA_process_mol/%s/epoch_%d/conditional/" % (args.exp_name, epoch),
                wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": anion_nll.item() + cation_nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, anion_nodes_dist, cation_nodes_dist,
         partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            if i + 1 == args.stop_test:
                break
            anion_x = data['anion_atom_position_list'].to(device, dtype)
            anion_node_mask = data['anion_node_mask_list'].to(device, dtype).unsqueeze(2)
            anion_edge_mask = data['anion_edge_mask_list'].to(device, dtype)
            anion_one_hot = data['anion_one_hot_list'].to(device, dtype)
            anion_charges = (data['anion_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)
            batch_size = anion_x.size(0)
            cation_x = data['cation_atom_position_list'].to(device, dtype)
            cation_node_mask = data['cation_node_mask_list'].to(device, dtype).unsqueeze(2)
            cation_edge_mask = data['cation_edge_mask_list'].to(device, dtype)
            cation_one_hot = data['cation_one_hot_list'].to(device, dtype)
            cation_charges = (data['cation_charge_list'].unsqueeze(-1) if args.include_charges else torch.zeros(0)).to(
                device, dtype)
            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                anion_eps = sample_center_gravity_zero_gaussian_with_mask(anion_x.size(),
                                                                          anion_x.device,
                                                                          anion_node_mask)
                anion_x = anion_x + anion_eps * args.augment_noise

                cation_eps = sample_center_gravity_zero_gaussian_with_mask(cation_x.size(),
                                                                           cation_x.device,
                                                                           cation_node_mask)
                cation_x = cation_x + cation_eps * args.augment_noise

            anion_x = remove_mean_with_mask(anion_x, anion_node_mask)
            check_mask_correct([anion_x, anion_one_hot, anion_charges], anion_node_mask)
            assert_mean_zero_with_mask(anion_x, anion_node_mask)

            anion_h = {'categorical': anion_one_hot, 'integer': anion_charges}

            cation_x = remove_mean_with_mask(cation_x, cation_node_mask)
            check_mask_correct([cation_x, cation_one_hot, cation_charges], cation_node_mask)
            assert_mean_zero_with_mask(cation_x, cation_node_mask)

            cation_h = {'categorical': cation_one_hot, 'integer': cation_charges}

            anion_context, cation_context = chemutils.prepare_context(data)
            anion_context = anion_context.to(device, dtype)
            cation_context = cation_context.to(device, dtype)
            assert_correctly_masked(anion_context, anion_node_mask)
            assert_correctly_masked(cation_context, cation_node_mask)

            # transform batch through flow
            anion_nll, cation_nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
                args, eval_model,
                anion_nodes_dist, anion_x, anion_h, anion_node_mask, anion_edge_mask,
                cation_nodes_dist, cation_x, cation_h, cation_node_mask, cation_edge_mask,
                anion_context, cation_context)
            # standard nll from forward KL

            nll_epoch += anion_nll.item() * batch_size + cation_nll.item() * batch_size

            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch / n_samples:.2f}")

    return nll_epoch / n_samples


def save_and_sample_chain(model, args, device, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    anion_one_hot, anion_charges, anion_x, cation_one_hot, cation_charges, cation_x = sample_chain(
        args=args, device=device, flow=model,
        n_tries=1, prop_dist=prop_dist)
    vis.save_xyz_file(f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      anion_one_hot, anion_charges, anion_x, anion_data_info, id_from, name='anion chain')
    vis.save_xyz_file(f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      cation_one_hot, cation_charges, cation_x, cation_data_info, id_from, name='cation chain')

    return anion_one_hot, anion_charges, anion_x, cation_one_hot, cation_charges, cation_x


def sample_different_sizes_and_save(model, anion_nodes_dist, cation_nodes_dist, args, device, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        anion_nodesxsample = anion_nodes_dist.sample(batch_size)
        cation_nodesxsample = cation_nodes_dist.sample(batch_size)
        (anion_one_hot, anion_charges, anion_x, anion_node_mask,
         cation_one_hot, cation_charges, cation_x, cation_node_mask) = sample(
            args, device, model, prop_dist=prop_dist,
            anion_nodesxsample=anion_nodesxsample, cation_nodesxsample=cation_nodesxsample
        )
        print(f"Generated anion molecule: Positions {anion_x[:-1, :, :]}")
        print(f"Generated cation molecule: Positions {anion_x[:-1, :, :]}")
        vis.save_xyz_file(f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{batch_id}/',
                          anion_one_hot, anion_charges, anion_x, anion_data_info,
                          batch_size * counter, name='anion')
        vis.save_xyz_file(f'ZDataD_Molecules/ZDA_process_mol/{args.exp_name}/epoch_{epoch}_{batch_id}/',
                          cation_one_hot, cation_charges, cation_x, cation_data_info,
                          batch_size * counter, name='cation')


def analyze_and_save(epoch, model_sample, args, device, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    anion_molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    cation_molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        # anion_nodesxsample = nodes_dist.sample(batch_size)
        # cation_nodesxsample = nodes_dist.sample(batch_size)
        (anion_one_hot, anion_charges, anion_x, anion_node_mask,
         cation_one_hot, cation_charges, cation_x, cation_node_mask) = sample(args, device, model_sample, prop_dist)

        anion_molecules['one_hot'].append(anion_one_hot.detach().cpu())
        anion_molecules['x'].append(anion_x.detach().cpu())
        anion_molecules['node_mask'].append(anion_node_mask.detach().cpu())

        cation_molecules['one_hot'].append(cation_one_hot.detach().cpu())
        cation_molecules['x'].append(cation_x.detach().cpu())
        cation_molecules['node_mask'].append(cation_node_mask.detach().cpu())

    anion_molecules = {key: torch.cat(anion_molecules[key], dim=0) for key in anion_molecules}
    anion_validity_dict, anion_rdkit_tuple = analyze_stability_for_molecules(anion_molecules, anion_data_info)

    cation_molecules = {key: torch.cat(cation_molecules[key], dim=0) for key in cation_molecules}
    cation_validity_dict, cation_rdkit_tuple = analyze_stability_for_molecules(cation_molecules, cation_data_info)

    wandb.log(anion_validity_dict)
    wandb.log(cation_validity_dict)
    if anion_rdkit_tuple is not None:
        wandb.log({'Validity': anion_rdkit_tuple[0][0], 'Uniqueness': anion_rdkit_tuple[0][1],
                   'Novelty': anion_rdkit_tuple[0][2]})
        wandb.log({'Validity': cation_rdkit_tuple[0][0], 'Uniqueness': cation_rdkit_tuple[0][1],
                   'Novelty': cation_rdkit_tuple[0][2]})
    return anion_validity_dict, cation_validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, epoch=0, id_from=0):
    (anion_one_hot, anion_charges, anion_x, anion_node_mask,
     cation_one_hot, cation_charges, cation_x, cation_node_mask) = sample_sweep_conditional(
        args, device, model, prop_dist, anion_n_nodes=10, cation_n_nodes=10, n_frames=100)

    vis.save_xyz_file(
        path='ZDataD_Molecules/ZDA_process_mol/%s/epoch_%d/conditional/' % (args.exp_name, epoch),
        one_hot=anion_one_hot,
        charges=anion_charges,
        positions=anion_x,
        dataset_info=anion_data_info,
        id_from=id_from, name='anion', node_mask=anion_node_mask)
    vis.save_xyz_file(
        path='ZDataD_Molecules/ZDA_process_mol/%s/epoch_%d/conditional/' % (args.exp_name, epoch),
        one_hot=cation_one_hot,
        charges=cation_charges,
        positions=cation_x,
        dataset_info=cation_data_info,
        id_from=id_from, name='cation', node_mask=cation_node_mask)

    return anion_one_hot, anion_charges, anion_x, cation_one_hot, cation_charges, cation_x
