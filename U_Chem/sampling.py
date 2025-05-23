import sys

import numpy as np
import torch
import torch.nn.functional as F
from U_Chem.analyze import check_stability
from U_Chem.losses import assert_correctly_masked
from U_Chem.dataset_info import anion_data_info, cation_data_info
from U_Sub_Structure.EDM_model.utils import assert_mean_zero_with_mask


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = [z]
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, prop_dist=None):
    n_samples = 1
    anion_n_nodes = 10
    cation_n_nodes = 10

    if args.anion_context_node_nf > 0:
        anion_context, cation_context = prop_dist.sample(anion_n_nodes, cation_n_nodes)
        anion_context = anion_context.unsqueeze(1).unsqueeze(0).repeat(1, anion_n_nodes, 1).to(device)
        cation_context = cation_context.unsqueeze(1).unsqueeze(0).repeat(1, cation_n_nodes, 1).to(device)
        # context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        anion_context = None
        cation_context = None

    anion_node_mask = torch.ones(n_samples, anion_n_nodes, 1).to(device)
    anion_edge_mask = (1 - torch.eye(anion_n_nodes)).unsqueeze(0)
    anion_edge_mask = anion_edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    cation_node_mask = torch.ones(n_samples, cation_n_nodes, 1).to(device)
    cation_edge_mask = (1 - torch.eye(cation_n_nodes)).unsqueeze(0)
    cation_edge_mask = cation_edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    anion_one_hot, anion_charges, anion_x, cation_one_hot, cation_charges, cation_x = None, None, None, None, None, None
    for i in range(n_tries):
        anion_chain, cation_chain = flow.sample_chain(
            n_samples,
            anion_n_nodes, anion_node_mask, anion_edge_mask, anion_context,
            cation_n_nodes, cation_node_mask, cation_edge_mask, cation_context,
            keep_frames=100)
        anion_chain = reverse_tensor(anion_chain)
        cation_chain = reverse_tensor(cation_chain)

        """ANION"""
        # Repeat last frame to see final sample better.
        anion_chain = torch.cat([anion_chain, anion_chain[-1:].repeat(10, 1, 1)], dim=0)
        anion_x = anion_chain[-1:, :, 0:3]
        anion_one_hot = anion_chain[-1:, :, 3:-1]
        anion_one_hot = torch.argmax(anion_one_hot, dim=2)

        anion_atom_type = anion_one_hot.squeeze(0).cpu().detach().numpy()
        anion_x_squeeze = anion_x.squeeze(0).cpu().detach().numpy()
        anion_mol_stable = check_stability(anion_x_squeeze, anion_atom_type, anion_data_info)[0]

        # Prepare entire chain.
        anion_x = anion_chain[:, :, 0:3]
        anion_one_hot = anion_chain[:, :, 3:-1]
        anion_one_hot = F.one_hot(
            torch.argmax(anion_one_hot, dim=2), num_classes=len(anion_data_info['atom_decoder']))
        anion_charges = torch.round(anion_chain[:, :, -1:]).long()

        """CATION"""
        cation_chain = torch.cat([cation_chain, cation_chain[-1:].repeat(10, 1, 1)], dim=0)
        cation_x = cation_chain[-1:, :, 0:3]
        cation_one_hot = cation_chain[-1:, :, 3:-1]
        cation_one_hot = torch.argmax(cation_one_hot, dim=2)

        cation_atom_type = cation_one_hot.squeeze(0).cpu().detach().numpy()
        cation_x_squeeze = cation_x.squeeze(0).cpu().detach().numpy()
        cation_mol_stable = check_stability(cation_x_squeeze, cation_atom_type, cation_data_info)[0]

        # Prepare entire chain.
        cation_x = cation_chain[:, :, 0:3]
        cation_one_hot = cation_chain[:, :, 3:-1]
        cation_one_hot = F.one_hot(
            torch.argmax(cation_one_hot, dim=2), num_classes=len(cation_data_info['atom_decoder']))
        cation_charges = torch.round(cation_chain[:, :, -1:]).long()

        if anion_mol_stable and cation_mol_stable:
            print('Found stable molecule to visualize :)')
            break
        elif i == n_tries - 1:
            print('Did not find stable molecule, showing last sample.')

    return anion_one_hot, anion_charges, anion_x, cation_one_hot, cation_charges, cation_x


def sample_chain_separate(args, device, flow, n_tries, prop_dist=None, context=None, n_nodes=None):
    n_samples = 1
    n_nodes = 10 if n_nodes is None else n_nodes
    if context is None:
        if args.context_node_nf > 0:
            anion_context, cation_context = prop_dist.sample(n_nodes, n_nodes)
            context = anion_context if args.ion == "anion" else cation_context
            context = context.unsqueeze(1).unsqueeze(0).repeat(1, n_nodes, 1).to(device)
            # context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        else:
            context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)
    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    one_hot, charges, x = None, None, None
    for i in range(n_tries):  # = 1 when training
        chain = flow.sample_chain(
            n_samples, n_nodes, node_mask, edge_mask, context,
            keep_frames=100)
        chain = reverse_tensor(chain)

        # Repeat last frame to see final sample better.
        chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
        x = chain[-1:, :, 0:3]
        one_hot = chain[-1:, :, 3:-1]
        one_hot = torch.argmax(one_hot, dim=2)

        atom_type = one_hot.squeeze(0).cpu().detach().numpy()
        x_squeeze = x.squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type,
                                     anion_data_info if args.ion == "anion" else cation_data_info)[0]

        # Prepare entire chain.
        x = chain[:, :, 0:3]
        one_hot = chain[:, :, 3:-1]
        one_hot = F.one_hot(
            torch.argmax(one_hot, dim=2), num_classes=len(anion_data_info['atom_decoder']))
        anion_charges = torch.round(chain[:, :, -1:]).long()

        if mol_stable:
            print('Found stable molecule to visualize :)')
            break
        elif i == n_tries - 1:
            print('Did not find stable molecule, showing last sample.')

    return one_hot, charges, x


def sample(args, device, generative_model,
           prop_dist=None, anion_nodesxsample=torch.tensor([10]), cation_nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    anion_max_n_nodes = anion_data_info['max_n_nodes']
    cation_max_n_nodes = cation_data_info['max_n_nodes']

    assert int(torch.max(anion_nodesxsample)) <= anion_max_n_nodes
    assert int(torch.max(cation_nodesxsample)) <= cation_max_n_nodes
    batch_size = len(anion_nodesxsample)

    anion_node_mask = torch.zeros(batch_size, anion_max_n_nodes)
    cation_node_mask = torch.zeros(batch_size, cation_max_n_nodes)

    for i in range(batch_size):
        anion_node_mask[i, 0: anion_nodesxsample[i]] = 1
        cation_node_mask[i, 0: cation_nodesxsample[i]] = 1

    # Compute edge_mask
    anion_edge_mask = anion_node_mask.unsqueeze(1) * anion_node_mask.unsqueeze(2)
    anion_diag_mask = ~torch.eye(anion_edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    anion_edge_mask *= anion_diag_mask
    anion_edge_mask = anion_edge_mask.view(batch_size * anion_max_n_nodes * anion_max_n_nodes, 1).to(device)
    anion_node_mask = anion_node_mask.unsqueeze(2).to(device)

    cation_edge_mask = cation_node_mask.unsqueeze(1) * cation_node_mask.unsqueeze(2)
    cation_diag_mask = ~torch.eye(cation_edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    cation_edge_mask *= cation_diag_mask
    cation_edge_mask = cation_edge_mask.view(batch_size * cation_max_n_nodes * cation_max_n_nodes, 1).to(device)
    cation_node_mask = cation_node_mask.unsqueeze(2).to(device)

    anion_context, cation_context = prop_dist.sample_batch(anion_nodesxsample, cation_nodesxsample)
    anion_context = anion_context.unsqueeze(1).repeat(1, anion_max_n_nodes, 1).to(device) * anion_node_mask
    cation_context = cation_context.unsqueeze(1).repeat(1, cation_max_n_nodes, 1).to(device) * cation_node_mask
    anion_x, anion_h = generative_model.anion_evd.sample(
        batch_size, anion_max_n_nodes, anion_node_mask, anion_edge_mask, anion_context, fix_noise=fix_noise)
    cation_x, cation_h = generative_model.cation_evd.sample(
        batch_size, cation_max_n_nodes, cation_node_mask, cation_edge_mask, cation_context, fix_noise=fix_noise)

    assert_correctly_masked(anion_x, anion_node_mask)
    assert_mean_zero_with_mask(anion_x, anion_node_mask)
    anion_one_hot = anion_h['categorical']
    anion_charges = anion_h['integer']
    assert_correctly_masked(anion_one_hot.float(), anion_node_mask)

    assert_correctly_masked(cation_x, cation_node_mask)
    assert_mean_zero_with_mask(cation_x, cation_node_mask)
    cation_one_hot = cation_h['categorical']
    cation_charges = cation_h['integer']
    assert_correctly_masked(cation_one_hot.float(), cation_node_mask)

    if args.include_charges:
        assert_correctly_masked(cation_charges.float(), cation_node_mask)
    return (anion_one_hot, anion_charges, anion_x, anion_node_mask,
            cation_one_hot, cation_charges, cation_x, cation_node_mask)


def sample_separate(args, device, generative_model,
                    prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
                    fix_noise=False):
    max_n_nodes = anion_data_info['max_n_nodes'] if args.ion == "anion" else cation_data_info['max_n_nodes']

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)

    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1

    # Compute edge_mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    if context is None:
        anion_context, cation_context = prop_dist.sample_batch(nodesxsample, nodesxsample)
        context = anion_context if args.ion == "anion" else cation_context
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask

    x, h = generative_model.sample(
        batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)
    one_hot = h['categorical']
    charges = h['integer']

    return one_hot, charges, x, node_mask


def sample_sweep_conditional(args, device, generative_model, prop_dist,
                             anion_n_nodes=10, cation_n_nodes=10, n_frames=100):
    anion_nodesxsample = torch.tensor([anion_n_nodes] * n_frames)
    cation_nodesxsample = torch.tensor([cation_n_nodes] * n_frames)
    """
    anion_context = []
    cation_context = []
    for key in prop_dist.anion_distributions:

        min_val, max_val = prop_dist.anion_distributions[key][anion_n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / mad
        max_val = (max_val - mean) / mad
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)

        context_row = torch.tensor(np.linspace(-1, 1, n_frames)).unsqueeze(1)
        anion_context.append(context_row)
    anion_context = torch.cat(anion_context, dim=1).float().to(device)
    for key in prop_dist.cation_distributions:

        min_val, max_val = prop_dist.anion_distributions[key][anion_n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / mad
        max_val = (max_val - mean) / mad
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)

        context_row = torch.tensor(np.linspace(-1, 1, n_frames)).unsqueeze(1)
    cation_context.append(context_row)
    cation_context = torch.cat(cation_context, dim=1).float().to(device)
    """
    (anion_one_hot, anion_charges, anion_x, anion_node_mask,
     cation_one_hot, cation_charges, cation_x, cation_node_mask) = sample(
        args, device, generative_model, prop_dist,
        anion_nodesxsample=anion_nodesxsample, cation_nodesxsample=cation_nodesxsample,
        fix_noise=True)
    return (anion_one_hot, anion_charges, anion_x, anion_node_mask,
            cation_one_hot, cation_charges, cation_x, cation_node_mask)


def sample_sweep_conditional_separate(args, device, generative_model, prop_dist,
                                      n_nodes=10, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)
    (one_hot, charges, x, node_mask) = sample_separate(args, device, generative_model, prop_dist,
                                                       nodesxsample=nodesxsample, fix_noise=True)
    return one_hot, charges, x, node_mask
