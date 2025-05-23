import os

import torch
from torch.distributions.categorical import Categorical
import numpy as np
from tqdm import trange

from U_Sub_Structure.EDM_model.En_diffusion import EnVariationalDiffusion, ILEnVariationalDiffusion
from U_Sub_Structure.EGNN_model.EGNN_dynamics import EGNN_dynamics_QM9


"""
def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)
"""


def get_ionic_liquid_model(args, device, anion_dataset_info, cation_dataset_info, dataloader_train):
    anion_histogram = anion_dataset_info['n_nodes']
    anion_in_node_nf = len(anion_dataset_info['atom_decoder']) + int(args.include_charges)
    anion_nodes_dist = DistributionNodes(anion_histogram)

    cation_histogram = cation_dataset_info['n_nodes']
    cation_in_node_nf = len(cation_dataset_info['atom_decoder']) + int(args.include_charges)
    cation_nodes_dist = DistributionNodes(cation_histogram)

    prop_dist = DistributionProperty(dataloader_train)
    if args.condition_time:
        anion_dynamics_in_node_nf = anion_in_node_nf + 1
        cation_dynamics_in_node_nf = cation_in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        anion_dynamics_in_node_nf = anion_in_node_nf
        cation_dynamics_in_node_nf = cation_in_node_nf

    anion_net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=anion_dynamics_in_node_nf, context_node_nf=args.anion_context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    cation_net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=cation_dynamics_in_node_nf, context_node_nf=args.cation_context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = ILEnVariationalDiffusion(
            anion_dynamics=anion_net_dynamics,
            anion_in_node_nf=anion_in_node_nf,
            cation_dynamics=cation_net_dynamics,
            cation_in_node_nf=cation_in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )

        return vdm, anion_nodes_dist, cation_nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_ion_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)
    prop_dist = DistributionProperty(dataloader_train, ion=args.ion)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )

        return vdm, nodes_dist, prop_dist
    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)
    return optim


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, num_bins=1000, ion=None):
        self.num_bins = num_bins
        self.ion = ion
        anion_distribution_path = "ZDataB_ProcessedData/Diffusion_Cache/anion_distributions.tensor"
        cation_distribution_path = "ZDataB_ProcessedData/Diffusion_Cache/cation_distributions.tensor"
        if ion is None:
            self.anion_distributions = {}
            self.cation_distributions = {}
            if os.access(anion_distribution_path, os.F_OK):
                self.anion_distributions = torch.load(anion_distribution_path)
                self.cation_distributions = torch.load(cation_distribution_path)
            else:
                for i_ in range(dataloader.dataset.dataset.labels.shape[1]):
                    self.anion_distributions[i_] = {}
                    self.cation_distributions[i_] = {}
                    self._create_prob_dist(dataloader, i_)
                torch.save(self.anion_distributions, anion_distribution_path)
                torch.save(self.cation_distributions, cation_distribution_path)
        else:
            if ion == "anion":
                self.anion_distributions = {}

                if os.access(anion_distribution_path, os.F_OK):
                    self.anion_distributions = torch.load(anion_distribution_path)
                else:
                    for i_ in range(dataloader.dataset.dataset.labels.shape[1]):
                        self.anion_distributions[i_] = {}
                        self._create_prob_dist(dataloader, i_, ion)
                    torch.save(self.anion_distributions, anion_distribution_path)
            else:
                self.cation_distributions = {}
                if os.access(cation_distribution_path, os.F_OK):
                    self.cation_distributions = torch.load(cation_distribution_path)
                else:
                    for i_ in range(dataloader.dataset.dataset.labels.shape[1]):
                        self.cation_distributions[i_] = {}
                        self._create_prob_dist(dataloader, i_, ion)
                    torch.save(self.cation_distributions, cation_distribution_path)

    def _create_prob_dist(self, dataloader, i_, ion=None):
        values = dataloader.dataset.dataset.labels[:, i_]
        # values.shape is 273171,
        if ion == "anion":
            anion_nodes_arr = dataloader.dataset.dataset.data['anion_n_nodes_list']
            anion_min_nodes, anion_max_nodes = torch.min(anion_nodes_arr), torch.max(anion_nodes_arr)
            for n_nodes in trange(int(anion_min_nodes), int(anion_max_nodes) + 1):

                idxs = anion_nodes_arr == n_nodes
                values_filtered = values[idxs]
                if len(values_filtered) > 0:
                    probs, params = self._create_prob_given_nodes(values_filtered)
                    self.anion_distributions[i_][n_nodes] = {'probs': probs, 'params': params}
                """
                
                """
        elif ion == "cation":
            cation_nodes_arr = dataloader.dataset.dataset.data['cation_n_nodes_list']
            cation_min_nodes, cation_max_nodes = torch.min(cation_nodes_arr), torch.max(cation_nodes_arr)
            for n_nodes in trange(int(cation_min_nodes), int(cation_max_nodes) + 1):
                idxs = cation_nodes_arr == n_nodes
                values_filtered = values[idxs]
                if len(values_filtered) > 0:
                    probs, params = self._create_prob_given_nodes(values_filtered)
                    self.cation_distributions[i_][n_nodes] = {'probs': probs, 'params': params}
                """
                idxs = cation_nodes_arr == n_nodes
                cation_idxs = dataloader.dataset.dataset.data['cation_id'][idxs]
                values_filtered = torch.tensor(
                    [values[i__] for i__ in range(len(values)) if dataloader.dataset.dataset.labels[i__, 1] in cation_idxs]
                )
    
                if len(values_filtered) > 0:
                    probs, params = self._create_prob_given_nodes(values_filtered)
                    self.cation_distributions[i_][n_nodes] = {'probs': probs, 'params': params}
                """

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i__ = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i__ == n_bins:
                i__ = n_bins - 1
            histogram[i__] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs.clone().detach())
        params = [prop_min, prop_max]
        return probs, params

    def sample(self, anion_n_nodes, cation_n_nodes):
        anion_vals = []
        cation_vals = []
        if hasattr(self, 'anion_distributions'):
            for i__ in range(len(self.anion_distributions)):
                dist = self.anion_distributions[i__][anion_n_nodes]
                idx = dist['probs'].sample((1,))
                val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
                anion_vals.append(val)
            anion_vals = torch.cat(anion_vals)
        elif hasattr(self, 'cation_distributions'):
            for i__ in range(len(self.cation_distributions)):
                dist = self.cation_distributions[i__][cation_n_nodes]
                idx = dist['probs'].sample((1,))
                val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
                cation_vals.append(val)
            cation_vals = torch.cat(cation_vals)
        return anion_vals, cation_vals

    def sample_batch(self, anion_nodesxsample, cation_nodesxsample):
        anion_vals_list = []
        cation_vals_list = []
        if self.ion == "anion":
            for i__ in range(len(anion_nodesxsample)):
                anion_vals, cation_vals = self.sample(int(anion_nodesxsample[i__]), int(cation_nodesxsample[i__]))
                anion_vals = anion_vals.unsqueeze(0)
                anion_vals_list.append(anion_vals)
            anion_vals_list = torch.cat(anion_vals_list, dim=0)
        elif self.ion == "cation":
            for i__ in range(len(cation_nodesxsample)):
                anion_vals, cation_vals = self.sample(int(anion_nodesxsample[i__]), int(cation_nodesxsample[i__]))
                cation_vals = cation_vals.unsqueeze(0)
                cation_vals_list.append(cation_vals)
            cation_vals_list = torch.cat(cation_vals_list, dim=0)
        elif self.ion is None:
            for i__ in range(len(anion_nodesxsample)):
                anion_vals, cation_vals = self.sample(int(anion_nodesxsample[i__]), int(cation_nodesxsample[i__]))
                anion_vals = anion_vals.unsqueeze(0)
                cation_vals = cation_vals.unsqueeze(0)
                anion_vals_list.append(anion_vals)
                cation_vals_list.append(cation_vals)
            anion_vals_list = torch.cat(anion_vals_list, dim=0)
            cation_vals_list = torch.cat(cation_vals_list, dim=0)
        return anion_vals_list, cation_vals_list

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
