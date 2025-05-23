import torch


def compute_mean_mad(dataloaders):
    return compute_mean_mad_from_dataloader(dataloaders['train'])


def compute_mean_mad_from_dataloader(dataloader):
    property_norms = {}
    for property_key in range(dataloader.dataset.dataset.labels.shape[1]):
        values = dataloader.dataset.dataset.labels[:, property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(minibatch, ion=None):
    if ion == "anion":
        """ANION"""
        batch_size, anion_n_nodes, _ = minibatch['anion_atom_position_list'].size()
        anion_node_mask = minibatch['anion_node_mask_list'].unsqueeze(2)
        anion_context_node_nf = minibatch['labels'].shape[1]
        context_list = []

        for i in range(anion_context_node_nf):
            properties = minibatch['labels'][:, i]
            reshaped = properties.view(batch_size, 1, 1).repeat(1, anion_n_nodes, 1)
            context_list.append(reshaped)
        # Concatenate
        context = torch.cat(context_list, dim=2)
        # Mask disabled nodes!
        anion_context = context * anion_node_mask
        return anion_context
    elif ion == "cation":
        """CATION"""
        batch_size, cation_n_nodes, _ = minibatch['cation_atom_position_list'].size()
        cation_node_mask = minibatch['cation_node_mask_list'].unsqueeze(2)
        context_list = []
        cation_context_node_nf = minibatch['labels'].shape[1]
        for i in range(cation_context_node_nf):
            properties = minibatch['labels'][:, i]
            reshaped = properties.view(batch_size, 1, 1).repeat(1, cation_n_nodes, 1)
            context_list.append(reshaped)
        # Concatenate
        context = torch.cat(context_list, dim=2)
        # Mask disabled nodes!
        cation_context = context * cation_node_mask
        return cation_context
    elif ion is None:
        """ANION"""
        batch_size, anion_n_nodes, _ = minibatch['anion_atom_position_list'].size()
        anion_node_mask = minibatch['anion_node_mask_list'].unsqueeze(2)
        anion_context_node_nf = int((minibatch['labels'].shape[1] - 2) / 3)
        context_list = []

        for i in range(2, anion_context_node_nf + 2):
            properties = minibatch['labels'][:, i]
            reshaped = properties.view(batch_size, 1, 1).repeat(1, anion_n_nodes, 1)
            context_list.append(reshaped)
        # Concatenate
        context = torch.cat(context_list, dim=2)
        # Mask disabled nodes!
        anion_context = context * anion_node_mask

        """CATION"""
        batch_size, cation_n_nodes, _ = minibatch['cation_atom_position_list'].size()
        cation_node_mask = minibatch['cation_node_mask_list'].unsqueeze(2)
        context_list = []
        cation_context_node_nf = minibatch['labels'].shape[1] - anion_context_node_nf - 2
        for i in range(2 + anion_context_node_nf, 2 + anion_context_node_nf + cation_context_node_nf):
            properties = minibatch['labels'][:, i]
            reshaped = properties.view(batch_size, 1, 1).repeat(1, cation_n_nodes, 1)
            context_list.append(reshaped)
        # Concatenate
        context = torch.cat(context_list, dim=2)
        # Mask disabled nodes!
        cation_context = context * cation_node_mask

        return anion_context, cation_context
