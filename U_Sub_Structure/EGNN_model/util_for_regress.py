import numpy as np
import torch
import os
from A_Preprocess.AU_regress_dataset import RegressDataset, RegressDatasetSeparate


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def mean_mad(dataset=None, property_=None):
    if dataset is not None:
        values = dataset.data['value_list']
        mean_value = torch.mean(values)
        ma = torch.abs(values - mean_value)
        mad = torch.mean(ma)
    elif property_ is not None:
        dataset = RegressDatasetSeparate(property_)
        mean_value, mad = mean_mad(dataset)
    return mean_value, mad


def min_max(dataset=None, property_=None):
    if dataset is not None:
        values = dataset.data['value_list']
        min_value = torch.min(values)
        max_value = torch.max(values)
    elif property_ is not None:
        dataset = RegressDatasetSeparate(property_)
        min_value, max_value = mean_mad(dataset)
    return min_value, max_value


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


def mean_mad_t_p():
    if os.path.exists("ZDataB_ProcessedData/T_p_mean_mad.tensor"):
        t_mean, t_mad, p_mean, p_mad = torch.load("ZDataB_ProcessedData/T_p_mean_mad.tensor")
    else:
        data_s = RegressDataset(property_="Solubility")
        data_v = RegressDataset(property_="Viscosity")
        data_h = RegressDataset(property_="Heat_capacity")
        data_d = RegressDataset(property_="Density")
        temperature = torch.cat(
            (data_s.data['temperature_list'],
             data_v.data['temperature_list'],
             data_h.data['temperature_list'],
             data_d.data['temperature_list']), dim=0)
        pressure = torch.cat(
            (data_s.data['pressure_list'],
             data_v.data['pressure_list'],
             data_h.data['pressure_list'],
             data_d.data['pressure_list']), dim=0)
        t_mean = torch.mean(temperature)
        p_mean = torch.mean(pressure)
        t_mad = torch.mean(torch.abs(temperature - t_mean))
        p_mad = torch.mean(torch.abs(pressure - p_mean))
        torch.save([t_mean, t_mad, p_mean, p_mad], "ZDataB_ProcessedData/T_p_mean_mad.tensor")
    print([t_mean, t_mad, p_mean, p_mad])
    return t_mean, t_mad, p_mean, p_mad


def min_max_t_p():
    if os.path.exists("ZDataB_ProcessedData/T_p_min_max.tensor"):
        t_min, t_max, p_min, p_max = torch.load("ZDataB_ProcessedData/T_p_min_max.tensor")
    else:
        data_s = RegressDataset(property_="Solubility")
        data_v = RegressDataset(property_="Viscosity")
        data_h = RegressDataset(property_="Heat_capacity")
        data_d = RegressDataset(property_="Density")
        temperature = torch.cat(
            (data_s.data['temperature_list'],
             data_v.data['temperature_list'],
             data_h.data['temperature_list'],
             data_d.data['temperature_list']), dim=0)
        pressure = torch.cat(
            (data_s.data['pressure_list'],
             data_v.data['pressure_list'],
             data_h.data['pressure_list'],
             data_d.data['pressure_list']), dim=0)
        t_min = torch.min(temperature)
        p_min = torch.min(pressure)
        t_max = torch.max(temperature)
        p_max = torch.max(pressure)
        torch.save([t_min, t_max, p_min, p_max], "ZDataB_ProcessedData/T_p_min_max.tensor")
    print([t_min, t_max, p_min, p_max])
    return t_min, t_max, p_min, p_max


