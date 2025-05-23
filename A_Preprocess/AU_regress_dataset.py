from torch.utils.data import Dataset
import torch
from A_Preprocess.AA_read_data import unpickle_dataset, unpickle_separate_dataset
from B_Train.BA_regress.BAU_regress_parser import init_parser
import numpy as np

args = init_parser()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32


class RegressDataset(Dataset):
    def __init__(self, property_, tensor=True):
        data = unpickle_dataset(property_)
        self.data = {key: val for key, val in data.items()}
        if tensor:
            for k in self.data.keys():
                self.data[k] = torch.tensor(self.data[k]).to(dtype)

    def __len__(self):
        return len(self.data['value_list'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


class RegressDatasetSeparate(Dataset):
    def __init__(self, property_, tensor=True):
        data = unpickle_separate_dataset(property_)
        self.data = {key: val for key, val in data.items()}
        if tensor:
            for k in self.data.keys():
                self.data[k] = torch.tensor(self.data[k]).to(dtype)

    def __len__(self):
        return len(self.data['value_list'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


class AllILDataset(Dataset):
    def __init__(self):
        self.data = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
        self.data = {key: torch.tensor(val) for key, val in self.data.items() if 'smiles' not in key}

    def __len__(self):
        return len(self.data["anion_charge_list"]) * len(self.data["cation_charge_list"])

    def __getitem__(self, idx):
        anion_dict = {key: val[int(idx / len(self.data["cation_charge_list"]))] for key, val in self.data.items() if
                      "anion" in key}
        cation_dict = {key: val[idx % len(self.data["cation_charge_list"])] for key, val in self.data.items() if
                       "cation" in key}
        return dict(anion_dict, **cation_dict)
