from torch.utils.data import Dataset
import torch
from B_Train.BA_regress.BAU_regress_parser import init_parser
import numpy as np

args = init_parser()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32


class PropertyDiffusionDataset(Dataset):
    def __init__(self):
        self.data = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
        self.data = {key: torch.tensor(val) for key, val in self.data.items() if 'smiles' not in key}
        self.labels = torch.load("ZDataB_ProcessedData/all_il_property_list.tensor")
        self.data = {**self.data,
                     **{'anion_id': torch.arange(start=0, end=len(self.data["anion_charge_list"])),
                        'cation_id': torch.arange(start=0, end=len(self.data["cation_charge_list"]))}
                     }
        anion_id = torch.repeat_interleave(
            torch.arange(0, len(self.data["anion_charge_list"])), repeats=len(self.data["cation_charge_list"]),
            dim=0
        ).unsqueeze(-1)
        cation_id = torch.arange(0, len(self.data["cation_charge_list"])).repeat(
            len(self.data["anion_charge_list"])).unsqueeze(-1)
        self.labels = torch.cat((anion_id, cation_id, self.labels), dim=1)

    def __len__(self):
        return len(self.data["anion_charge_list"]) * len(self.data["cation_charge_list"])

    def __getitem__(self, idx):
        labels = {'labels': self.labels[idx]}
        anion_dict = {key: val[labels['labels'][0].to(torch.int)] for key, val in self.data.items() if
                      "anion" in key}
        cation_dict = {key: val[labels['labels'][1].to(torch.int)] for key, val in self.data.items() if
                       "cation" in key}

        return {**anion_dict, **cation_dict, **labels}


class EmbeddingDiffusionDataset(Dataset):
    def __init__(self):
        self.data = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
        self.data = {key: torch.tensor(val) for key, val in self.data.items()}
        self.labels = torch.load("ZDataB_ProcessedData/all_il_embedding_list.tensor")
        self.data = {**self.data,
                     **{'anion_id': torch.arange(start=0, end=len(self.data["anion_charge_list"])),
                        'cation_id': torch.arange(start=0, end=len(self.data["cation_charge_list"]))}
                     }
        anion_id = torch.repeat_interleave(
            torch.arange(0, len(self.data["anion_charge_list"])), repeats=len(self.data["cation_charge_list"]),
            dim=0
        ).unsqueeze(-1)
        cation_id = torch.arange(0, len(self.data["cation_charge_list"])).repeat(
            len(self.data["anion_charge_list"])).unsqueeze(-1)
        self.labels = torch.cat((anion_id, cation_id, self.labels), dim=1)

    def __len__(self):
        return len(self.data["anion_charge_list"]) * len(self.data["cation_charge_list"])

    def __getitem__(self, idx):
        labels = {'labels': self.labels[idx]}
        anion_dict = {key: val[labels['labels'][0].to(torch.int)] for key, val in self.data.items() if
                      "anion" in key}
        cation_dict = {key: val[labels['labels'][1].to(torch.int)] for key, val in self.data.items() if
                       "cation" in key}

        return {**anion_dict, **cation_dict, **labels}


class SeparateEmbeddingDiffusionDataset(Dataset):
    def __init__(self, ion):
        self.ion = ion
        data = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
        self.data = {key: torch.tensor(val) for key, val in data.items() if 'smiles' not in key}
        self.data['anion_smiles_list'] = data['anion_smiles_list']
        self.data['cation_smiles_list'] = data['cation_smiles_list']
        self.labels = torch.load("ZDataB_ProcessedData/Embedding_List/" + ion + "_embedding_list.tensor")

    def __len__(self):
        return len(self.data[self.ion + "_charge_list"])

    def __getitem__(self, idx):
        labels = {'labels': self.labels[idx]}
        data = {key: val[idx] for key, val in self.data.items() if self.ion in key}
        return {**data, **labels}
