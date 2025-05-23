from A_Preprocess.AA_read_data_seperate_ion import *
import rdkit.Chem as Chem
import numpy as np
from collections import Counter

"""
a = unpickle_ionic_liquid_smiles_list()
max_c = 0
max_a = 0
for smiles in a:
    cation = smiles.split(".")[0]
    anion = smiles.split(".")[1]
    if '+' in cation:
        max_c = max(max_c, Chem.MolFromSmiles(cation).GetNumAtoms())
    if '-' in anion:
        max_a = max(max_a, Chem.MolFromSmiles(anion).GetNumAtoms())
"""

"""
a = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
b = a['anion_n_nodes_list']
b = np.sort(b)
atom_count = Counter(b)
"""

"""
a = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
b = a['cation_n_nodes_list']
b = np.sort(b)
atom_count = Counter(b)
"""

"""
a = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
b = a['anion_one_hot_list']
c = b.reshape(-1,10).sum(axis=0)
"""

a = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
b = a['cation_one_hot_list']
c = b.reshape(-1, 10).sum(axis=0)
