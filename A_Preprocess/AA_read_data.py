import os
import json
import pickle
import numpy as np
import re
from tqdm import tqdm, trange
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"
co2_x = 44
atom_dict = {'5': 0,
             '6': 1,
             '7': 2,
             '8': 3,
             '9': 4,
             '15': 5,
             '16': 6,
             '17': 7,
             '35': 8,
             '53': 9}

atom_name_dict = {'B': 5,
                  'C': 6,
                  'N': 7,
                  'O': 8,
                  'F': 9,
                  'P': 15,
                  'S': 16,
                  'Cl': 17,
                  'Br': 35,
                  'I': 53}

# NOTE 隐式氢的效果更好
# NOTE 最多有101个原子
max_atom_num = 101


class DataPoint:
    def __init__(self, temperature_, pressure_, value_, ionic_liquid_):
        self.temperature = temperature_
        self.pressure = pressure_
        self.value = value_
        self.ionic_liquid = ionic_liquid_


class IonicLiquid:
    def __init__(self, smiles_):
        mol = Chem.MolFromSmiles(smiles_)
        self.smiles = smiles_
        self.num_atoms = mol.GetNumAtoms()
        self.mol_charge = np.zeros(max_atom_num)
        self.mol_coordinate = np.zeros((max_atom_num, 3))
        mol = Chem.AddHs(mol)
        Chem.Kekulize(mol)
        self.return_value = AllChem.EmbedMolecule(mol)
        statis = Chem.MolToMolBlock(mol)
        coord = re.findall(r"(-?\d.\d+)\s+(-?\d.\d+)\s+(-?\d.\d+)\s(\w)\s+", statis)
        mol_coordinate = []
        mol_charge = []
        one_hot = []
        for i in range(len(coord)):
            if coord[i][-1] != 'H':
                mol_coordinate.append(coord[i][:-1])
                mol_charge.append(atom_name_dict[coord[i][-1]])
                one_hot_slice = np.zeros(10)
                one_hot_slice[atom_dict[str(atom_name_dict[coord[i][-1]])]] = 1
                one_hot.append(one_hot_slice)

        self.mol_charge = np.array(mol_charge)
        self.one_hot = np.array(one_hot)
        self.mol_coordinate = np.array(mol_coordinate).astype(np.float64)
        self.mol_charge = np.pad(self.mol_charge, (0, max_atom_num - len(self.mol_charge)),
                                 'constant', constant_values=0)
        self.one_hot = np.pad(self.one_hot, ((0, max_atom_num - len(self.one_hot)), (0, 0)),
                              'constant', constant_values=0)
        self.mol_coordinate = np.pad(self.mol_coordinate,
                                     ((0, max_atom_num - len(self.mol_coordinate)), (0, 0)),
                                     'constant', constant_values=0)

        self.atom_mask = self.mol_charge != 0
        self.atom_mask.astype(np.float64)
        self.edge_mask = np.ones((max_atom_num, max_atom_num)) - np.eye(max_atom_num)
        self.edge_mask[self.num_atoms:, self.num_atoms:] = 0


def unpickle_ionic_liquid_smiles_list():
    with open("ZDataB_ProcessedData/ionic_liquid_list.data", "rb") as f_:
        return pickle.load(f_)


def pickle_ionic_liquid_list(ionic_liquid_list_):
    with open("ZDataB_ProcessedData/ionic_liquid_list.data", "wb") as f_:
        pickle.dump(ionic_liquid_list_, f_)


def unpickle_ion_smiles_list(ion_type):
    if ion_type == "Anion":
        with open("ZDataB_ProcessedData/anion_smiles_list.data", "rb") as f_:
            pickled = pickle.load(f_)
    elif ion_type == "Cation":
        with open("ZDataB_ProcessedData/cation_smiles_list.data", "rb") as f_:
            pickled = pickle.load(f_)
    return pickled


def pickle_ion_smiles_list(ion_type, ion_list):
    if ion_type == "Anion":
        with open("ZDataB_ProcessedData/anion_smiles_list.data", "wb") as f_:
            pickle.dump(ion_list, f_)
    elif ion_type == "Cation":
        with open("ZDataB_ProcessedData/cation_smiles_list.data", "wb") as f_:
            pickle.dump(ion_list, f_)


def unpickle_smiles_list(ion_type):
    ion_smiles_list = []
    ion_list = unpickle_ion_smiles_list(ion_type)
    for ion in ion_list:
        ion_smiles_list.append(ion.smiles)
    return ion_smiles_list


def unpickle_data_point_list(property_):
    with open("ZDataB_ProcessedData/property_" + property_ + "_data_point_list.data", "rb") as f_:
        return pickle.load(f_)


def pickle_data_point_list(property_, data_point_list_):
    pickled_data_point_list_ = []
    for data_point in data_point_list_:
        pickled_data_point_list_.append(
            [
                data_point.temperature,
                data_point.pressure,
                data_point.value,
                data_point.ionic_liquid
            ]
        )
    with open("ZDataB_ProcessedData/property_" + property_ + "_data_point_list.data", "wb") as f_:
        pickle.dump(pickled_data_point_list_, f_)


def unpickle_dataset(property_):
    return np.load("ZDataB_ProcessedData/property_" + property_ + "_dataset.npz")


def unpickle_separate_dataset(property_):
    return np.load("ZDataB_ProcessedData/property_" + property_ + "_separated_dataset.npz")


def load_data_origin():
    print("Start: Loading origin data from Data/origin_data_list.data")
    try:
        with open("ZDataB_ProcessedData/origin_data_list.data", "rb") as f_:
            data_list__ = pickle.load(f_)
    except IOError:
        print("Warning: File origin_data_list.data not found, reinitializing")
        try:
            with open("ZDataA_InputData/data_2023Sep.txt") as f_:
                data_list_ = json.loads(f_.read())
        except IOError:
            print("Error: File ZDataA_InputData/data_2023Sep.txt not found")
            sys.exit()
        else:
            with open("ZDataB_ProcessedData/origin_data_list.data", "wb") as f_:
                pickle.dump(data_list_, f_)
                print("Finish: Saving origin data from Data/data_2021Jul.txt")
        with open("ZDataB_ProcessedData/origin_data_list.data", "rb") as f_:
            data_list__ = pickle.load(f_)
    print("Finish: Loading origin data from ZDataB_ProcessedData/origin_data_list.data")

    return data_list__


def name_to_il(name_smiles_list_, name):
    for il in name_smiles_list_:
        if name == il[0]:
            return il[1] + '.' + il[2]
    return -1


def prepare_ionic_liquid_list():
    data_list_ = load_data_origin()
    component_name_list = []

    for data in data_list_:
        for component in data[1]["components"]:
            if component["name"] not in component_name_list:
                component_name_list.append(component["name"])

    with open("ZDataB_ProcessedData/component_name_list.txt", "w") as f_:
        for item in component_name_list:
            f_.write(item + "\n")
    """
    os.system("java -jar A_Preprocess/opsin.jar -osmi 
    ZDataB_ProcessedData/component_name_list.txt 
    ZDataB_ProcessedData/component_smiles_list.txt")
    """

    name_smiles_list_ = []
    ionic_liquid_smiles_list_ = []
    anion_smiles_list = []
    cation_smiles_list = []
    with open("ZDataB_ProcessedData/component_smiles_list.txt", "r") as f_:
        component_smiles_list = f_.readlines()
    with open("ZDataB_ProcessedData/component_name_list.txt", "r") as f_:
        component_name_list = f_.readlines()
    for i_ in trange(len(component_name_list)):
        component_name_list[i_] = component_name_list[i_].replace("\n", "")
        component_smiles_list[i_] = component_smiles_list[i_].replace("\n", "")

        if (
                component_smiles_list[i_].count(".") == 1
                and "Ga" not in component_smiles_list[i_]
                and "Re" not in component_smiles_list[i_]
                and "Al" not in component_smiles_list[i_]
                and "As" not in component_smiles_list[i_]
                and "Sb" not in component_smiles_list[i_]
                and len(component_smiles_list[i_].split(".")[0]) > 6
                and len(component_smiles_list[i_].split(".")[1]) > 6
        ):
            part_1 = component_smiles_list[i_].split(".")[0]
            part_2 = component_smiles_list[i_].split(".")[1]
            if "-" in part_1:
                if part_1 not in anion_smiles_list:
                    anion_smiles_list.append(part_1)
                if part_2 not in cation_smiles_list:
                    cation_smiles_list.append(part_2)
                name_smiles_list_.append([component_name_list[i_], part_1, part_2])
            else:
                if part_1 not in cation_smiles_list:
                    cation_smiles_list.append(part_1)
                if part_2 not in anion_smiles_list:
                    anion_smiles_list.append(part_2)
                name_smiles_list_.append([component_name_list[i_], part_2, part_1])

    with open("ZDataB_ProcessedData/name_smiles_list.data", "wb") as f__:
        pickle.dump(name_smiles_list_, f__)

    print(
        "Notice: Totally %d anions and %d cations to be added to ILs list"
        % (len(anion_smiles_list), len(cation_smiles_list))
    )
    anion_smiles_list = list(set(anion_smiles_list))
    cation_smiles_list = list(set(cation_smiles_list))

    for anion_smiles in anion_smiles_list:
        for cation_smiles in cation_smiles_list:
            ionic_liquid_smiles_list_.append(cation_smiles + "." + anion_smiles)

    pickle_ion_smiles_list("Anion", anion_smiles_list)
    pickle_ion_smiles_list("Cation", cation_smiles_list)
    pickle_ionic_liquid_list(ionic_liquid_smiles_list_)


def prepare_solubility_data_classified():
    print("Start: Classifying origin data")
    data_list_ = load_data_origin()
    with open("ZDataB_ProcessedData/name_smiles_list.data", "rb") as f_:
        name_smiles_list_ = pickle.load(f_)

    # TODO 需要以下性质信息：粘度，密度，二氧化碳溶解度，比热容（定压）
    equilibrium_pressure_list = []
    weight_fraction_list = []
    henry_constant_mole_fraction_list = []
    mole_fraction_list = []

    for data_ in tqdm(data_list_):
        if len(data_[1]["components"]) == 2 and data_[1]["solvent"] is None:
            if (
                    data_[1]["components"][0]["formula"] == "CO<SUB>2</SUB>"
                    or data_[1]["components"][1]["formula"] == "CO<SUB>2</SUB>"
            ):
                if (
                        data_[1]["title"]
                        == "Phase transition properties: Equilibrium pressure"
                ):  # used
                    equilibrium_pressure_list.append(data_)
                elif (
                        data_[1]["title"]
                        == "Composition at phase equilibrium: Henry's Law constant for mole fraction of component"
                ):  # used
                    henry_constant_mole_fraction_list.append(data_)
                elif (
                        data_[1]["title"]
                        == "Composition at phase equilibrium: Weight fraction of component"
                ):  # used
                    weight_fraction_list.append(data_)
                elif (
                        data_[1]["title"]
                        == "Composition at phase equilibrium: Mole fraction of component"
                ):  # used
                    mole_fraction_list.append(data_)
    data_point_list_ = []

    for data_ in mole_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]["components"]:
            if component["name"] != "carbon dioxide":
                ionic_liquid_name_ = component["name"]
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print(
                "Data structure error at mole_fraction_list index "
                + str(mole_fraction_list.index(data_))
            )
            continue

        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue

        if any(
                "Mole fraction of carbon dioxide" in element_
                for element_ in data_[1]["dhead"]
        ):
            temperature_index, value_index, pressure_index = None, None, None
            for i in range(len(data_[1]["dhead"])):
                try:
                    if "Temperature" in data_[1]["dhead"][i][0]:
                        temperature_index = i
                    elif "Mole fraction of carbon dioxide" in data_[1]["dhead"][i][0]:
                        value_index = i
                    elif (
                            "Pressure" in data_[1]["dhead"][i][0]
                            and "kPa" in data_[1]["dhead"][i][0]
                    ):
                        pressure_index = i
                except TypeError:
                    print(data_[1]["dhead"])

            try:
                assert (
                        temperature_index is not None
                        and value_index is not None
                        and pressure_index is not None
                        and ionic_liquid_name_ is not None
                )
            except AssertionError:
                print(data_[1]["dhead"])
                print('Data structure error at mole_fraction_list index' + str(
                    mole_fraction_list.index(data_)))
                continue
            for point in data_[1]["data"]:
                temperature_ = float(point[temperature_index][0])
                pressure_ = float(point[pressure_index][0])
                mole_fraction_ = float(point[value_index][0])
                data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=mole_fraction_,
                                        ionic_liquid_=ionic_liquid_)
                data_point_list_.append(data_point_)

    for data_ in equilibrium_pressure_list:
        ionic_liquid_name_ = None
        for component in data_[1]["components"]:
            if component["name"] != "carbon dioxide":
                ionic_liquid_name_ = component["name"]
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print(
                "Data structure error at equilibrium_pressure_list index"
                + str(equilibrium_pressure_list.index(data_))
            )
            continue

        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue

        if any(
                "Mole fraction of carbon dioxide" in element_
                for element_ in data_[1]["dhead"]
        ):
            temperature_index, value_index, pressure_index = None, None, None
            for i in range(len(data_[1]["dhead"])):
                if any("Temperature" in dhead for dhead in data_[1]["dhead"][i]):
                    temperature_index = i
                elif any(
                        "Mole fraction of carbon dioxide" in dhead
                        for dhead in data_[1]["dhead"][i]
                ):
                    value_index = i
                elif any(
                        "Equilibrium pressure" in dhead for dhead in data_[1]["dhead"][i]
                ) and any("kPa" in dhead for dhead in data_[1]["dhead"][i]):
                    pressure_index = i

            try:
                assert (
                        temperature_index is not None
                        and value_index is not None
                        and pressure_index is not None
                        and ionic_liquid_name_ is not None
                )
            except AssertionError:
                print(data_[1]["dhead"])
                print(
                    "Data structure error at equilibrium_pressure_list index "
                    + str(equilibrium_pressure_list.index(data_))
                )
                continue
            for point in data_[1]["data"]:
                temperature_ = float(point[temperature_index][0])
                pressure_ = float(point[pressure_index][0])
                mole_fraction_ = float(point[value_index][0])
                data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=mole_fraction_,
                                        ionic_liquid_=ionic_liquid_)
                data_point_list_.append(data_point_)

    for data_ in henry_constant_mole_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]["components"]:
            if component["name"] != "carbon dioxide":
                ionic_liquid_name_ = component["name"]
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print(
                "Data structure error at henry_constant_mole_fraction_list index "
                + str(henry_constant_mole_fraction_list.index(data_))
            )
            continue
        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue
        for point in data_[1]["data"]:
            temperature_ = float(point[0][0])
            pressure_ = 101.325
            mole_fraction_ = pressure_ / float(point[2][0])
            if mole_fraction_ > 1.1:
                continue
            data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=mole_fraction_,
                                    ionic_liquid_=ionic_liquid_)
            data_point_list_.append(data_point_)

    for data_ in weight_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]["components"]:
            if component["name"] != "carbon dioxide":
                ionic_liquid_name_ = component["name"]
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print(
                "Data structure error at weight_fraction_list index "
                + str(weight_fraction_list.index(data_))
            )
            continue
        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        il_mol = Chem.MolFromSmiles(ionic_liquid_)
        il_x = Descriptors.MolWt(il_mol)
        if ionic_liquid_ == -1:
            continue
        for point in data_[1]["data"]:
            temperature_ = float(point[0][0])
            pressure_ = float(point[1][0])
            weight_fraction_ = float(point[2][0])
            mole_fraction_ = (weight_fraction_ / co2_x) / (weight_fraction_ / co2_x + (1 - weight_fraction_) / il_x)
            data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=mole_fraction_,
                                    ionic_liquid_=ionic_liquid_)
            data_point_list_.append(data_point_)

    pickle_data_point_list("Solubility", data_point_list_)
    return 1


def prepare_other_data_classified():
    print("Start: Classifying origin data")
    data_list_ = load_data_origin()
    with open("ZDataB_ProcessedData/name_smiles_list.data", "rb") as f_:
        name_smiles_list_ = pickle.load(f_)

    # TODO 需要以下性质信息：粘度，密度，二氧化碳溶解度，比热容（定压）
    viscosity_list = []
    density_list = []
    heat_capacity_list = []

    for data_ in tqdm(data_list_):
        if len(data_[1]["components"]) == 1 and data_[1]["solvent"] is None:
            if (
                    data_[1]["title"]
                    == "Transport properties: Viscosity"
            ):  # used
                viscosity_list.append(data_)
            elif (
                    data_[1]["title"]
                    == "Volumetric properties: Specific density"
            ):
                density_list.append(data_)
            elif (
                    data_[1]["title"]
                    == "Heat capacity and derived properties: Heat capacity at constant pressure"
            ):
                heat_capacity_list.append(data_)

    viscosity_data_point_list_ = []
    density_data_point_list_ = []
    heat_capacity_data_point_list_ = []

    for data_ in viscosity_list:
        ionic_liquid_name_ = data_[1]["components"][0]["name"]
        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue

        temperature_index, value_index, pressure_index = None, None, None
        for i in range(len(data_[1]["dhead"])):
            try:
                if "Temperature" in data_[1]["dhead"][i][0]:
                    temperature_index = i
                elif "Viscosity" in data_[1]["dhead"][i][0]:
                    value_index = i
                elif (
                        "Pressure" in data_[1]["dhead"][i][0]
                        and "kPa" in data_[1]["dhead"][i][0]
                ):
                    pressure_index = i
            except TypeError:
                print(data_[1]["dhead"])

        try:
            assert (
                    temperature_index is not None
                    and value_index is not None
                    and ionic_liquid_name_ is not None
            )
        except AssertionError:
            print(data_[1]["dhead"])
            print('Data structure error at ' + str(
                viscosity_list.index(data_)))
            continue
        for point in data_[1]["data"]:
            temperature_ = float(point[temperature_index][0])
            pressure_ = float(point[pressure_index][0]) if pressure_index is not None else 101.325
            viscosity_ = float(point[value_index][0])
            data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=viscosity_,
                                    ionic_liquid_=ionic_liquid_)
            viscosity_data_point_list_.append(data_point_)
    pickle_data_point_list("Viscosity", viscosity_data_point_list_)

    for data_ in density_list:
        ionic_liquid_name_ = data_[1]["components"][0]["name"]
        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue

        temperature_index, value_index, pressure_index = None, None, None
        for i in range(len(data_[1]["dhead"])):
            try:
                if "Temperature" in data_[1]["dhead"][i][0]:
                    temperature_index = i
                elif "Specific density" in data_[1]["dhead"][i][0]:
                    value_index = i
                elif (
                        "Pressure" in data_[1]["dhead"][i][0]
                        and "kPa" in data_[1]["dhead"][i][0]
                ):
                    pressure_index = i
            except TypeError:
                print(data_[1]["dhead"])

        try:
            assert (
                    temperature_index is not None
                    and value_index is not None
                    and ionic_liquid_name_ is not None
            )
        except AssertionError:
            print(data_[1]["dhead"])
            print('Data structure error at ' + str(
                density_list.index(data_)))
            continue

        for point in data_[1]["data"]:
            temperature_ = float(point[temperature_index][0])
            pressure_ = float(point[pressure_index][0]) if pressure_index is not None else 101.325
            density_ = float(point[value_index][0])
            data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=density_,
                                    ionic_liquid_=ionic_liquid_)
            density_data_point_list_.append(data_point_)
    pickle_data_point_list("Density", density_data_point_list_)

    for data_ in heat_capacity_list:
        ionic_liquid_name_ = data_[1]["components"][0]["name"]
        ionic_liquid_ = name_to_il(
            name_smiles_list_, ionic_liquid_name_
        )
        if ionic_liquid_ == -1:
            continue

        temperature_index, value_index, pressure_index = None, None, None
        for i in range(len(data_[1]["dhead"])):
            try:
                if "Temperature" in data_[1]["dhead"][i][0]:
                    temperature_index = i
                elif "Heat capacity at constant pressure, J/K/mol" in data_[1]["dhead"][i][0]:
                    value_index = i
                elif (
                        "Pressure" in data_[1]["dhead"][i][0]
                        and "kPa" in data_[1]["dhead"][i][0]
                ):
                    pressure_index = i
            except TypeError:
                print(data_[1]["dhead"])

        try:
            assert (
                    temperature_index is not None
                    and value_index is not None
                    and ionic_liquid_name_ is not None
            )
        except AssertionError:
            print(data_[1]["dhead"])
            print('Data structure error at ' + str(
                heat_capacity_list.index(data_)))
            continue
        for point in data_[1]["data"]:
            temperature_ = float(point[temperature_index][0])
            pressure_ = float(point[pressure_index][0]) if pressure_index is not None else 101.325
            heat_capacity_ = float(point[value_index][0])
            data_point_ = DataPoint(temperature_=temperature_, pressure_=pressure_, value_=heat_capacity_,
                                    ionic_liquid_=ionic_liquid_)
            heat_capacity_data_point_list_.append(data_point_)
    pickle_data_point_list("Heat_capacity", heat_capacity_data_point_list_)


def prepare_regress_dataset():
    il_smiles_cache = []
    il_mol_cache = []
    bad_il_cache = []

    def prepare_regress_dataset_for(property_):
        out_file_path = "ZDataB_ProcessedData/property_" + property_ + "_dataset.npz"
        data_point_list = unpickle_data_point_list(property_)
        charge_list = []
        atom_position_list = []
        node_mask_list = []
        edge_mask_list = []
        n_nodes_list = []
        one_hot_list = []
        temperature_list = []
        pressure_list = []
        value_list = []
        for data_point in tqdm(data_point_list):
            if 70 < data_point[1] < 5000:
                if data_point[3] not in bad_il_cache:
                    if data_point[3] not in il_smiles_cache:
                        il = IonicLiquid(data_point[3])
                        if il.return_value == 0:
                            il_smiles_cache.append(data_point[3])
                            il_mol_cache.append(il)
                        else:
                            bad_il_cache.append(data_point[3])
                            continue
                    else:
                        il = il_mol_cache[il_smiles_cache.index(data_point[3])]
                    charge_list.append(il.mol_charge)
                    atom_position_list.append(il.mol_coordinate)
                    node_mask_list.append(il.atom_mask)
                    edge_mask_list.append(il.edge_mask)
                    n_nodes_list.append(il.num_atoms)
                    one_hot_list.append(il.one_hot)
                    temperature_list.append(data_point[0])
                    pressure_list.append(data_point[1])
                    value_list.append(data_point[2])
        charge_list = np.array(charge_list)
        atom_position_list = np.array(atom_position_list)
        node_mask_list = np.array(node_mask_list)
        edge_mask_list = np.array(edge_mask_list)
        n_nodes_list = np.array(n_nodes_list)
        temperature_list = np.array(temperature_list)
        pressure_list = np.array(pressure_list)
        value_list = np.array(value_list)

        np.savez(out_file_path,
                 charge_list=charge_list,
                 atom_position_list=atom_position_list,
                 node_mask_list=node_mask_list,
                 edge_mask_list=edge_mask_list,
                 n_nodes_list=n_nodes_list,
                 one_hot_list=one_hot_list,
                 temperature_list=temperature_list,
                 pressure_list=pressure_list,
                 value_list=value_list
                 )

    prepare_regress_dataset_for(property_="Solubility")
    prepare_regress_dataset_for(property_="Viscosity")
    prepare_regress_dataset_for(property_="Heat_capacity")
    prepare_regress_dataset_for(property_="Density")
    print(bad_il_cache)


if __name__ == "__main__":
    # prepare_ionic_liquid_list()
    # prepare_solubility_data_classified()
    # prepare_other_data_classified()
    # a = unpickle_data_point_list(property_="Solubility")
    # prepare_regress_dataset()
    # pass
    # NOTE Totally 322 anions and 860 cations to be added to ILs list
    pass
