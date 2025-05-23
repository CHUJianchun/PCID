import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from A_Preprocess.AA_read_data import load_data_origin, name_to_il, co2_x
from tqdm import tqdm, trange
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
import matplotlib.patches as patches
from functools import reduce
from U_Simulation.energy_simulation import energy_cost


data_list_ = load_data_origin()
solubility_list = []
solubility_name_list = []

with open("ZDataB_ProcessedData/name_smiles_list.data", "rb") as f_:
    name_smiles_list_ = pickle.load(f_)

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
                solubility_list.append(data_)
                solubility_name_list.append(
                    data_[1]["components"][0]["name"] if data_[1]["components"][0]["name"] != "carbon dioxide" else
                    data_[1]["components"][1]["name"])
            elif (
                    data_[1]["title"]
                    == "Composition at phase equilibrium: Henry's Law constant for mole fraction of component"
            ):  # used
                solubility_list.append(data_)
                solubility_name_list.append(
                    data_[1]["components"][0]["name"] if data_[1]["components"][0]["name"] != "carbon dioxide" else
                    data_[1]["components"][1]["name"])
            elif (
                    data_[1]["title"]
                    == "Composition at phase equilibrium: Weight fraction of component"
            ):  # used
                solubility_list.append(data_)
                solubility_name_list.append(
                    data_[1]["components"][0]["name"] if data_[1]["components"][0]["name"] != "carbon dioxide" else
                    data_[1]["components"][1]["name"])
            elif (
                    data_[1]["title"]
                    == "Composition at phase equilibrium: Mole fraction of component"
            ):  # used
                solubility_list.append(data_)
                solubility_name_list.append(data_[1]["components"][0]["name"])

solubility_name_list = list(set(solubility_name_list))
full_property_list = []

for name in solubility_name_list:
    solubility_sub_list = []
    for i in range(len(solubility_list)):
        if (solubility_list[i][1]["components"][0]["name"] == name or
                solubility_list[i][1]["components"][1]["name"] == name):
            solubility_sub_list.append(solubility_list[i][1])
    full_property_list.append([
        name,
        solubility_sub_list])

full_reg_prop_list = []
full_reg_name_list = []
full_reg_smiles_list = []

for idx in range(len(full_property_list)):
    full_property = full_property_list[idx]
    name = full_property[0]
    solubility_sub_list = full_property[1]
    data_point_list = []
    ionic_liquid_name_ = None
    for component in solubility_sub_list[0]["components"]:
        if component["name"] != "carbon dioxide":
            ionic_liquid_name_ = component["name"]
            break
    try:
        assert ionic_liquid_name_ is not None
    except AssertionError:
        print("Data structure error at weight_fraction_list index ")
        continue
    ionic_liquid_ = name_to_il(
        name_smiles_list_, ionic_liquid_name_
    )

    if ionic_liquid_ == -1:
        continue
    il_mol = Chem.MolFromSmiles(ionic_liquid_)
    il_x = Descriptors.MolWt(il_mol)

    for solubility_dataframe in solubility_sub_list:
        if (
                solubility_dataframe["title"]
                == "Phase transition properties: Equilibrium pressure"
        ):  # used
            if any(
                    "Mole fraction of carbon dioxide" in element_
                    for element_ in solubility_dataframe["dhead"]
            ):
                temperature_index, value_index, pressure_index = None, None, None
                for i in range(len(solubility_dataframe["dhead"])):
                    if any("Temperature" in dhead for dhead in solubility_dataframe["dhead"][i]):
                        temperature_index = i
                    elif any(
                            "Mole fraction of carbon dioxide" in dhead
                            for dhead in solubility_dataframe["dhead"][i]
                    ):
                        value_index = i
                    elif any(
                            "Equilibrium pressure" in dhead for dhead in solubility_dataframe["dhead"][i]
                    ) and any("kPa" in dhead for dhead in solubility_dataframe["dhead"][i]):
                        pressure_index = i
                try:
                    assert (
                            temperature_index is not None
                            and value_index is not None
                            and pressure_index is not None
                    )
                except AssertionError:
                    print("Data structure error at equilibrium_pressure_list index ")
                    print(solubility_dataframe["dhead"])
                    continue
                for point in solubility_dataframe["data"]:
                    temperature_ = float(point[temperature_index][0])
                    pressure_ = float(
                        point[pressure_index][0]) if float(point[pressure_index][0]) - 101.325 > 0.5 else 101.325
                    mole_fraction_ = float(point[value_index][0])
                    data_point_ = [temperature_, pressure_, mole_fraction_]
                    data_point_list.append(data_point_)
        elif (
                solubility_dataframe["title"]
                == "Composition at phase equilibrium: Henry's Law constant for mole fraction of component"
        ):  # used
            if any(
                    "Mole fraction of carbon dioxide" in element_
                    for element_ in solubility_dataframe["dhead"]
            ):
                temperature_index, value_index = None, None
                for point in solubility_dataframe["data"]:
                    temperature_ = float(point[0][0])
                    pressure_ = 101.325
                    mole_fraction_ = pressure_ / float(point[2][0])
                    if mole_fraction_ > 1.1:
                        continue
                    data_point_ = [temperature_, pressure_, mole_fraction_]
                    data_point_list.append(data_point_)
        elif (
                solubility_dataframe["title"]
                == "Composition at phase equilibrium: Weight fraction of component"
        ):  # used

            for point in solubility_dataframe["data"]:
                temperature_ = float(point[0][0])
                pressure_ = float(point[1][0]) if float(
                    point[1][0]) - 101.325 > 0.5 else 101.325
                weight_fraction_ = float(point[2][0])
                mole_fraction_ = (weight_fraction_ / co2_x) / (weight_fraction_ / co2_x + (1 - weight_fraction_) / il_x)
                data_point_ = [temperature_, pressure_, mole_fraction_]
                data_point_list.append(data_point_)
        elif (
                solubility_dataframe["title"]
                == "Composition at phase equilibrium: Mole fraction of component"
        ):
            if any(
                    "Mole fraction of carbon dioxide" in element_
                    for element_ in solubility_dataframe["dhead"]
            ):
                temperature_index, value_index, pressure_index = None, None, None
                for i in range(len(solubility_dataframe["dhead"])):
                    try:
                        if "Temperature" in solubility_dataframe["dhead"][i][0]:
                            temperature_index = i
                        elif "Mole fraction of carbon dioxide" in solubility_dataframe["dhead"][i][0]:
                            value_index = i
                        elif (
                                "Pressure" in solubility_dataframe["dhead"][i][0]
                                and "kPa" in solubility_dataframe["dhead"][i][0]
                        ):
                            pressure_index = i
                    except TypeError:
                        print(solubility_dataframe["dhead"])

                try:
                    assert (
                            temperature_index is not None
                            and value_index is not None
                            and pressure_index is not None
                    )
                except AssertionError:
                    print('Data structure error at mole_fraction_list index')
                    print(solubility_dataframe["dhead"])
                    continue
                for point in solubility_dataframe["data"]:
                    temperature_ = float(point[temperature_index][0])
                    pressure_ = float(
                        point[pressure_index][0]) if float(point[pressure_index][0]) - 101.325 > 0.5 else 101.325
                    mole_fraction_ = float(point[value_index][0])
                    data_point_ = [temperature_, pressure_, mole_fraction_]
                    data_point_list.append(data_point_)
    if len(data_point_list) > 10:
        data_point_list = np.array(data_point_list)
        if max(data_point_list[:, 0]) - min(data_point_list[:, 0]) < 20:
            if max(data_point_list[:, 1]) - min(data_point_list[:, 1]) < 1000:
                continue
    else:
        continue
    linear_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
    linear_reg.fit(data_point_list[:, :-1], data_point_list[:, -1])
    linear_score = linear_reg.score(data_point_list[:, :-1], data_point_list[:, -1])
    poly_features = PolynomialFeatures(degree=2)
    poly_x = poly_features.fit_transform(data_point_list[:, :-1])
    poly_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
    poly_reg.fit(poly_x, data_point_list[:, -1])
    poly_score = poly_reg.score(poly_x, data_point_list[:, -1])
    if linear_score > poly_score:
        regressed_solubility_335 = linear_reg.predict(np.array([335.64, 1000]).reshape(1, -1))
        regressed_solubility_353 = linear_reg.predict(np.array([353.15, 101.325]).reshape(1, -1))
        regressed_solubility_339 = linear_reg.predict(np.array([339.14, 101.325]).reshape(1, -1))
    else:
        regressed_solubility_335 = poly_reg.predict(
            poly_features.fit_transform(np.array([335.64, 1000]).reshape(1, -1)))
        regressed_solubility_353 = poly_reg.predict(
            poly_features.fit_transform(np.array([353.15, 101.325]).reshape(1, -1)))
        regressed_solubility_339 = poly_reg.predict(
            poly_features.fit_transform(np.array([339.14, 101.325]).reshape(1, -1)))

    if all(np.array([il_x,
                     regressed_solubility_335[0],
                     regressed_solubility_353[0],
                     regressed_solubility_339[0]
                     ]) > 0) and regressed_solubility_335[0] < 1:
        full_reg_prop_list.append([
            il_x,
            regressed_solubility_335[0],
            regressed_solubility_353[0],
            regressed_solubility_339[0]
        ])
        full_reg_name_list.append(ionic_liquid_name_)
        full_reg_smiles_list.append(ionic_liquid_)
full_reg_prop_list = np.array(full_reg_prop_list)

mol_for_df = []
energy_cost_for_df = []
name_for_df = []
smiles_for_df = []
properties_for_df = []

for i in range(len(full_reg_prop_list)):
    mol_for_df.append(
        Chem.MolFromSmiles(full_reg_smiles_list[i]))
    energy_cost_for_df.append(full_reg_prop_list[i, 1])
    smiles_for_df.append(full_reg_smiles_list[i])
    name_for_df.append(full_reg_name_list[i])
    properties_for_df.append(full_reg_prop_list[i])
properties_for_df = np.array(properties_for_df)

df = pd.DataFrame()
df['Name'] = name_for_df
df['Smiles'] = smiles_for_df
df['Molecular Weight'] = properties_for_df[:, 0]
df['Solubility in 335.64K, 3Mpa'] = properties_for_df[:, 1]
df['Solubility in 353.15K'] = properties_for_df[:, 2]
df['Solubility in 339.14K'] = properties_for_df[:, 3]
df['Mass solubility difference'] = (properties_for_df[:, 1] - properties_for_df[:, 2]) * 44 / properties_for_df[:, 0]

df = df.sort_values(by='Mass solubility difference', ascending=False)

PandasTools.AddMoleculeColumnToFrame(df, 'Smiles', 'Figure')
PandasTools.SaveXlsxFromFrame(df, 'ZDataE_AnalysisResult/database_solubility_difference.xlsx', molCol='Figure')
