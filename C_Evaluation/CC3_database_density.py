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
density_list = []
density_name_list = []

with open("ZDataB_ProcessedData/name_smiles_list.data", "rb") as f_:
    name_smiles_list_ = pickle.load(f_)

for data_ in tqdm(data_list_):
    if len(data_[1]["components"]) == 1 and data_[1]["solvent"] is None:
        if (
                data_[1]["title"]
                == "Volumetric properties: Specific density"
        ):
            density_list.append(data_)
            density_name_list.append(data_[1]["components"][0]["name"])

density_name_list = list(set(density_name_list))
full_property_list = []

for name in density_name_list:
    density_sub_list = []
    for i in range(len(density_list)):
        if density_list[i][1]["components"][0]["name"] == name:
            density_sub_list.append(density_list[i][1])
    full_property_list.append([
        name,
        density_sub_list])

full_reg_prop_list = []
full_reg_name_list = []
full_reg_smiles_list = []

for idx in range(len(full_property_list)):
    full_property = full_property_list[idx]
    name = full_property[0]
    density_sub_list = full_property[1]
    data_point_list = []
    ionic_liquid_name_ = None
    for component in density_sub_list[0]["components"]:
        ionic_liquid_name_ = component["name"]
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

    for density_dataframe in density_sub_list:
        temperature_index, value_index, pressure_index = None, None, None
        for i in range(len(density_dataframe["dhead"])):
            try:
                if "Temperature" in density_dataframe["dhead"][i][0]:
                    temperature_index = i
                elif "Specific density" in density_dataframe["dhead"][i][0]:
                    value_index = i
                elif (
                        "Pressure" in density_dataframe["dhead"][i][0]
                        and "kPa" in density_dataframe["dhead"][i][0]
                ):
                    pressure_index = i
            except TypeError:
                print(density_dataframe["dhead"])

        try:
            assert (
                    temperature_index is not None
                    and value_index is not None
            )
        except AssertionError:
            print('Data structure error')
            print(density_dataframe["dhead"])
        for point in density_dataframe["data"]:
            temperature_ = float(point[temperature_index][0])
            pressure_ = float(
                point[pressure_index][0]) if (pressure_index is not None and float(point[pressure_index][0]) - 101.325 > 0.5) else 101.325
            density_ = float(point[value_index][0])
            data_point_ = [temperature_, pressure_, density_]
            data_point_list.append(data_point_)

    data_point_list = np.array(data_point_list)
    if len(data_point_list) < 5:
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
        regressed_density_335 = linear_reg.predict(np.array([335.64, 101.325]).reshape(1, -1))
        regressed_density_353 = linear_reg.predict(np.array([353.15, 101.325]).reshape(1, -1))
    else:
        regressed_density_335 = poly_reg.predict(
            poly_features.fit_transform(np.array([335.64, 101.325]).reshape(1, -1)))
        regressed_density_353 = poly_reg.predict(
            poly_features.fit_transform(np.array([353.15, 101.325]).reshape(1, -1)))

    if all(np.array([il_x,
                     regressed_density_335[0],
                     regressed_density_353[0],
                     ]) > 0):
        full_reg_prop_list.append([
            il_x,
            regressed_density_335[0],
            regressed_density_353[0],
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
df['Density in 335.64K'] = properties_for_df[:, 1]
df['Density in 353.15K'] = properties_for_df[:, 2]

df = df.sort_values(by='Density in 335.64K', ascending=False)

PandasTools.AddMoleculeColumnToFrame(df, 'Smiles', 'Figure')
PandasTools.SaveXlsxFromFrame(df, 'ZDataE_AnalysisResult/database_density.xlsx', molCol='Figure')
