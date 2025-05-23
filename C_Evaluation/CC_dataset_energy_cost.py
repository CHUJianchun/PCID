import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from A_Preprocess.AU_diffusion_dataset import PropertyDiffusionDataset

#  1.94244774e+00  6.84932225e+02  1.75138082e+03 -6.18670952e-01
#  -1.59819189e+00 -6.45504691e-02  5.95993410e-02 -1.51191258e+03
#  -1.53403028e+03
# 3534.1255499715735

# NOTE BMIM BF4
# ec = energy_cost(
#         M=226.03,  # 离子液体相对分子质量，自变量
#         dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
#         dynamic_viscosity_353=0.0139,  # 离子液体在353.15K下的粘度 Pa s
#         density_335=1174.36,  # 离子液体的密度 kg/m3
#         density_353=1161.93,  # 离子液体的密度 kg/m3
#         heat_capacity_335=384.5,  # 离子液体在335.64K下的比热容 J/K/mol
#         heat_capacity_353=393.0,  # 离子液体353.15K下的比热容 J/K/mol
#         solubility_335=0.2,  # 离子液体在335.64K, 3MPa下的mole溶解度 ！！！高！！！
#         solubility_353=0.0062,  # 离子液体在353.15K, 101kPa下的mole溶解度
#         solubility_339=0.025,  # 离子液体在339.14K, 101kPa下的mole溶解度
#     )


dataset = PropertyDiffusionDataset()
label = dataset.labels.data.data[:, 2:]

energy_cost_list = np.load("ZDataB_ProcessedData/energy_cost_list.npy")

"""
linear_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
linear_reg.fit(energy_cost_list[:, :-1], energy_cost_list[:, -1])
print(linear_reg.coef_)
print(linear_reg.intercept_)
print(linear_reg.score(energy_cost_list[:, :-1], energy_cost_list[:, -1]))
print(linear_reg.predict(np.array(
    [226.03, 0.022, 0.0139, 1174.36, 1161.93, 384.5, 393.0, 0.2, 0.0062, 0.025]).reshape(1, -1)))
"""
# NOTE 2.1724696390624176 for BMIM BF4
poly_features = PolynomialFeatures(degree=2)
poly_x = poly_features.fit_transform(energy_cost_list[:, :-1])
poly_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
poly_reg.fit(poly_x, energy_cost_list[:, -1])
energy_cost = poly_reg.predict(poly_features.fit_transform(label))

data = np.load("ZDataB_ProcessedData/all_il_separated_dataset.npz")
anion_smiles = data['anion_smiles_list']
cation_smiles = data['cation_smiles_list']

best_num = 400
best_index = np.argpartition(energy_cost, best_num)[:best_num]
best_il_index = dataset.labels[best_index][:, :2].numpy()
best_il_smiles = []
best_il_mol = []
for i in range(best_num):
    best_il_smiles.append(anion_smiles[int(best_il_index[i][0])] + '.' + cation_smiles[int(best_il_index[i][1])])
best_il_smiles = [smiles for smiles in best_il_smiles if
                  ('FC(CCSCCCN1C=[N+](C=C1)CCCSCCC(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)'
                   '(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)F') not in smiles and ('FC(CCSC(C[N+]1=CN(C=C1)C)CSCCC(C(C(C(C(C(C(C(F)'
                                                                                               '(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)'
                                                                                               '(C(C(C(C(C(C(C(F)(F)F)(F)F)'
                                                                                               '(F)F)(F)F)(F)F)(F)F)(F)F)F') not in smiles]

best_il_mol = [Chem.MolFromSmiles(best_il_smiles[i]) for i in range(len(best_il_smiles))]
fig = Draw.MolsToGridImage(best_il_mol, molsPerRow=10, subImgSize=(400, 400))
fig.save('ZDataE_AnalysisResult/energy_cost_dataset.png')
# print(poly_reg.coef_)
# print(poly_reg.intercept_)
# print(poly_reg.score(poly_x, energy_cost_list[:, -1]))
# print(poly_reg.predict(poly_features.fit_transform(np.array([226.03, 0.022, 0.0139, 1174.36, 1161.93, 384.5, 393.0, 0.2, 0.0062, 0.025]).reshape(1, -1))))
