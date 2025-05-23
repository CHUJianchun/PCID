import os

from U_Simulation.energy_simulation import energy_cost
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import warnings
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")


# 226.03, 0.022, 0.0139, 1174.36, 1161.93, 384.5, 393.0, 0.2, 0.0062, 0.025
def observe_property_importance():
    m_range = (226.03, 236.03)
    dynamic_viscosity_335_range = (0.022, 0.032)
    dynamic_viscosity_353_range = (0.0139, 0.0159)
    density_335_range = (1174.36, 1100)
    density_353_range = (1161.93, 1050)
    heat_capacity_335_range = (384.5, 404.5)
    heat_capacity_353_range = (393.0, 424.5)
    solubility_335_range = (0.2, 0.3)
    solubility_353_range = (0.0062, 0.0088)
    solubility_339_range = (0.025, 0.035)
    energy_cost_list = []
    for m in m_range:
        for dynamic_viscosity_335 in dynamic_viscosity_335_range:
            for dynamic_viscosity_353 in dynamic_viscosity_353_range:
                for density_335 in density_335_range:
                    for density_353 in density_353_range:
                        for heat_capacity_335 in heat_capacity_335_range:
                            for heat_capacity_353 in heat_capacity_353_range:
                                for solubility_335 in solubility_335_range:
                                    for solubility_353 in solubility_353_range:
                                        for solubility_339 in solubility_339_range:
                                            energy_cost_list.append(
                                                [
                                                    m,
                                                    dynamic_viscosity_335,
                                                    dynamic_viscosity_353,
                                                    density_335,
                                                    density_353,
                                                    heat_capacity_335,
                                                    heat_capacity_353,
                                                    solubility_335,
                                                    solubility_353,
                                                    solubility_339
                                                ]
                                            )
    energy_cost_list = np.array(energy_cost_list)

    def cal_energy_cost(idx):
        file_name = str(idx) + ".txt"
        if not os.path.exists("ZDataB_ProcessedData/energy_cost/" + file_name):
            try:
                ec = energy_cost(*energy_cost_list[idx])
                with open("ZDataB_ProcessedData/energy_cost/" + file_name, "w") as f:
                    f.write(str(ec))
            except:
                with open("ZDataB_ProcessedData/energy_cost/" + file_name, "w") as f:
                    f.write("x")

    Parallel(n_jobs=-1, verbose=10)(
        delayed(cal_energy_cost)(idx) for idx in range(len(energy_cost_list))
    )


def conclude_property_importance():
    m_range = (226.03, 236.03)
    dynamic_viscosity_335_range = (0.022, 0.032)
    dynamic_viscosity_353_range = (0.0139, 0.0159)
    density_335_range = (1174.36, 1100)
    density_353_range = (1161.93, 1050)
    heat_capacity_335_range = (384.5, 404.5)
    heat_capacity_353_range = (393.0, 424.5)
    solubility_335_range = (0.2, 0.3)
    solubility_353_range = (0.0062, 0.0088)
    solubility_339_range = (0.025, 0.035)
    energy_cost_input_list = []
    energy_cost_list = []
    for m in m_range:
        for dynamic_viscosity_335 in dynamic_viscosity_335_range:
            for dynamic_viscosity_353 in dynamic_viscosity_353_range:
                for density_335 in density_335_range:
                    for density_353 in density_353_range:
                        for heat_capacity_335 in heat_capacity_335_range:
                            for heat_capacity_353 in heat_capacity_353_range:
                                for solubility_335 in solubility_335_range:
                                    for solubility_353 in solubility_353_range:
                                        for solubility_339 in solubility_339_range:
                                            energy_cost_input_list.append(
                                                [
                                                    m,
                                                    dynamic_viscosity_335,
                                                    dynamic_viscosity_353,
                                                    density_335,
                                                    density_353,
                                                    heat_capacity_335,
                                                    heat_capacity_353,
                                                    solubility_335,
                                                    solubility_353,
                                                    solubility_339
                                                ]
                                            )
    energy_cost_input_list = np.array(energy_cost_input_list)
    for i in trange(len(energy_cost_input_list)):
        file_name = str(i) + ".txt"
        with open("ZDataB_ProcessedData/energy_cost/" + file_name, "r") as f:
            value = f.read()
            if value != 'x':
                energy_cost_list.append(np.concatenate((energy_cost_input_list[i], float(value))))
    energy_cost_list = np.array(energy_cost_list)
    return energy_cost_list


def regress_property_importance():
    energy_cost_list = np.load("ZDataB_ProcessedData/energy_cost_list.npy")
    linear_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
    linear_reg.fit(energy_cost_list[:, :-1], energy_cost_list[:, -1])
    print(linear_reg.coef_)
    print(linear_reg.intercept_)
    print(linear_reg.score(energy_cost_list[:, :-1], energy_cost_list[:, -1]))
    print(linear_reg.predict(np.array(
        [226.03, 0.022, 0.0139, 1174.36, 1161.93, 384.5, 393.0, 0.2, 0.0062, 0.025]).reshape(1, -1)))
    # NOTE 2.1724696390624176 for BMIM BF4
    poly_features = PolynomialFeatures(degree=2)
    poly_x = poly_features.fit_transform(energy_cost_list[:, :-1])
    poly_reg = linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
    poly_reg.fit(poly_x, energy_cost_list[:, -1])
    # print(poly_reg.coef_)
    coef_ = np.array(poly_reg.coef_)
    # coef_ = np.round(coef_, 2)
    np.set_printoptions(suppress=True)
    print(coef_)
    print(poly_reg.intercept_)
    print(poly_reg.score(poly_x, energy_cost_list[:, -1]))
    print(poly_reg.predict(poly_features.fit_transform(np.array(
        [226.03, 0.022, 0.0139, 1174.36, 1161.93, 384.5, 393.0, 0.2, 0.0062, 0.025]).reshape(1, -1))))


if __name__ == "__main__":
    # observe_property_importance()
    regress_property_importance()
    #  1.94244774e+00  6.84932225e+02  1.75138082e+03 -6.18670952e-01
    #  -1.59819189e+00 -6.45504691e-02  5.95993410e-02 -1.51191258e+03
    #  -1.53403028e+03
    # 3534.1255499715735
