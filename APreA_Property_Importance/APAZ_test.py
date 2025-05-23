from U_Simulation.energy_simulation import energy_cost
import numpy as np


def try_ec(idx):
    m_range = (100, 200, 400)
    dynamic_viscosity_335_range = (0.01, 0.1, 0.6)
    dynamic_viscosity_353_range = (0.01, 0.1, 0.6)
    density_335_range = (800, 1100, 1400)
    density_353_range = (750, 1050, 1350)
    heat_capacity_335_range = (400, 800, 1200)
    heat_capacity_353_range = (400, 800, 1200)
    solubility_335_range = (0.3, 0.6, 0.9)
    solubility_353_range = (0.2, 0.5, 0.8)
    solubility_339_range = (0.25, 0.55, 0.85)
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
                                            """
                                            energy_cost_list.append(
                                                energy_cost(
                                                    m,
                                                    dynamic_viscosity_335,
                                                    dynamic_viscosity_353,
                                                    density_335,
                                                    density_353,
                                                    heat_capacity_335,
                                                    heat_capacity_353,
                                                    solubility_335,
                                                    solubility_353,
                                                )
                                            )
                                            """
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
                                                    solubility_339,
                                                ]
                                            )
    energy_cost_list = np.array(energy_cost_list)
    return energy_cost(*energy_cost_list[idx])


if __name__ == "__main__":
    # ec = try_ec(-1)
    # NOTE BMIM BF4, 2.17247
    ec = energy_cost(
        M=226.03,  # 离子液体相对分子质量，自变量
        dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
        dynamic_viscosity_353=0.0139,  # 离子液体在353.15K下的粘度 Pa s
        density_335=1174.36,  # 离子液体的密度 kg/m3
        density_353=1161.93,  # 离子液体的密度 kg/m3
        heat_capacity_335=384.5,  # 离子液体在335.64K下的比热容 J/K/mol
        heat_capacity_353=393.0,  # 离子液体353.15K下的比热容 J/K/mol
        solubility_335=0.2,  # 离子液体在335.64K, 3MPa下的mole溶解度 ！！！高！！！
        solubility_353=0.0062,  # 离子液体在353.15K, 101kPa下的mole溶解度
        solubility_339=0.025,  # 离子液体在339.14K, 101kPa下的mole溶解度
    )
    pass
