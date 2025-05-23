from U_Simulation.component import *

temperature_atm = 298.15
pressure_atm = 101.0

P = 1000  # 固定
volume_rate = np.array([0.15, 0.05, 0.8])  # 固定


# NOTE 压力的单位是kPa，但PropSI的输出是Pa
# NOTE 焓的单位是kJ/mol，但PropSI的输出是J/mol
def energy_cost(
        M=226.03,  # 离子液体相对分子质量，自变量
        dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
        dynamic_viscosity_353=0.0139,  # 离子液体在353.15K下的粘度 Pa s
        density_335=1174.36,  # 离子液体的密度 kg/m3
        density_353=1161.93,  # 离子液体的密度 kg/m3
        heat_capacity_335=384.5,  # 离子液体在335.64K下的比热容 J/K/mol
        heat_capacity_353=393.0,  # 离子液体353.15K下的比热容 J/K/mol
        solubility_335=0.06,  # 离子液体在335.64K, 1MPa下的mole溶解度 ！！！高！！！
        solubility_353=0.0062,  # 离子液体在353.15K, 101kPa下的mole溶解度
        solubility_339=0.025,  # 离子液体在339.14K, 101kPa下的mole溶解度
):
    viscosity_335 = dynamic_viscosity_335 / density_335 * 1e6
    viscosity_353 = dynamic_viscosity_353 / density_353 * 1e6
    heat_capacity_273 = (
            heat_capacity_335
            - (heat_capacity_353 - heat_capacity_335) / (353.15 - 335.64) * 273.15
    )
    heat_capacity_339 = heat_capacity_335 + (heat_capacity_353 - heat_capacity_335) / (
            353.15 - 335.64
    ) * (339.14 - 335.64)

    enthalpy_335 = (heat_capacity_273 + heat_capacity_335) * (335.64 - 273.15) / 2
    enthalpy_353 = (heat_capacity_273 + heat_capacity_353) * (353.64 - 273.15) / 2
    enthalpy_339 = (heat_capacity_273 + heat_capacity_339) * (339.14 - 273.15) / 2

    mass_rate = (
            volume_rate
            * np.array([44, 32, 28])
            / (volume_rate * np.array([44, 32, 28])).sum()
    )
    oyx_nit_mass_rate = mass_rate[1:] / mass_rate[1:].sum()
    mix1 = (
            gas1
            + "["
            + str(mass_rate[0])
            + "]&"
            + gas2
            + "["
            + str(mass_rate[1])
            + "]&"
            + gas3
            + "["
            + str(mass_rate[2])
            + "]"
    )
    mix2 = (
            gas2
            + "["
            + str(oyx_nit_mass_rate[0])
            + "]&"
            + gas3
            + "["
            + str(oyx_nit_mass_rate[1])
            + "]"
    )

    """Energy Analysis"""
    T1 = 298.15
    P1 = 101
    # T18 = 298.15
    # T19 = 298.15

    iseneff = 0.88  # 等熵效率
    mecheff = 0.95  # 机械效率
    isenteff = 0.88  # 透平等熵效率

    mgas = 1000
    mfluid = 3000
    mgasN = 1000 * (1 - mass_rate[0])
    # DelT = 5
    x = 0.5
    mf1 = mfluid * x
    mf2 = mfluid * (1 - x)
    mgasNO = mgasN

    CRat = (P / 101) ** 0.5
    Han1 = PropsSI("HMOLAR", "T", T1, "P", P1 * 1000, mix1)
    Shang1 = PropsSI("S", "T", T1, "P", P1 * 1000, mix1)
    Shangs2 = Shang1
    P2 = P1 * CRat

    Hans2 = PropsSI("HMOLAR", "P", P2 * 1000, "S", Shangs2, "REFPROP::" + mix1)

    Han2 = (Hans2 - Han1) / iseneff + Han1
    T2 = PropsSI("T", "HMOLAR", Han2, "P", P2 * 1000, "REFPROP::" + mix1)

    T17 = 333.15
    T18 = T17
    T19 = T17
    T3, T21 = intercooler(T2, T19, P2, 101, mgas, mf1, 5, mass_rate)
    Han3 = PropsSI("HMOLAR", "T", T3, "P", P2 * 1000, mix1)
    Shang3 = PropsSI("S", "T", T3, "P", P2 * 1000, mix1)
    Shangs4 = Shang3
    P4 = P2 * CRat
    Hans4 = PropsSI("HMOLAR", "P", P4 * 1000, "S", Shangs4, "REFPROP::" + mix1)
    Han4 = (Hans4 - Han3) / iseneff + Han3
    T4 = PropsSI("T", "HMOLAR", Han4, "P", P4 * 1000, "REFPROP::" + mix1)
    T5, T20 = intercooler(T4, T18, P4, 101, mgas, mf2, 5, mass_rate)
    Han20 = PropsSI("HMOLAR", "T", T20, "P", 101 * 1000, fluid1)
    Han21 = PropsSI("HMOLAR", "T", T21, "P", 101 * 1000, fluid1)
    Han22 = Han20 * x + Han21 * (1 - x)
    T22 = PropsSI("T", "HMOLAR", Han22, "P", 101 * 1000, fluid1)
    T23 = T22
    # Shang22 = PropsSI("S", "T", T22, "P", 101, fluid1)
    Cp1 = PropsSI("C", "T", T1, "P", 101 * 1000, mix1)
    Cv1 = PropsSI("O", "T", T1, "P", 101 * 1000, mix1)
    Cp2 = PropsSI("C", "T", T3, "P", P2 * 1000, mix1)
    Cv2 = PropsSI("O", "T", T3, "P", P2 * 1000, mix1)
    K1 = Cp1 / Cv1
    K2 = Cp2 / Cv2
    wcop1 = (
            mgas
            * Cp1
            * (CRat ** (1 - 1 / K1) - 1)
            * (T1 * ((Han2 - Han1) / (Hans2 - Han1)))
    )
    wcop2 = (
            mgas
            * Cp2
            * (CRat ** (1 - 1 / K2) - 1)
            * (T3 * ((Han4 - Han3) / (Hans4 - Han3)))
    )
    wcop = (wcop1 + wcop2) / 1000 / mecheff
    T28 = 353.15
    P28 = 101

    Diffmol = solubility_335 / (1 - solubility_335) - solubility_353 / (
            1 - solubility_353
    )
    mCO2 = (10 ** 6) * mass_rate[0] / 44
    mCO2ton = mCO2 * 44 / 1000 / 1000
    mCO2rich = mCO2 / Diffmol * solubility_335 / (1 - solubility_335)
    mCO2lean = mCO2 / Diffmol * solubility_353 / (1 - solubility_353)
    mIL = mCO2rich / Diffmol
    zIL = mIL * M / 1000
    peff1 = pumpeff(viscosity_335)
    peff2 = pumpeff(viscosity_353)
    Wpi = zIL * 9.81 * 20 / 0.95 / 1000
    Wp = (2 + peff1 + peff2) * Wpi
    CarDHan = PropsSI("HMOLAR", "T", T28, "P", P28 * 1000, "CO2")
    HEeff = 0.8
    Tlc = T28 - (T28 - T5) * HEeff

    Hanrh = (
            function_han(enthalpy_353, solubility_353, T28, 101) * (mIL + mCO2lean)
            - function_han(enthalpy_339, solubility_339, Tlc, P28) * (mIL + mCO2lean)
            + function_han(enthalpy_335, solubility_335, T5, P4) * (mIL + mCO2rich)
    )
    Qd = (
            function_han(enthalpy_353, solubility_353, T28, 101) * (mIL + mCO2lean)
            + mCO2 * CarDHan
            - Hanrh
    )

    QdCO2 = Qd / (10 ** 9) / mCO2ton
    WpCO2 = Wp / (10 ** 6) / mCO2ton

    P12 = P4
    T12 = T5
    y = 0.5
    mrec1 = mfluid * y
    mrec2 = mfluid * y
    T24 = T23
    [T26, T13] = LTES(T24, T12, 101, P12, mrec1, mgasNO, 2.5, oyx_nit_mass_rate)
    P13 = P12
    Han13 = PropsSI("HMOLAR", "T", T13, "P", P13 * 1000, "REFPROP::" + mix2)
    Shang13 = PropsSI("S", "T", T13, "P", P13 * 1000, "REFPROP::" + mix2)
    Shangs14 = Shang13
    TRat = (CRat ** 2) ** (1 / 3)
    P14 = P13 / TRat
    Hans14 = PropsSI("HMOLAR", "P", P14 * 1000, "S", Shangs14, "REFPROP::" + mix2)
    Han14 = Han13 - isenteff * (Han13 - Hans14)
    T14 = PropsSI("T", "HMOLAR", Han14, "P", P14 * 1000, "REFPROP::" + mix2)
    [T25, T15] = LTES(T23, T14, 101, P14, mrec2, mgasNO, 2.5, oyx_nit_mass_rate)
    Han15 = PropsSI("HMOLAR", "T", T15, "P", P14 * 1000, "REFPROP::" + mix2)
    Shang15 = PropsSI("S", "T", T15, "P", P14 * 1000, "REFPROP::" + mix2)

    Shangs16 = Shang15
    P16 = P14 / TRat
    Hans16 = PropsSI("HMOLAR", "P", P16 * 1000, "S", Shangs16, "REFPROP::" + mix2)
    Han16 = Han15 - isenteff * (Han15 - Hans16)

    Wout1 = mgasNO * (Han13 - Han14) * mecheff / (10 ** 9)
    Wout2 = mgasNO * (Han15 - Han16) * mecheff / (10 ** 9)
    Wout = Wout1 + Wout2

    Wnetelect = wcop / 1e6 - Wout
    WnetelectCO2 = Wnetelect / mCO2ton
    WtotalCO2 = WnetelectCO2 + WpCO2 + QdCO2  # 总能耗：压缩机，泵功，热耗
    print(WnetelectCO2)
    print(WpCO2)
    print(QdCO2)
    return WtotalCO2


if __name__ == "__main__":
    # print(energy_cost())
    """
    print(energy_cost(M=237.04,  # 离子液体相对分子质量，自变量
                      dynamic_viscosity_335=0.022,  # 离子液体在335.64K下的粘度 Pa s
                      dynamic_viscosity_353=0.017,  # 离子液体在353.15K下的粘度 Pa s
                      density_335=1168.712,  # 离子液体的密度 kg/m3
                      density_353=1156.715,  # 离子液体的密度 kg/m3
                      heat_capacity_335=476.45,  # 离子液体在335.64K下的比热容 J/K/mol 2.09 * 237.04
                      heat_capacity_353=502.52,  # 离子液体353.15K下的比热容 J/K/mol 2.05 * 237.04
                      solubility_335=0.442 * 0.001 * 237.04,  # 离子液体在335.64K, 1MPa下的mole溶解度 ！！！高！！！
                      solubility_353=0.0125 * 0.001 * 237.04,  # 离子液体在353.15K, 101kPa下的mole溶解度
                      solubility_339=0.0134 * 0.001 * 237.04,
                      ))  # 离子液体在339.14K, 101kPa下的mole溶解度))
    """
    print(energy_cost(M=240,  # 离子液体相对分子质量，自变量
                      dynamic_viscosity_335=0.02,  # 离子液体在335.64K下的粘度 Pa s
                      dynamic_viscosity_353=0.015,  # 离子液体在353.15K下的粘度 Pa s
                      density_335=1200,  # 离子液体的密度 kg/m3
                      density_353=1190,  # 离子液体的密度 kg/m3
                      heat_capacity_335=490,  # 离子液体在335.64K下的比热容 J/K/mol 2.09 * 237.04
                      heat_capacity_353=500,  # 离子液体353.15K下的比热容 J/K/mol 2.05 * 237.04
                      solubility_335=0.11,  # 离子液体在335.64K, 1MPa下的mole溶解度 ！！！高！！！
                      solubility_353=0.002,  # 离子液体在353.15K, 101kPa下的mole溶解度
                      solubility_339=0.003,
                      ))  # 离子液体在339.14K, 101kPa下的mole溶解度))
    """
    1.2458184400765617
    0.022342650484615444
    0.42311464149528455
    1.6912757320564618

    # BMIM BF4 1.8417663283688768
    print(energy_cost(M=260.237,  # 离子液体相对分子质量，自变量
                      dynamic_viscosity_335=0.0152991595756592,  # 离子液体在335.64K下的粘度 Pa s
                      dynamic_viscosity_353=0.0106137,  # 离子液体在353.15K下的粘度 Pa s
                      density_335=1348.751878,  # 离子液体的密度 kg/m3
                      density_353=1332.192619,  # 离子液体的密度 kg/m3
                      heat_capacity_335=394.3100281,  # 离子液体在335.64K下的比热容 J/K/mol
                      heat_capacity_353=399.960968,  # 离子液体353.15K下的比热容 J/K/mol
                      solubility_335=0.160782172,  # 离子液体在335.64K, 1MPa下的mole溶解度 ！！！高！！！
                      solubility_353=0.083702025,  # 离子液体在353.15K, 101kPa下的mole溶解度
                      solubility_339=0.109473435))  # 离子液体在339.14K, 101kPa下的mole溶解度))

1.2458184400765617
0.052060101660394854
0.5481319490476743
1.8460104907846309
"""