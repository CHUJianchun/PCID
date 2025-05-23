import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, "C:\\Program Files (x86)\\REFPROP\\")

gas1 = "CO2"
gas2 = "Oxygen"
gas3 = "Nitrogen"
fluid1 = "D6"


def liquid_volume(t):
    v = 5.031 * 1e-6 * t - 0.000305
    molv = v * 44.01 / 1000
    return molv


def intercooler(Thoti, Tcoldi, Photi, Pcoldi, qhot, qcold, delt, mass_rate):
    Thotout = [0]
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

    Hancoldi = PropsSI("HMOLAR", "T", Tcoldi, "P", Pcoldi * 1000, fluid1)
    Hanhoti = PropsSI("HMOLAR", "T", Thoti, "P", Photi * 1000, "REFPROP::" + mix1)
    deerta = Thoti - Tcoldi - 50
    Thotout[0] = Tcoldi + deerta - 0.1
    m = 10
    q = 0
    while m > delt:
        q += 1
        Thotout.append(Thotout[q - 1] - 1)
        Hanhoto = PropsSI(
            "HMOLAR", "T", Thotout[q], "P", Photi * 1000, "REFPROP::" + mix1
        )
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        Hancoldm, Hanhotm, Thotm, Tcoldm, DeltHm = (
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
        )
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI(
                "T", "HMOLAR", Hanhotm[i], "P", Photi * 1000, "REFPROP::" + mix1
            )
            Tcoldm[i] = PropsSI("T", "HMOLAR", Hancoldm[i], "P", Pcoldi * 1000, fluid1)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = DeltHm.min()

    r = 0
    Thotoutplus = [Thotout[q]]
    while m < delt:
        r += 1
        Thotoutplus.append(Thotoutplus[r - 1] + 0.02)
        Hanhoto = PropsSI(
            "HMOLAR", "T", Thotoutplus[r], "P", Photi * 1000, "REFPROP::" + mix1
        )
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI(
                "T", "HMOLAR", Hanhotm[i], "P", Photi * 1000, "REFPROP::" + mix1
            )
            Tcoldm[i] = PropsSI("T", "HMOLAR", Hancoldm[i], "P", Pcoldi * 1000, fluid1)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = DeltHm.min()

    Hancoldo = (qhot / qcold) * (Hanhoti - Hanhoto) + Hancoldi
    Tcoldout = PropsSI("T", "HMOLAR", Hancoldo, "P", Pcoldi * 1000, fluid1)
    Th = Thotoutplus[r]
    Tc = Tcoldout
    return Th, Tc


def second_virial(t):
    return -0.009642 * np.exp(-0.02284 * t) + (-0.0007305 * np.exp(-0.006279 * t))


def sat_pressure(t):
    return (
        -1.45218572296916e-07 * t**5
        + 0.000240185511973386 * t**4
        - 0.15756639189518 * t**3
        + 50.9572675380696 * t**2
        - 8039.98308882096 * t
        + 492468.480491414
    )


def pumpeff(viscosity):
    return 0.098 * 1.019**viscosity


def function_han(enthalpy, x, t, p):
    co2_enthalpy = PropsSI("HMOLAR", "T", t, "P", p * 1000, "CO2")
    return x * co2_enthalpy + (1 - x) * enthalpy


def LTES(Thoti, Tcoldi, Photi, Pcoldi, qhot, qcold, delt, oyx_nit_mass_rate):
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
    Hancoldi = PropsSI("HMOLAR", "T", Tcoldi, "P", Pcoldi * 1000, "REFPROP::" + mix2)
    Hanhoti = PropsSI("HMOLAR", "T", Thoti, "P", Photi * 1000, fluid1)
    deerta = Thoti - Tcoldi
    Thotout = [Tcoldi + deerta - 0.1]
    m = 10
    q = 0
    while m > delt:
        q += 1
        Thotout.append(Thotout[q - 1] - 1)
        Hanhoto = PropsSI("HMOLAR", "T", Thotout[q], "P", Photi * 1000, fluid1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        Hancoldm, Hanhotm, Thotm, Tcoldm, DeltHm = (
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
            np.zeros(50),
        )
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI("T", "HMOLAR", Hanhotm[i], "P", Photi * 1000, fluid1)
            Tcoldm[i] = PropsSI(
                "T", "HMOLAR", Hancoldm[i], "P", Pcoldi * 1000, "REFPROP::" + mix2
            )
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = np.min(DeltHm)
    s = 0
    Thotoutplus = [Thotout[q]]
    while m < delt:
        s += 1
        Thotoutplus.append(Thotoutplus[s - 1] + 0.02)
        Hanhoto = PropsSI("HMOLAR", "T", Thotoutplus[s], "P", Photi * 1000, fluid1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI("T", "HMOLAR", Hanhotm[i], "P", Photi * 1000, fluid1)
            Tcoldm[i] = PropsSI(
                "T", "HMOLAR", Hancoldm[i], "P", Pcoldi * 1000, "REFPROP::" + mix2
            )
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = np.min(DeltHm)
    Hancoldo = (qhot / qcold) * (Hanhoti - Hanhoto) + Hancoldi
    Tcoldout = PropsSI("T", "HMOLAR", Hancoldo, "P", Pcoldi * 1000, "REFPROP::" + mix2)
    Th = Thotoutplus[s]
    Tc = Tcoldout
    return Th, Tc
