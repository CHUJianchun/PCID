import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, 'C:\\Program Files (x86)\\REFPROP\\')

gas1 = 'CO2'
gas2 = 'Oxygen'
gas3 = 'Nitrogen'
fluid1 = 'D6'


def liquid_volume(t):
    v = 5.031 * 1e-6 * t - 0.000305
    molv = v * 44.01 / 1000
    return molv


def intercooler(Thoti, Tcoldi, Photi, Pcoldi, qhot, qcold, delt, mass_rate):
    Thotout = []
    mix1 = gas1 + '[' + str(mass_rate[0]) + ']&' + gas2 + '[' + str(mass_rate[1]) + ']&' + gas3 + '[' + str(
        mass_rate[2]) + ']'

    Hancoldi = PropsSI('H', 'T', Tcoldi, 'P', Pcoldi, fluid1)
    Hanhoti = PropsSI('H', 'T', Thoti, 'P', Photi, mix1)
    deerta = Thoti - Tcoldi - 50
    Thotout[0] = Tcoldi + deerta - 0.1
    m = 10
    q = 0
    while m > delt:
        q += 1
        Thotout.append(Thotout[q - 1] - 1)
        Hanhoto = PropsSI('H', 'T', Thotout[q], 'P', Photi, fluid1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        Hancoldm, Hanhotm, Thotm, Tcoldm, DeltHm = np.zeros(50)
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI('T', 'H', Hanhotm[i], 'P', Photi, mix1)
            Tcoldm[i] = PropsSI('T', 'H', Hancoldm[i], 'P', Pcoldi, fluid1)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = DeltHm.min()

    r = 0
    Thotoutplus = [Thotout[q]]
    while m < delt:
        r = r + 1
        Thotoutplus[r] = Thotoutplus[r - 1] + 0.02
        Hanhoto = PropsSI('H', 'T', Thotoutplus[r], 'P', Photi, mix1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI('T', 'H', Hanhotm[i], 'P', Photi, mix1)
            Tcoldm[i] = PropsSI('T', 'H', Hancoldm[i], 'P', Pcoldi, fluid1)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = DeltHm.min()

    Hancoldo = (qhot / qcold) * (Hanhoti - Hanhoto) + Hancoldi
    Tcoldout = PropsSI('T', 'H', Hancoldo, 'P', Pcoldi, fluid1)
    Th = Thotoutplus[r]
    Tc = Tcoldout
    return Th, Tc


def second_virial(t):
    return -0.009642 * np.exp(-0.02284 * t) + (-0.0007305 * np.exp(-0.006279 * t))


def sat_pressure(t):
    return -1.45218572296916E-07 * t ** 5 + \
        0.000240185511973386 * t ** 4 - \
        0.15756639189518 * t ** 3 + \
        50.9572675380696 * t ** 2 - \
        8039.98308882096 * t + 492468.480491414


def solution(B, p, T, v, ps):
    # TODO 这个是关于离子液体溶解度的，要改
    return 0


def pumpeff(t):
    # TODO 这个是关于离子液体溶解度的，要改
    return 0


def function_han(t, p):
    # TODO 这个是关于离子液体溶解度的，要改
    return 0


def LTES(Thoti, Tcoldi, Photi, Pcoldi, qhot, qcold, delt, mass_rate):
    oyx_nit_mass_rate = mass_rate[1:] / mass_rate[1:].sum()
    mix2 = gas2 + '[' + str(oyx_nit_mass_rate[0]) + ']&' + gas3 + '[' + str(oyx_nit_mass_rate[1]) + ']'
    Hancoldi = PropsSI('H', 'T', Tcoldi, 'P', Pcoldi, mix2)
    Hanhoti = PropsSI('H', 'T', Thoti, 'P', Photi, fluid1)
    deerta = Thoti - Tcoldi
    Thotout = Tcoldi + deerta - 0.1
    m = 10
    q = 0
    while m > delt:
        q += 1
        Thotout[q] = Thotout[q - 1] - 1
        Hanhoto = PropsSI('H', 'T', Thotout[q], 'P', Photi, fluid1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        Hancoldm, Hanhotm, Thotm, Tcoldm, DeltHm = np.zeros(50)
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI('T', 'H', Hanhotm[i], 'P', Photi, fluid1)
            Tcoldm[i] = PropsSI('T', 'H', Hancoldm[i], 'P', Pcoldi, mix2)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = np.min(DeltHm)
    s = 0
    Thotoutplus = [Thotout[q]]
    while m < delt:
        s += 1
        Thotoutplus[s] = Thotoutplus[s - 1] + 0.02
        Hanhoto = PropsSI('H', 'T', Thotoutplus[s], 'P', Photi, fluid1)
        PerH = qhot * (Hanhoti - Hanhoto) / 50
        for i in range(50):
            Hancoldm[i] = Hancoldi + PerH * (i + 1) / qcold
            Hanhotm[i] = Hanhoto + ((qcold / qhot) * (Hancoldm[i] - Hancoldi))
            Thotm[i] = PropsSI('T', 'H', Hanhotm[i], 'P', Photi, fluid1)
            Tcoldm[i] = PropsSI('T', 'H', Hancoldm[i], 'P', Pcoldi, mix2)
            DeltHm[i] = Thotm[i] - Tcoldm[i]
        m = np.min(DeltHm)
    Hancoldo = (qhot / qcold) * (Hanhoti - Hanhoto) + Hancoldi
    Tcoldout = PropsSI('T', 'H', Hancoldo, 'P', Pcoldi, mix2)
    Th = Thotoutplus[s]
    Tc = Tcoldout
    return Th, Tc
