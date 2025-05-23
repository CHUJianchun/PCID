import json
import numpy as np
import torch
from torch import tensor
from component import *

temperature_atm = 298.15
pressure_atm = 101.

P = 3000  # TODO 自变量，但可以固定
M = 226.03  # TODO 离子液体相对分子质量，自变量
volume_rate = np.array([0.15, 0.05, 0.8])  # TODO 自变量

mass_rate = volume_rate * np.array([44, 32, 28]) / (volume_rate * np.array([44, 32, 28])).sum()
oyx_nit_mass_rate = mass_rate[1:] / mass_rate[1:].sum()
mix1 = gas1 + '[' + str(mass_rate[0]) + ']&' + gas2 + '[' + str(mass_rate[1]) + ']&' + gas3 + '[' + str(
    mass_rate[2]) + ']'
mix2 = gas2 + '[' + str(oyx_nit_mass_rate[0]) + ']&' + gas3 + '[' + str(oyx_nit_mass_rate[1]) + ']'

Ham1 = PropsSI('H', 'T', temperature_atm, 'P', mix1)
Sam1 = PropsSI('S', 'T', temperature_atm, 'P', mix1)

Ham2 = PropsSI('H', 'T', temperature_atm, 'P', fluid1)
Sam2 = PropsSI('S', 'T', temperature_atm, 'P', fluid1)

Ham3 = PropsSI('H', 'T', temperature_atm, 'P', mix2)
Sam3 = PropsSI('S', 'T', temperature_atm, 'P', mix2)

Ham4 = PropsSI('H', 'T', temperature_atm, 'P', 'CO2')
Sam4 = PropsSI('S', 'T', temperature_atm, 'P', 'CO2')

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
DelT = 5
x = 0.5
mf1 = mfluid * x
mf2 = mfluid * (1 - x)
mgasNO = mgasN
mfluidhot = mfluid

CRat = (P / 101) ** 0.5
Han1 = PropsSI('H', 'T', T1, 'P', P1, mix1)
Shang1 = PropsSI('S', 'T', T1, 'P', P1, mix1)
Shangs2 = Shang1
P2 = P1 * CRat

Hans2 = PropsSI('H', 'P', P2, 'S', Shangs2, mix1)
Han2 = (Hans2 - Han1) / iseneff + Han1
T2 = PropsSI('T', 'H', Han2, 'P', P2, mix1)
Shang2 = PropsSI('S', 'T', T2, 'P', P2, mix1)
T17 = 333.15
T18 = T17
T19 = T17
T3, T21 = intercooler(T2, T19, P2, 101, mgas, mf1, 5, mass_rate)
Han3 = PropsSI('H', 'T', T3, 'P', P2, mix1)
Shang3 = PropsSI('S', 'T', T3, 'P', P2, mix1)
Shangs4 = Shang3
P4 = P2 * CRat
Hans4 = PropsSI('H', 'P', P4, 'S', Shangs4, mix1)
Han4 = (Hans4 - Han3) / iseneff + Han3
T4 = PropsSI('T', 'H', Han4, 'P', P4, mix1)
Shang4 = PropsSI('S', 'T', T4, 'P', P4, mix1)
T5, T20 = intercooler(T4, T18, P4, 101, mgas, mf2, 5, mass_rate)
Han5 = PropsSI('H', 'T', T5, 'P', P4, mix1)
Shang5 = PropsSI('S', 'T', T5, 'P', P4, mix1)
Han20 = PropsSI('H', 'T', T20, 'P', 101, fluid1)
Shang20 = PropsSI('S', 'T', T20, 'P', 101, fluid1)
Han21 = PropsSI('H', 'T', T21, 'P', 101, fluid1)
Shang21 = PropsSI('S', 'T', T21, 'P', 101, fluid1)
Han17 = PropsSI('H', 'T', T17, 'P', 101, fluid1)
Shang17 = PropsSI('S', 'T', T17, 'P', 101, fluid1)
Han22 = Han20 * x + Han21 * (1 - x)
T22 = PropsSI('T', 'H', Han22, 'P', 101, fluid1)
T23 = T22
Shang22 = PropsSI('S', 'T', T22, 'P', 101, fluid1)
Cp1 = PropsSI('C', 'T', T1, 'P', 101, mix1)
Cv1 = PropsSI('O', 'T', T1, 'P', 101, mix1)
Cp2 = PropsSI('C', 'T', T3, 'P', P2, mix1)
Cv2 = PropsSI('O', 'T', T3, 'P', P2, mix1)
K1 = Cp1 / Cv1
K2 = Cp2 / Cv2
wcop1 = mgas * Cp1 * (CRat ** (1 - 1 / K1) - 1) * (T1 * ((Han2 - Han1) / (Hans2 - Han1)))
wcop2 = mgas * Cp2 * (CRat ** (1 - 1 / K2) - 1) * (T3 * ((Han4 - Han3) / (Hans4 - Han3)))
wcop = (wcop1 + wcop2) / 1000 / mecheff
T28 = 353.15
P28 = 101
Han28 = PropsSI('H', 'T', T28, 'P', 101, 'CO2')
Shang28 = PropsSI('S', 'T', T28, 'P', 101, 'CO2')
v1 = liquid_volume(T5)
B1 = second_virial(T5)
ps1 = sat_pressure(T5)
v2 = liquid_volume(T28)
B2 = second_virial(T28)
ps2 = sat_pressure(T28)
x1 = solution(B1, P4, T5, v1, ps1)
x2 = solution(B2, P28, T28, v2, ps2)
Diffmol = x1 / (1 - x1) - x2 / (1 - x2)
mCO2 = (10 ** 6) * mass_rate(1) / 44
mCO2ton = mCO2 * 44 / 1000 / 1000
mCO2rich = mCO2 / Diffmol * x1 / (1 - x1)
mCO2lean = mCO2 / Diffmol * x2 / (1 - x2)
mIL = mCO2rich / Diffmol
zIL = mIL * M / 1000
peff1 = pumpeff(T5)
peff2 = pumpeff(T28)  # T28 353.15
Wpi = zIL * 9.81 * 20 / 0.95 / 1000
Wp = (2 + peff1 + peff2) * Wpi
CarDHan = (PropsSI('H', 'T', T28, 'P', P28, 'CO2')) / 1000 * 0.044
HEeff = 0.8
Tlc = T28 - (T28 - T5) * HEeff
T11 = Tlc

Han6 = function_han(T5, P4) * (mIL + mCO2rich)
Han11 = function_han(Tlc, P4) * (mIL + mCO2lean)
Han9 = function_han(T28, 101) * (mIL + mCO2lean)
Hanrh = function_han(T28, 101) * (mIL + mCO2lean) - function_han(Tlc, P28) * (mIL + mCO2lean) + function_han(T5, P4) * (
        mIL + mCO2rich)
Qd = (function_han(T28, 101) * (mIL + mCO2lean) + mCO2 * CarDHan - Hanrh)
Wtotal = Qd + wcop + Wp
QdCO2 = Qd / (10 ** 6) / mCO2ton
WcopCO2 = wcop / (10 ** 6) / mCO2ton
WpCO2 = Wp / (10 ** 6) / mCO2ton
WtotalGJ = Wtotal / (10 ** 6) / mCO2ton
Han12 = PropsSI('H', 'T', T5, 'P', P4, mix2)
Shang12 = PropsSI('S', 'T', T5, 'P', P4, mix2)
P12 = P4
T12 = T5
y = 0.5
mrec1 = mfluid * y
mrec2 = mfluid * y
T24 = T23
[T26, T13] = LTES(T24, T12, 101, P12, mrec1, mgasNO, 2.5)
P13 = P12
Han13 = PropsSI('H', 'T', T13, 'P', P13, mix2)
Shang13 = PropsSI('S', 'T', T13, 'P', P13, mix2)
Han26 = PropsSI('H', 'T', T26, 'P', 101, fluid1)
Shang26 = PropsSI('S', 'T', T26, 'P', 101, fluid1)
Shangs14 = Shang13
TRat = (CRat ** 2) ** (1 / 3)
P14 = P13 / TRat
Hans14 = PropsSI('H', 'P', P14, 'S', Shangs14, mix2)
Han14 = Han13 - isenteff * (Han13 - Hans14)
T14 = PropsSI('T', 'H', Han14, 'P', P14, mix2)
Shang14 = PropsSI('S', 'T', T14, 'P', P14, mix2)
[T25, T15] = LTES(T23, T14, 101, P14, mrec2, mgasNO, 2.5)
Han15 = PropsSI('H', 'T', T15, 'P', P14, mix2)
Shang15 = PropsSI('S', 'T', T15, 'P', P14, mix2)
Han25 = PropsSI('H', 'T', T25, 'P', 101, fluid1)
Shang25 = PropsSI('S', 'T', T25, 'P', 101, fluid1)

Shangs16 = Shang15
P16 = P14 / TRat
Hans16 = PropsSI('H', 'P', P16, 'S', Shangs16, mix2)
Han16 = Han15 - isenteff * (Han15 - Hans16)
T16 = PropsSI('T', 'H', Han16, 'P', P16, mix2)
Shang16 = PropsSI('S', 'T', T16, 'P', P16, mix2)
Han27 = y * Han26 + (1 - y) * Han25
T27 = PropsSI('T', 'H', Han27, 'P', 101, fluid1)
Shang27 = PropsSI('S', 'T', T27, 'P', 101, fluid1)

Wout1 = mgasNO * (Han13 - Han14) * mecheff / (10 ** 9)
Wout2 = mgasNO * (Han15 - Han16) * mecheff / (10 ** 9)
Wout = Wout1 + Wout2

WoutCO2 = Wout / mCO2ton
Han19 = Han17
Han18 = Han17
Shang19 = Shang17
Shang18 = Shang17
Han23 = Han22
Han24 = Han22
Shang23 = Shang22
Shang24 = Shang22

ex1 = Han1 - Ham1 - temperature_atm * (Shang1 - Sam1)
ex2 = Han2 - Ham1 - temperature_atm * (Shang2 - Sam1)
ex3 = Han3 - Ham1 - temperature_atm * (Shang3 - Sam1)
ex4 = Han4 - Ham1 - temperature_atm * (Shang4 - Sam1)
ex5 = Han5 - Ham1 - temperature_atm * (Shang5 - Sam1)

ex12 = Han12 - Ham3 - temperature_atm * (Shang12 - Sam3)
ex13 = Han13 - Ham3 - temperature_atm * (Shang13 - Sam3)
ex14 = Han14 - Ham3 - temperature_atm * (Shang14 - Sam3)
ex15 = Han15 - Ham3 - temperature_atm * (Shang15 - Sam3)
ex16 = Han16 - Ham3 - temperature_atm * (Shang16 - Sam3)

ex17 = Han17 - Ham2 - temperature_atm * (Shang17 - Sam2)
ex18 = ex17
ex19 = ex17
ex20 = Han20 - Ham2 - temperature_atm * (Shang20 - Sam2)
ex21 = Han21 - Ham2 - temperature_atm * (Shang21 - Sam2)
ex22 = Han22 - Ham2 - temperature_atm * (Shang22 - Sam2)
ex23 = ex22
ex24 = ex22
ex25 = Han25 - Ham2 - temperature_atm * (Shang25 - Sam2)
ex26 = Han26 - Ham2 - temperature_atm * (Shang26 - Sam2)
ex27 = Han27 - Ham2 - temperature_atm * (Shang27 - Sam2)

ex28 = Han28 - Ham4 - temperature_atm * (Shang28 - Sam4)

ex1c = mgas * (ex1 - ex2) / 1000 + wcop1 / 1000
ex1h = mgas * (ex2 - ex3) + mf1 * (ex19 - ex21) / 1000
ex2c = mgas * (ex3 - ex4) / 1000 + wcop2 / 1000
ex2h = (mgas * (ex4 - ex5) + mf2 * (ex18 - ex28)) / 1000

exREC1 = (mgasNO * ex12 - ex13) + mrec1 * (ex24 - ex26) / 1000
exREC2 = (mgasNO * (ex14 - ex15) + mrec2 * (ex23 - ex25)) / 1000
exTUR1 = mgasNO * (ex13 - ex14) / 1000 - Wout1 * 1e6
exTUR2 = mgasNO * (ex15 - ex16) / 1000 - Wout2 * 1e6
exCT = (mfluidhot * ex27 - mf1 * ex19 - mf2 * ex18) / 1000
exHT = (mf1 * ex20 + mf2 * ex21 - mfluidhot * ex22) / 1000
exCapture = (mgas * ex5 - mgasNO * ex12 - mgas * mass_rate[0] * ex28 + Qd * (1 - temperature_atm / T28)) / 1000

ex = np.array([ex1c, ex1h, ex2c, ex2h, exREC1, exREC2, exTUR1, exTUR2, exCT, exHT, exCapture])

exTotal = ex.sum()
ets = Wout / (Wtotal / 1e6)
erts = Wout / ((Wtotal - Qd) / 1e6 + exCapture / 1e6)
Wnetelect = wcop / 1e6 - Wout
WnetelectCO2 = Wnetelect / mCO2ton
WtotalCO2 = WnetelectCO2 + WpCO2 + QdCO2  # 总能耗：压缩机，泵功，热耗
