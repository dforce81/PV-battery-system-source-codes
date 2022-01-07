from __future__ import division

from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

model = AbstractModel()
dp = DataPortal()


### Set indices 

## Scenario = index for solar irradiance scenario 1 to 5*5*5*5 = 625
model.SCEN = Set()

## DTYPE = index for type of day d : "holiday", "weekend_winter", "workday_winter", "weekend_summer", "workday_summer"
model.DTYPE = Set()

## Day = index for day 1 to day 365
model.Day = Set()

model.Day_WeeklyApp = Set()

## Time = index for time period, at the end of ...
model.Time = Set()
model.ReducedTime_1 = Set(initialize=[1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
model.ReducedTime_2 = Set(initialize=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 ])

model.ReducedDay_1 = Set(initialize=[1, 2, 	3, 	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	36,	37,	38,	39,	40,	41,	42,	43,	44,	45,	46,	47,	48,	49,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	62,	63,	64,	65,	66,	67,	68,	69,	70,	71,	72,	73,	74,	75,	76,	77,	78,	79,	80,	81,	82,	83,	84,	85,	86,	87,	88,	89,	90,	91,	92,	93,	94,	95,	96,	97,	98,	99,	100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,	112,	113,	114,	115,	116,	117,	118,	119,	120,	121,	122,	123,	124,	125,	126,	127,	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,	160,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170,	171,	172,	173,	174,	175,	176,	177,	178,	179,	180,	181,	182,	183,	184,	185,	186,	187,	188,	189,	190,	191,	192,	193,	194,	195,	196,	197,	198,	199,	200,	201,	202,	203,	204,	205,	206,	207,	208,	209,	210,	211,	212,	213,	214,	215,	216,	217,	218,	219,	220,	221,	222,	223,	224,	225,	226,	227,	228,	229,	230,	231,	232,	233,	234,	235,	236,	237,	238,	239,	240,	241,	242,	243,	244,	245,	246,	247,	248,	249,	250,	251,	252,	253,	254,	255,	256,	257,	258,	259,	260,	261,	262,	263,	264,	265,	266,	267,	268,	269,	270,	271,	272,	273,	274,	275,	276,	277,	278,	279,	280,	281,	282,	283,	284,	285,	286,	287,	288,	289,	290,	291,	292,	293,	294,	295,	296,	297,	298,	299,	300,	301,	302,	303,	304,	305,	306,	307,	308,	309,	310,	311,	312,	313,	314,	315,	316,	317,	318,	319,	320,	321,	322,	323,	324,	325,	326,	327,	328,	329,	330,	331,	332,	333,	334,	335,	336,	337,	338,	339,	340,	341,	342,	343,	344,	345,	346,	347,	348,	349,	350,	351,	352,	353,	354,	355,	356,	357,	358,	359,	360,	361,	362,	363,	364])
model.ReducedDay_2 = Set(initialize=[2, 	3, 	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	36,	37,	38,	39,	40,	41,	42,	43,	44,	45,	46,	47,	48,	49,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	62,	63,	64,	65,	66,	67,	68,	69,	70,	71,	72,	73,	74,	75,	76,	77,	78,	79,	80,	81,	82,	83,	84,	85,	86,	87,	88,	89,	90,	91,	92,	93,	94,	95,	96,	97,	98,	99,	100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,	112,	113,	114,	115,	116,	117,	118,	119,	120,	121,	122,	123,	124,	125,	126,	127,	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,	160,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170,	171,	172,	173,	174,	175,	176,	177,	178,	179,	180,	181,	182,	183,	184,	185,	186,	187,	188,	189,	190,	191,	192,	193,	194,	195,	196,	197,	198,	199,	200,	201,	202,	203,	204,	205,	206,	207,	208,	209,	210,	211,	212,	213,	214,	215,	216,	217,	218,	219,	220,	221,	222,	223,	224,	225,	226,	227,	228,	229,	230,	231,	232,	233,	234,	235,	236,	237,	238,	239,	240,	241,	242,	243,	244,	245,	246,	247,	248,	249,	250,	251,	252,	253,	254,	255,	256,	257,	258,	259,	260,	261,	262,	263,	264,	265,	266,	267,	268,	269,	270,	271,	272,	273,	274,	275,	276,	277,	278,	279,	280,	281,	282,	283,	284,	285,	286,	287,	288,	289,	290,	291,	292,	293,	294,	295,	296,	297,	298,	299,	300,	301,	302,	303,	304,	305,	306,	307,	308,	309,	310,	311,	312,	313,	314,	315,	316,	317,	318,	319,	320,	321,	322,	323,	324,	325,	326,	327,	328,	329,	330,	331,	332,	333,	334,	335,	336,	337,	338,	339,	340,	341,	342,	343,	344,	345,	346,	347,	348,	349,	350,	351,	352,	353,	354,	355,	356,	357,	358,	359,	360,	361,	362,	363,	364,	365])

## Appliance = index for appliances
model.Appliance = Set()
model.Appliance_S = Set()
model.Appliance_S_1 = Set(initialize=[1,2])
model.Appliance_F = Set()
                


### Set parameters

#model.Scenario =  Param (model.Day, within=NonNegativeReals)

## M = big number
model.M = Param (initialize = 10000)

## APV = Area of PV array (m^2) => it means the capacity of 5.6kw
#model.APV = Param (initialize = 32.7485)

## RPV = PV array efficiency rate 
#model.RPV = Param (initialize = 0.342)  # 200% of the 0.171

## C_PV = annualized cost of single PV module ($/ea) (=315W, LG NEON2 BLACK LG315N1K-V5 315W MONO SOLAR PANEL, $440, life cycle 20 years))
model.C_PV = Param (initialize = 102.7808576)  # cost of single PV module = $255

## C_BAT = annualized cost of single battery ($/kWh) (=3.3kWh, LG CHEM RESU 3.3 LI-ION BATTERY STORAGE 3.3KWH (48V), $2400, life cycle 5 years)
model.C_BAT = Param (initialize = 163.3651734)  # cost of single PV module = £1,919.00 = $2,000

## C_DG = Life Cycle cost of diesel generator to generate 1kWh energy ($/kWh)
## Life-Cycle Costs = Installation Costs + Decommissioning Costs + Maintenance Costs + Fuel Costs - Revenues
model.C_DG = Param (initialize = 425)

## SPV = Basic PV array size (kW)
model.SPV = Param (initialize = 0.315)

## CBAT = Basic battery capacity (kWh)
model.BCAP = Param (initialize = 3.3)

## RINV = Inverter efficiency rate
model.RINV = Param (initialize = 0.98) 

## EM = Energy margin to operate occasionally used appliances (kWh)
model.EM = Param (model.Day, model.Time, within=NonNegativeReals)

## S = Forecasted solar radiation incident on the PV array during period t of day d (kWh/m^2) of scenario s
model.S = Param (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## RDCH = Battery self-discharging efficiency rate
model.RDCH = Param (initialize = 0.000139)

## RCHR = Battery charging efficiency rate
model.RCHR = Param (initialize = 0.99)

## SMIN = Battery minimum state of charge (%)
model.SMIN = Param (initialize = 0.05)

## SMAX = Battery maximum state of charge (%)
model.SMAX = Param (initialize = 0.95)

## EIAB : ## initial amount of energy in the battery on day 1 of period 1 (kWh)
model.EIAB = Param (initialize = 3)

## SEND : Battery state of charge at the last period of the day(%)
model.SEND = Param (initialize = 3)


## RMCH = Battery maximum charging energy (kWh)
model.RMCH = Param (initialize = 5)

## RMDC =  Battery maximum discharging energy (kWh)
model.RMDC = Param (initialize = 5)

## EAPP = Energy consumption of appliance a (kWh)
model.EAPP = Param (model.Day, model.Appliance, within=NonNegativeReals)

## T = Required number of periods to operate appliance a
model.T = Param (model.Day, model.Appliance, within=NonNegativeReals)

## P = Time preference (0/1) to operate appliance a during period t
model.P = Param (model.Day, model.Appliance, model.Time, within=NonNegativeReals)

## PROB = Prob. of scenatio s happen
model.PROB = Param (model.SCEN, within=NonNegativeReals)

model.DATE = Param (model.Day)

dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_SCEN", set=model.SCEN)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_DTYPE", set=model.DTYPE)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Day", set=model.Day)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Day_WeeklyApp", set=model.Day_WeeklyApp)

dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Time", set=model.Time)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Appliance", set=model.Appliance)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Appliance_S", set=model.Appliance_S)
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_Appliance_F", set=model.Appliance_F)

dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_S", index=(model.SCEN, model.Day, model.Time), param=(model.S))

dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_EM", index=(model.Day, model.Time), param=(model.EM))
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_EAPP", index=(model.Day, model.Appliance), param=(model.EAPP))
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_T", index=(model.Day, model.Appliance), param=(model.T))

dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_P", index=(model.Day, model.Appliance, model.Time), param=(model.P))
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_PROB", index=(model.SCEN), param=(model.PROB))
dp.load(filename="C:/....directory.../2. 3rd_paper_pyomo_param.xlsx", range="model_DATE", index=(model.Day), param=(model.DATE))



### Set variables

## x_pv = number of basic size(0.315kW) PV array (ea) 
model.x_pv = Var(within = NonNegativeIntegers) 

## x_bat = Number of basic capacity(3.3kWh) Battery (ea)
model.x_bat = Var(within = NonNegativeIntegers) 

## EPV = Forecasted PV array output during period t (kWh) of day d of scanario s
model.EPV = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## eapp_bat = Energy discharged from the battery to operate the appliances during period t (kWh)
model.eapp_bat = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## echr = Energy from the PV array used to charge the battery during period t (kWh)
model.echr = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## eloss_pv = energy remaining after charging the battery and meeting the demand during time t on day i
model.eloss_pv = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## eapp_pv = Energy from the PV array to operate the appliances during period t (kWh)
model.eapp_pv = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## y_d = total amount of energy provided by diesel generator under scenario s (kWh)
model.y_d = Var (model.SCEN, within=NonNegativeReals)

## y_d =  amount of energy provided by diesel generator during period t of day d under scenario s (kWh)
model.e_dg = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## ebat = Available enrrgy in the battery during period t of day d  (kWh)
model.ebat = Var (model.SCEN, model.Day, model.Time, within=NonNegativeReals)

## ychr = Binary variable to indicate that the battery is charging during period t : binary (0,1)
model.ychr = Var (model.SCEN, model.Day, model.Time, within=Binary)

## ydch = Binary variable to indicate if the battery is discharging during period t : binary (0,1)
model.ydch = Var (model.SCEN, model.Day, model.Time, within=Binary)

## xapp_state = Binary variable to indicate the operating state of appliance a during period t : binary (0,1)
model.xapp_state = Var (model.SCEN, model.Day, model.Appliance_S, model.Time, within=Binary)

## xapp_end = Binary variable to indicate that appliance a finished operation at the end of period t : binary (0,1)
model.xapp_end = Var (model.SCEN, model.Day, model.Appliance_S, model.Time, within=Binary)

## z = Binary variable to determine if appliance a is operated or not during day d : binary (0,1)
#model.z = Var (model.Appliance, within=Binary)
  
## eiab = Initial amount of energy in the battery on day d under scenario s (kWh) (from day 2)
model.eiab = Var (model.SCEN, model.Day, within=NonNegativeReals)

model.d_fixed = Var (model.Day, model.Time, within=NonNegativeReals)



## Define obj.function : minimize total system cost
def obj_rule (model) :
    return model.C_PV * model.x_pv + model.C_BAT * model.x_bat + sum(model.PROB[s] * model.C_DG * model.y_d[s] for s in model.SCEN)
model.obj = Objective (rule=obj_rule, sense=minimize)



## Set Constraints

## PV Energy Output 1
def PVEnergyOutput1_rule (model, s, d, t) :
    if model.S[s,d,t] == 0 :
        return model.EPV[s,d,t] == 0
    return model.EPV[s,d,t] == model.SPV * model.S[s,d,t] * model.x_pv
model.PVEnergyOutput1 = Constraint (model.SCEN, model.Day, model.Time, rule=PVEnergyOutput1_rule)


## PV Energy Output 2
def PVEnergyOutput2_rule (model, s, d, t) :
    return model.EPV[s,d,t] == model.eapp_pv[s,d,t] + model.echr[s,d,t] + model.eloss_pv[s,d,t] 
model.PVEnergyOutput2 = Constraint (model.SCEN, model.Day, model.Time, rule=PVEnergyOutput2_rule)

## Total Energy Consumed
def TotalEnergyConsumtion_rule  (model, s,d, t) :
    return sum (model.EAPP[d,a] * model.xapp_state[s,d,a,t] for a in model.Appliance_S) + model.EM[d,t]  + sum (model.EAPP[d,a] * model.P[d,a,t] for a in model.Appliance_F) == (model.eapp_pv[s,d,t] + model.eapp_bat[s,d,t])*model.RINV + model.e_dg[s,d,t]
model.TotalEnergyConsumtion = Constraint (model.SCEN, model.Day, model.Time, rule=TotalEnergyConsumtion_rule) 

## Total Diesel Energy Consumtion
def  TotalDieselEnergyConsumtion_rule (model,s) :
    return model.y_d[s] == sum (model.e_dg[s,d,t] for d in model.Day for t in model.Time) 
model.TotalDieselEnergyConsumtion = Constraint (model.SCEN, rule = TotalDieselEnergyConsumtion_rule)

## Battery Charge and Discharge 1-1
def BatteryChargeandDischarge1_1_1rule (model,s) :
        return model.ebat[s,1,1] == model.EIAB*(1-model.RDCH) + (model.echr[s,1,1]*model.RCHR) - model.eapp_bat[s,1,1] 
model.BatteryChargeandDischarge1_1_1 = Constraint (model.SCEN, rule=BatteryChargeandDischarge1_1_1rule)

## Battery Charge and Discharge 1-1
def BatteryChargeandDischarge1_1_2rule (model,s,t) :
        return model.ebat[s,1,t] == model.ebat[s,1,t-1]*(1-model.RDCH) + (model.echr[s,1,t]*model.RCHR) - model.eapp_bat[s,1,t]
model.BatteryChargeandDischarge1_1_2 = Constraint (model.SCEN, model.ReducedTime_2, rule=BatteryChargeandDischarge1_1_2rule)

## Battery Charge and Discharge 1-2
def BatteryChargeandDischarge1_2_1rule (model,s,d) :
        return model.ebat[s,d,1] == model.eiab[s,d]*(1-model.RDCH) + (model.echr[s,d,1]*model.RCHR) - model.eapp_bat[s,d,1] 
model.BatteryChargeandDischarge1_2_1 = Constraint (model.SCEN, model.ReducedDay_2, rule=BatteryChargeandDischarge1_2_1rule)

## Battery Charge and Discharge 1-2
def BatteryChargeandDischarge1_2_2rule (model,s,d,t) :
        return model.ebat[s,d,t] == model.ebat[s,d,t-1]*(1-model.RDCH) + (model.echr[s,d,t]*model.RCHR) - model.eapp_bat[s,d,t]
model.BatteryChargeandDischarge1_2_2 = Constraint (model.SCEN, model.ReducedDay_2, model.ReducedTime_2, rule=BatteryChargeandDischarge1_2_2rule)

## Battery Charge and Discharge 2
def BatteryConstraint2_rule (model,s,d,t) :
    return model.SMIN * model.BCAP * model.x_bat <= model.ebat[s,d,t]
model.BatteryConstraint2 = Constraint (model.SCEN, model.Day, model.Time, rule=BatteryConstraint2_rule)

## Battery Charge and Discharge 3
def BatteryConstraint3_rule (model,s,d,t) :
    return model.ebat[s,d,t] <= model.SMAX * model.BCAP * model.x_bat
model.BatteryConstraint3 = Constraint (model.SCEN, model.Day, model.Time, rule=BatteryConstraint3_rule)

## Battery Charge and Discharge 4
def BatteryConstraint4_rule (model,s,d,t) :
    return model.ychr[s,d,t] + model.ydch[s,d,t] <= 1 
model.BatteryConstraint4 = Constraint (model.SCEN, model.Day, model.Time, rule=BatteryConstraint4_rule)

## Battery Charge and Discharge 5
def BatteryConstraint5_rule (model,s,d,t) :
    return model.eapp_bat[s,d,t] <= model.ydch[s,d,t] * model.RMDC 
model.BatteryConstraint5 = Constraint (model.SCEN, model.Day, model.Time, rule=BatteryConstraint5_rule)

## Battery Charge and Discharge 6
def BatteryConstraint6_rule (model,s,d,t) :
    return model.echr[s,d,t] <= model.ychr[s,d,t] * model.RMCH 
model.BatteryConstraint6 = Constraint (model.SCEN, model.Day, model.Time, rule=BatteryConstraint6_rule)

## Battery Charge and Discharge 7
def BatteryConstraint7_rule (model,s,d) :
    return model.ebat[s,d,24] >= model.SEND 
model.BatteryConstraint7 = Constraint (model.SCEN, model.Day, rule=BatteryConstraint7_rule)

## Battery Charge and Discharge 8
def BatteryConstraint8_rule (model,s,d) :
    return model.ebat[s,d,24] == model.eiab[s,d+1] 
model.BatteryConstraint8 = Constraint (model.SCEN, model.ReducedDay_1, rule=BatteryConstraint8_rule)


### Scheduling Constraints ###


## Appliance Daily Operation 1
def ApplianceDailyOperation1_rule (model, s, d, a) :
    return sum (model.xapp_state[s,d,a,t] for t in model.Time) == model.T[d,a] 
model.ApplianceDailyOperation1 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, rule=ApplianceDailyOperation1_rule)

## Appliance Daily Operation 1
def ApplianceDailyOperation2_rule (model, s, d, a) :
    return sum (model.xapp_state[s,d,3,t] for t in model.Time) == model.T[d,3] 
model.ApplianceDailyOperation2 = Constraint (model.SCEN, model.Day, model.Appliance_S, rule=ApplianceDailyOperation2_rule)

## Operation Period Preference 
def OperationPeriodPreference1_rule (model,s,d, a, t) :
    return model.xapp_state[s,d,a,t] <= model.P[d,a,t] 
model.OperationPeriodPreference1 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, model.Time, rule=OperationPeriodPreference1_rule)

def OperationPeriodPreference2_rule (model,s,d, t) :
    return model.xapp_state[s,d,3,t] <= model.P[d,3,t] 
model.OperationPeriodPreference2 = Constraint (model.SCEN, model.Day, model.Time, rule=OperationPeriodPreference2_rule)

## Uninterruptible Operation Constraints 3-1-1
def UninterruptibleOp3_1_1_rule (model, s, d, a, t) :
    return model.xapp_state[s,d,a,t] <= 1-model.xapp_end[s,d,a,t] 
model.UninterruptibleOp3_1_1 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, model.Time, rule=UninterruptibleOp3_1_1_rule)

## Uninterruptible Operation Constraints 3-1-2
def UninterruptibleOp3_1_2_rule (model,s, d, t) :
    return model.xapp_state[s,d,3,t] <= 1-model.xapp_end[s,d,3,t] 
model.UninterruptibleOp3_1_2 = Constraint (model.SCEN, model.Day, model.Time, rule=UninterruptibleOp3_1_2_rule)

## Uninterruptible Operation Constraints 3-2-1
def UninterruptibleOp3_2_1_rule (model, s,d, a, t) :
    return model.xapp_state[s,d,a,t] - model.xapp_state[s,d,a,t+1] <= model.xapp_end[s,d,a,t+1] 
model.UninterruptibleOp3_2_1 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, model.ReducedTime_1, rule=UninterruptibleOp3_2_1_rule)

## Uninterruptible Operation Constraints 3-2-2
def UninterruptibleOp3_2_2_rule (model, s,d, t) :
    return model.xapp_state[s,d,3,t] - model.xapp_state[s,d,3,t+1] <= model.xapp_end[s,d,3,t+1] 
model.UninterruptibleOp3_2_2 = Constraint (model.SCEN, model.Day,  model.ReducedTime_1, rule=UninterruptibleOp3_2_2_rule)

## Uninterruptible Operation Constraints 3-3-1
def UninterruptibleOp3_3_1_rule (model, s,d, a, t) :
    return model.xapp_end[s,d,a,t] <= model.xapp_end[s,d,a,t+1] 
model.UninterruptibleOp3_3_1 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, model.ReducedTime_1, rule=UninterruptibleOp3_3_1_rule)

## Uninterruptible Operation Constraints 3-3-2
def UninterruptibleOp3_3_2_rule (model, s,d, t) :
    return model.xapp_end[s,d,3,t] <= model.xapp_end[s,d,3,t+1] 
model.UninterruptibleOp3_3_2 = Constraint (model.SCEN, model.Day,  model.ReducedTime_1, rule=UninterruptibleOp3_3_2_rule)


## Sequential Processing of Interruptible Appliances Constraints, when appliance 1 starts earlier than applaince 2

def xapp_endProcess1_rule (model, s, d):
    return (model.xapp_state[s,d,2,1]) == 0
model.xapp_endProcess1 = Constraint (model.SCEN, model.Day_WeeklyApp, rule=xapp_endProcess1_rule)

def xapp_endProcess2_rule (model, s, d, a, t) : 
    return model.T[1,1] - sum (model.xapp_state[s,d,1,n] for n in range(1, t+1)) <=  model.M*(1-model.xapp_state[s,d,2,t+1]) 
model.xapp_endProcess2 = Constraint (model.SCEN, model.Day_WeeklyApp, model.Appliance_S_1, model.ReducedTime_1, rule=xapp_endProcess2_rule)


    
from pyomo.opt import SolverFactory
opt = SolverFactory('gurobi')
opt.options['mipgap'] = 0.01
opt.options['mipgapabs'] = 5.0
#opt.options['mipdisplay'] = 2

#instant.pprint()
instance = model.create_instance(dp)                               

#instance = model.create_instance("C:/Users/David_Cho/OneDrive - Auburn University/Research/3rd_paper/AMPL_code/3rd_paper_pyomo.dat")                               
#instance.pprint()

results = opt.solve(instance, tee=True)


###########################################################################
### If you'd like to, you could use Neo Server to solve ##
opt = SolverFactory('cplex')  # Select solver
solver_manager = SolverManagerFactory('neos')  # Solve in neos server
results = solver_manager.solve(instance, opt=opt, tee=True)
###########################################################################


results.write(num=3)

for s in instance.SCEN :
    print(instance.y_d[s].value)

for v in instance.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(instance, str(v))
    for index in varobject:
        print ("   ",index, varobject[index].value)
        
