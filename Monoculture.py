import numpy as np 
import FlowCytometryTools
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import FCPlate
from FlowCytometryTools import ThresholdGate, PolyGate
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import datetime as dt

#set thresholds
FSCHgateupper = ThresholdGate(6550.0, ['FSC-H'], region='below')
FSCHgatelower = ThresholdGate(5100.0, ['FSC-H'], region='above')
SSCHgatelower = ThresholdGate(4800.0, ['SSC-H'], region='above') 
SSCHgateupper = ThresholdGate(5600.0, ['SSC-H'], region='below')
scatter_gates = FSCHgateupper & FSCHgatelower & SSCHgatelower & SSCHgateupper

bfp_threshold = [[{"A1":7425,"A3":7425,"C1":7425,"C3":7425,"B1":7570,"B3":7570,"D1":7650,"D3":7750,"E1":7550,"E3":7495}],
                 [{"A1":7425,"A3":7425,"C1":7475,"C3":7475,"B1":7570,"B3":7570,"D1":7600,"D3":7600,"E1":7600,"E3":7600},
                  {"A1":7425,"A3":7425,"C1":7425,"C3":7450,"B1":7570,"B3":7570,"D1":7600,"D3":7700,"E1":7600,"E3":7500},
                  {"A1":7450,"A3":7425,"C1":7450,"C3":7450,"B1":7600,"B3":7600,"D1":7800,"D3":7800,"E1":7600,"E3":7500}]]
yfp_threshold = 6150

days = list(range(2))
plate_num = list(range(1,4))
plates = {}

for day in days:
    list_plates = []
    datadir = '/Users/kristenlok/Desktop/2.8.22_monoculture_flucts/D{}/'.format(str(day))
    if day == 1:
        for p in plate_num:
            datadir = '/Users/kristenlok/Desktop/2.8.22_monoculture_flucts/D1/p{}/'.format(str(p))
            plate = FCPlate.from_dir(ID='96-well', path=datadir, parser='name')
            plate_transformed = plate.transform('hlog', channels=['BFP-H','YFP-H'], b=100.0).transform('hlog', channels=['FSC-H','SSC-H'], b=10.0**2).gate(scatter_gates)
            list_plates.append(plate_transformed)
    else:
        plate = FCPlate.from_dir(ID='96-well', path=datadir, parser='name')
        plate_transformed = plate.transform('hlog', channels=['BFP-H','YFP-H'], b=100.0).transform('hlog', channels=['FSC-H','SSC-H'], b=10.0**2).gate(scatter_gates)
        list_plates.append(plate_transformed) 
    plates[day] = list_plates
    
def f1_expectation(theta,f1obs_data,f2obs_data,nmax=100):
    f1,lam = theta
    f2=1-f1
    f1obs=0
    f2obs=0
    for k in range(1,nmax):
        f1obs+=((f1*lam)**k)/scipy.special.factorial(k)/(np.exp(lam)-1)
        f2obs+=((f2*lam)**k)/scipy.special.factorial(k)/(np.exp(lam)-1)
    return f1obs-f1obs_data, f2obs-f2obs_data


def get_bfp_freq(data, day, plate_num, well):
    bfp_pos = data['BFP-H']>bfp_threshold[day][plate_num][well]
    yfp_pos = data['YFP-H']>yfp_threshold
    bfp_neg = data['BFP-H']<bfp_threshold[day][plate_num][well]
    yfp_neg = data['YFP-H']<yfp_threshold

    bfp = len(data[bfp_pos & yfp_neg])
    yfp = len(data[yfp_pos & bfp_neg])
    dp = len(data[bfp_pos & yfp_pos])
    Q = bfp + yfp + dp
    func = lambda theta: f1_expectation(theta,bfp/Q,yfp/Q)
    res = sp.optimize.root(func, [bfp/Q, 0.5])

    bfp_mean = np.float(data[bfp_pos & yfp_neg]['BFP-H'].mean())
    yfp_mean = np.float(data[yfp_pos & bfp_neg]['YFP-H'].mean())
    return res.x, Q, bfp_mean, yfp_mean


#set up groups based on wells
b_data=[]

cols = [1,3]
rows = ['A','B','C','D','E']

for day in days:
    for row in rows:
        for col in cols:
            for plate in plates[day]:
                well = row + str(col)
                well_sort = str(col) + row
                try:
                    res,Q,bfp,yfp = get_bfp_freq(plate[well].data, day, plates[day].index(plate), well)
                except:
                    continue
                if day == 0 and col == 1:
                    group = "A-G1-3"
                if day == 0 and col == 3:  
                    group = "A-E4-6"
                if day == 1 and plates[day].index(plate)+1 == 1:
                    if row == 'A':
                        group = "A-G1-3_A"
                    if row == 'B':
                        group = "A-G1-3_B"
                    if row == 'C':
                        group = "A-G1-3_C"
                    if row == 'D':
                        group = "A-G1-3_D"
                    if row == 'E':
                        group = "A-G1-3_E"
                if day == 1 and plates[day].index(plate)+1 == 2:
                    if col == 1:
                        if row == 'A':
                            group = "A-G1-3_A"
                        if row == 'B':
                            group = "A-G1-3_B"
                        if row == 'C':
                            group = "A-G1-3_C"
                        if row == 'D':
                            group = "A-G1-3_D"
                        if row == 'E':
                            group = "A-G1-3_E"
                    if col == 3:
                        if row == 'A':
                            group = "A-E4-6_A"
                        if row == 'B':
                            group = "A-E4-6_B"
                        if row == 'C':
                            group = "A-E4-6_C"
                        if row == 'D':
                            group = "A-E4-6_D"
                        if row == 'E':
                            group = "A-E4-6_E"
                if day == 1 and plates[day].index(plate)+1 == 3:
                    if row == 'A':
                        group = "A-E4-6_A"
                    if row == 'B':
                        group = "A-E4-6_B"
                    if row == 'C':
                        group = "A-E4-6_C"
                    if row == 'D':
                        group = "A-E4-6_D"
                    if row == 'E':
                        group = "A-E4-6_E"
                    
                b_data.append({
                    'day':day,
                    'plate':plates[day].index(plate)+1,
                    'well':well,
                    'sort by well': well_sort,
                    'group': group,
                    'S freq': res[0],
                    'lambda': res[1],
                    'total count':Q,
                    'corrected total count': Q*res[1]/(1-np.exp(-res[1])),
                    'BFP':bfp,
                    'YFP':yfp,
                })     
            
df = pd.DataFrame(b_data)
#perform variance stabilizing transform
df['f_trafo'] = np.arcsin(2*df['S freq'] - 1) 
df.sort_values(['day','plate','sort by well'])

#calculate variance
df_var = df[df['day']==1].groupby(['group']).var().reset_index() 
title_dict = {"A-G1-3_A":"REL 606 BFP:REL 606 YFP 50:50, daily mix", 
              "A-G1-3_B":"S BFP:S YFP 50:50, daily mix",
              "A-G1-3_C":"Sl1 BFP:Sl1YFP 50:50, daily mix", 
              "A-G1-3_D":"S BFP:L YFP 50:50, daily mix",
              "A-G1-3_E":"S BFP:Sl1 YFP 50:50, daily mix",
              "A-E4-6_A":"REL 606 BFP:REL 606 YFP 50:50, premixed",
              "A-E4-6_B":"S BFP:S YFP 50:50, premixed",
              "A-E4-6_C":"Sl1 BFP:Sl1YFP 50:50, premixed",
              "A-E4-6_D":"S BFP:L YFP 5:95, premixed",
              "A-E4-6_E":"S BFP:Sl1 YFP 95:5, premixed"}
df_var['group'] = df_var['group'].replace(title_dict)

#plot var(f_trafo) of Premixed vs. Daily Mix
labels = ["R+R", "S+S", "Sl+Sl", "S+L", "S+Sl"]  
colors = ["red", "purple", "orange", "green", "blue"]

daily_var = df_var['f_trafo'][5:]
premix_var = df_var['f_trafo'][:5]
plt.scatter(daily_var, premix_var, c=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.008)
plt.ylim(0, 0.008)
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)
plt.plot([0,0.008],[0,0.008],'--')
plt.show()

#CIs
def CI(alpha):
    n = 3
    a = sp.stats.chi2.isf(1-alpha/2, n-1)
    b = sp.stats.chi2.isf(alpha/2, n-1)
    df_var['u'] = ((n-1)*df_var['f_trafo'])/a - df_var['f_trafo']
    df_var['l'] = df_var['f_trafo'] - ((n-1)*df_var['f_trafo'])/b

#plot with 68% CI
CI(0.32)
labels = ["R+R", "S+S", "Sl+Sl", "S+L", "S+Sl"]  
colors = ["red", "purple", "orange", "green", "blue"]

daily_var = df_var['f_trafo'][5:]
premix_var = df_var['f_trafo'][:5]
plt.scatter(daily_var, premix_var, c=colors)
plt.errorbar(daily_var, premix_var, 
             yerr = (df_var['l'][:5].to_list(),df_var['u'][:5].to_list()), #fmt='o')  
             xerr = (df_var['l'][5:].to_list(),df_var['u'][5:].to_list()), fmt='none', ecolor=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix with 68% CI")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.01)
plt.ylim(0, 0.001)
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)
plt.plot([0,0.01],[0,0.01],'--')
plt.show()
#show all: (0.04,0.004)

#plot with 95% CI
CI(0.05)
labels = ["R+R", "S+S", "Sl+Sl", "S+L", "S+Sl"]  
colors = ["red", "purple", "orange", "green", "blue"]

daily_var = df_var['f_trafo'][5:]
premix_var = df_var['f_trafo'][:5]
plt.scatter(daily_var, premix_var)
plt.errorbar(daily_var, premix_var, 
             yerr = (df_var['l'][:5].to_list(),df_var['u'][:5].to_list()), #fmt='o') 
             xerr = (df_var['l'][5:].to_list(),df_var['u'][5:].to_list()), fmt='none', ecolor=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix with 95% CI")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.01)
plt.ylim(0, 0.01)
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)
plt.plot([0,0.01],[0,0.01],'--')
plt.show()
#view all: (0.01, 0.01)