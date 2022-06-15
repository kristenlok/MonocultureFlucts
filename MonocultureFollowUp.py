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


bfp_threshold = [[{"A1":7535, "B1":7650, "C1":7540, "D1":7600, "E1":7650, "F1":7600, "A3":7650, "B3":7650, "C3":7650, "A5":7500, "B5":7600, "C5":7500, "D5":7650, "E5":7550, "F5":7650, "A7":7600, "B7":7650, "C7":7600, "A9":7500, "B9":7600, "C9":7500, "D9":7575, "E9":7575, "F9":7675, "A11":7575, "B11":7600, "C11":7600}, 
                  {"A1":7525, "B1":7700, "C1":7550, "D1":7700, "E1":7675, "F1":7900, "A3":7600, "B3":7700, "C3":7650}], 
                 [{"A1":7500, "B1":7575, "C1":7500, "D1":7650, "E1":7600, "F1":7650, "A3":7500, "B3":7650, "C3":7550, "D3":7650, "E3":7600, "F3":7725, "A5":7500, "B5":7650, "C5":7550, "D5":7650, "E5":7650, "F5":7675, "A7":7450, "B7":7600, "C7":7500, "D7":7600, "E7":7600, "F7":7700, "A9":7500, "B9":7650, "C9":7550, "D9":7600, "E9":7600, "F9":7650, "A11":7475, "B11":7600, "C11":7525, "D11":7600, "E11":7600, "F11":7650}, 
                  {"A1":7650, "B1":7700, "C1":7600, "D1":7650, "E1":7675, "F1":7675, "A3":7650, "B3":7675, "C3":7650, "D3":7600, "E3":7675, "F3":7675, "A5":7675, "B5":7700, "C5":7600, "D5":7650, "E5":7625, "F5":7600, "A7":7600, "B7":7615, "C7":7675, "D7":7675, "E7":7650, "F7":7600, "A9":7600, "B9":7725, "C9":7600, "D9":7600, "E9":7700, "F9":7700, "A11":7600, "B11":7700, "C11":7700, "D11":7750, "E11":7700, "F11":7800}, 
                  {"A1":7500, "B1":7675, "C1":7525, "D1":7700, "E1":7650, "F1":7650, "A3":7500, "B3":7650, "C3":7550, "D3":7700, "E3":7600, "F3":7700, "A5":7500, "B5":7625, "C5":7500, "D5":7700, "E5":7700, "F5":7800, "A7":7500, "B7":7600, "C7":7450, "D7":7600, "E7":7600, "F7":7700, "A9":7500, "B9":7650, "C9":7500, "D9":7700, "E9":7650, "F9":7800, "A11":7500, "B11":7700, "C11":7500, "D11":7650, "E11":7650, "F11":7750}]]
yfp_threshold = 6100

days = list(range(2))
plates = {}
for day in days:
    list_plates = []
    datadir = '/Users/kristenlok/Desktop/5.22.22_monoculture_flucts/D{}/'.format(str(day))
    if day == 0:
        for p in list(range(1,3)):
            datadir = '/Users/kristenlok/Desktop/5.22.22_monoculture_flucts/D0/p{}/'.format(str(p))
            plate = FCPlate.from_dir(ID='96-well', path=datadir, parser='name')
            plate_transformed = plate.transform('hlog', channels=['BFP-H','YFP-H'], b=100.0).transform('hlog', channels=['FSC-H','SSC-H'], b=10.0**2).gate(scatter_gates)
            list_plates.append(plate_transformed)
    else:
        for p in list(range(1,4)):
            datadir = '/Users/kristenlok/Desktop/5.22.22_monoculture_flucts/D1/p{}/'.format(str(p))
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

cols = [1,3,5,7,9,11]
rows = ['A','B','C','D','E','F']

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
                if day == 0 and plates[day].index(plate)+1 == 1: 
                    if col == 1 or col == 5 or col == 9:
                        if row == 'A':
                            group = "A-FDaily_A"
                        if row == 'B':
                            group = "A-FDaily_B"
                        if row == 'C':
                            group = "A-FDaily_C"
                        if row == 'D':
                            group = "A-FDaily_D"
                        if row == 'E':
                            group = "A-FDaily_E"
                        if row == 'F':
                            group = "A-FDaily_F"    
                    if col == 3 or col == 7 or col == 11:  
                        if row == 'A':
                            group = "G-IDaily_G"
                        if row == 'B':
                            group = "G-IDaily_H"
                        if row == 'C':
                            group = "G-IDaily_I"
                if day == 0 and plates[day].index(plate)+1 == 2:
                    if col == 1:
                        group = "A-FPreM"
                    if col == 3:  
                        group = "G-IPreM"
                if day == 1 and plates[day].index(plate)+1 == 1:
                    if row == 'A':
                        group = "DlyA-F_A"
                    if row == 'B':
                        group = "DlyA-F_B"
                    if row == 'C':
                        group = "DlyA-F_C"
                    if row == 'D':
                        group = "DlyA-F_D"
                    if row == 'E':
                        group = "DlyA-F_E"
                    if row == 'F':
                        group = "DlyA-F_F"
                if day == 1 and plates[day].index(plate)+1 == 2:
                    if row == 'A':
                        group = "DlyG-I_G"
                    if row == 'B':
                        group = "DlyG-I_H"
                    if row == 'C':
                        group = "DlyG-I_I"
                    if row == 'D':
                        group = "PreG-I_G"
                    if row == 'E':
                        group = "PreG-I_H"
                    if row == 'F':
                        group = "PreG-I_I"
                if day == 1 and plates[day].index(plate)+1 == 3:
                    if row == 'A':
                        group = "PreA-F_A"
                    if row == 'B':
                        group = "PreA-F_B"
                    if row == 'C':
                        group = "PreA-F_C"
                    if row == 'D':
                        group = "PreA-F_D"
                    if row == 'E':
                        group = "PreA-F_E"
                    if row == 'F':
                        group = "PreA-F_F"
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

#perform variance stabilizing transformation
df = pd.DataFrame(b_data)
df['f_trafo'] = np.arcsin(2*df['S freq'] - 1) 
df.sort_values(['day','plate','sort by well'])

#calculate variance for day0 (technical variance - between same culture)
df_d0 = df.loc[(df['day']==0) & (df['plate']==1)]
df_d0.sort_values(['plate','sort by well'])
df_vard0 = df_d0.groupby(['group']).var().reset_index() 

#calculate variance for day1 (biological variance - between replicates)
df_d1 = df[df['day']==1]
df_d1.sort_values(['plate','sort by well'])
df_vard1 = df_d1.groupby(['group']).var().reset_index() 

title_dict = {"DlyA-F_A":"REL 606 BFP:REL 606 YFP 50:50, daily mix", 
              "DlyA-F_B":"S BFP:S YFP 50:50, daily mix",
              "DlyA-F_C":"Sl1 BFP:Sl1YFP 50:50, daily mix", 
              "DlyA-F_D":"S BFP:L YFP 50:50, daily mix",
              "DlyA-F_E":"S BFP:L YFP 95:5, daily mix",
              "DlyA-F_F":"S BFP:L YFP 5:95, daily mix", 
              "DlyG-I_G":"S BFP:Sl1 YFP 50:50, daily mix",
              "DlyG-I_H":"S BFP:Sl1 YFP 5:95, daily mix",
              "DlyG-I_I":"S BFP:Sl1 YFP 95:5, daily mix",
              "PreA-F_A":"REL 606 BFP:REL 606 YFP 50:50, premixed", 
              "PreA-F_B":"S BFP:S YFP 50:50, premixed",
              "PreA-F_C":"Sl1 BFP:Sl1YFP 50:50, premixed", 
              "PreA-F_D":"S BFP:L YFP 50:50, premixed",
              "PreA-F_E":"S BFP:L YFP 95:5, premixed",
              "PreA-F_F":"S BFP:L YFP 5:95, premixed", 
              "PreG-I_G":"S BFP:Sl1 YFP 50:50, premixed",
              "PreG-I_H":"S BFP:Sl1 YFP 5:95, premixed",
              "PreG-I_I":"S BFP:Sl1 YFP 95:5, premixed"}
df_vard1['group'] = df_vard1['group'].replace(title_dict)

#plot var(f_trafo) of Premixed vs. Daily Mix on d1
labels = ["R+R", "S+S", "Sl+Sl", "S+L 50:50", "S+L 95:5", "S+L 5:95", "S+Sl 50:50", "S+Sl 5:95", "S+Sl 95:5"]  
colors = ["red", "purple", "orange", "green", "green", "green", "blue", "blue", "blue"]

premix_var = df_vard1['f_trafo'][9:]
daily_var = df_vard1['f_trafo'][:9]
plt.scatter(daily_var, premix_var,c=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.005)
plt.ylim(0, 0.005) #0.005
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)
plt.plot([0,0.005],[0,0.005],'--')
plt.show()

#CIs
def CI(alpha, df_var):
    n = 6
    a = sp.stats.chi2.isf(1-alpha/2, n-1)
    b = sp.stats.chi2.isf(alpha/2, n-1)
    df_var['u'] = ((n-1)*df_var['f_trafo'])/a - df_var['f_trafo']
    df_var['l'] = df_var['f_trafo'] - ((n-1)*df_var['f_trafo'])/b

#plot with 68% CI
CI(0.32, df_vard1)
labels = ["R+R", "S+S", "Sl+Sl", "S+L 50:50", "S+L 95:5", "S+L 5:95", "S+Sl 50:50", "S+Sl 5:95", "S+Sl 95:5"]  
colors = ["red", "purple", "orange", "green", "green", "green", "blue", "blue", "blue"]


premix_var = df_vard1['f_trafo'][9:]
daily_var = df_vard1['f_trafo'][:9]
plt.scatter(daily_var, premix_var, c=colors)
plt.errorbar(daily_var, premix_var, 
             xerr = (df_vard1['l'][:9].to_list(),df_vard1['u'][:9].to_list()), #fmt='o')  
             yerr = (df_vard1['l'][9:].to_list(),df_vard1['u'][9:].to_list()), fmt='none', ecolor=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix with 68% CI")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.005) #0.01
plt.ylim(0, 0.001) #0.005
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)

plt.plot([0,0.005],[0,0.005],'--')
plt.show()


#plot with 95% CI
CI(0.05, df_vard1)
labels = ["R+R", "S+S", "Sl+Sl", "S+L 50:50", "S+L 95:5", "S+L 5:95", "S+Sl 50:50", "S+Sl 5:95", "S+Sl 95:5"]  
colors = ["red", "purple", "orange", "green", "green", "green", "blue", "blue", "blue"]

premix_var = df_vard1['f_trafo'][9:]
daily_var = df_vard1['f_trafo'][:9]
plt.scatter(daily_var, premix_var, c=colors)
plt.errorbar(daily_var, premix_var, 
             xerr = (df_vard1['l'][:9].to_list(),df_vard1['u'][:9].to_list()), #fmt='o')  
             yerr = (df_vard1['l'][9:].to_list(),df_vard1['u'][9:].to_list()), fmt='none', ecolor=colors)

plt.title("var(f_{trafo}) of Premixed vs. Daily Mix with 95% CI")
plt.xlabel("Daily Mix")
plt.ylabel("Premixed")
plt.xlim(0, 0.025) #0.01
plt.ylim(0, 0.003) #0.005
for daily, pre, label in zip(daily_var, premix_var, labels):
    plt.text(daily, pre, label)
plt.plot([0,0.005],[0,0.005],'--')
plt.show()