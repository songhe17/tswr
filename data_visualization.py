#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:58:32 2020

@author: songhewang
"""

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import datetime
import os
date = datetime.date(2017,1,25)

wd = date.weekday()


path = 'data/apple/County/California County.csv'
data = pd.read_csv(path)
keys = data.keys()[6:]
numbers = []
#all_counties = ['Orange', 'Los Angeles', 'Santa Clara', 'San Francisco', 'San Diego', 'Humboldt', 'Sacramento', 'Solano', 'Marin', 'Napa', 'Sonoma', 'Alameda', 'Placer', 'San Mateo', 'Contra Costa', 'Yolo', 'Fresno', 'Madera', 'Riverside', 'Santa Cruz', 'Shasta', 'San Joaquin', 'Ventura', 'Stanislaus', 'Tulare', 'San Benito', 'San Luis Obispo', 'San Bernardino', 'Santa Barbara', 'Nevada', 'Kern', 'Monterey', 'Mendocino', 'Amador', 'Imperial', 'Butte', 'El Dorado', 'Siskiyou', 'Yuba', 'Calaveras', 'Merced', 'Mono', 'Inyo', 'Sutter', 'Colusa', 'Kings', 'Glenn', 'Tuolumne', 'Alpine', 'Plumas', 'Del Norte', 'Tehama', 'Lake', 'Mariposa', 'Trinity', 'Sierra', 'Lassen']
all_counties = []


flow_dict = {}
for i in data.iloc:

    county = i['region'][:-7]
    all_counties.append(county)
    keys = i.keys()[6:]
    for key in keys:
        if key not in flow_dict:
            flow_dict[key] = {}
        flow_dict[key][county] = i[key]
            
print(all_counties)

dump = False
if dump:
    case_path = 'data/covid_info/us-counties.csv'
    case_data = pd.read_csv(case_path)
    case_dict = {}
    
    for i in tqdm(case_data.iloc):
        state = i['state']
        case = i['cases']
        death = i['deaths']
        date = i['date']
        date = date.replace('-', '/')
        date = date[6:] + '/' + date[:4] 
    
        county = i['county']
        if state != 'California' or county not in all_counties or county == 'Unknown':
            continue
        if date not in case_dict:
            case_dict[date] = {c:[0,0] for c in all_counties}
            
        case_dict[date][county] = [case, death]
        
    with open('data/preprocessed/cases.pkl','wb') as f:
        pickle.dump(case_dict, f)

#print(case_dict)
with open('data/preprocessed/cases.pkl','rb') as f:
    case_dict = pickle.load(f)
    
for date, value in case_dict.items():

    value_sum = sum([value[i][0] for i in value])

    value = {i:value[i]/value_sum for i in value}

    case_dict[date] = value
    
#print(case_dict)
county_info_path = 'data/county_basic_info/selected_data_norm.csv'
county_info_data = pd.read_csv(county_info_path)
reg_keys = ['population_density_per_sqmi',
            'percent_smokers', 'percent_adults_with_obesity',
            'percent_excessive_drinking', 'percent_uninsured',
            'percent_unemployed_CHR',
            'violent_crime_rate','life_expectancy',
            'percent_65_and_over', 'per_capita_income', 'percent_below_poverty']
county_dict = {}
for ele in county_info_data.iloc:
    county = ele['county']
    if county not in all_counties:
        continue
    county_dict[county] = [ele[factor] for factor in reg_keys]
    
selected_county = 'Los Angeles'
save = {'x':[], 'x_other':[], 'c':[], 'c_other':[], 'weekday':[], 'flow':[], 'google':[]}
for date, flow_value in flow_dict.items():
    if date not in case_dict:
        continue
    split_date = date.split('/')
    if int(split_date[0]) > 4 or int(split_date[0]) < 3:
        continue
    wd = datetime.date(int(split_date[2]),int(split_date[0]),int(split_date[1])).weekday()
    weekday = [[0.0] * 7]
    weekday[0][wd] = 1.0
    covid_value = case_dict[date]
    #print(covid_value)
    c = [[covid_value[selected_county][0]]]
    c_other = [[covid_value[key][0] for key in covid_value if key != selected_county]]
    x = [county_dict[selected_county]]
    x_other = np.transpose([county_dict[key] for key in county_dict if key != selected_county]).tolist()
    flow = [flow_value[selected_county]]
    for key in save:
        save[key].append(eval(key))

# =============================================================================
# print(len(save['x']))
# #plt.plot(save['flow'])
# reshaped_c = np.reshape(save['c'],(-1))
# 
# first_der = []
# for i in range(1,len(save['c'])-1):
#     if reshaped_c[i] == 0.0 or reshaped_c[i] == 1.0:
#         continue
#     print(reshaped_c[i])
#     #der = (save['c'][i][0][0] - save['c'][i-1][0][0]) / sum(save['c'][:i][0][0])
#     der = (reshaped_c[i] - reshaped_c[i-1]) / sum(reshaped_c[:i])
# 
#     first_der.append(der)
#     
# plt.plot(first_der)
# 
# second_der = []
# for i in range(1,len(first_der)-1):
#     der = (first_der[i+1] - first_der[i]) / first_der[i] / 10
#     second_der.append(der)
# plt.plot(second_der)
# =============================================================================
#plt.plot(reshaped_c)
save_path = f'data/preprocessed/input_{selected_county}.pkl'
with open(save_path,'wb') as f:
    pickle.dump(save,f)

root = 'google trend/California/'
for file in os.listdir(root):
    path = os.path.join(root, file)
    data = pd.read_csv(path)
    print(data.keys())
    break
    