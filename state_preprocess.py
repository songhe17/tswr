#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:53:17 2020

@author: songhewang
"""

import os
import pandas as pd
import numpy as np
import datetime
import pickle
from tqdm import tqdm


path = 'data/apple/State/Apple Mobility States.csv'
data = pd.read_csv(path)
keys = data.keys()[6:]
numbers = []
#all_counties = ['Orange', 'Los Angeles', 'Santa Clara', 'San Francisco', 'San Diego', 'Humboldt', 'Sacramento', 'Solano', 'Marin', 'Napa', 'Sonoma', 'Alameda', 'Placer', 'San Mateo', 'Contra Costa', 'Yolo', 'Fresno', 'Madera', 'Riverside', 'Santa Cruz', 'Shasta', 'San Joaquin', 'Ventura', 'Stanislaus', 'Tulare', 'San Benito', 'San Luis Obispo', 'San Bernardino', 'Santa Barbara', 'Nevada', 'Kern', 'Monterey', 'Mendocino', 'Amador', 'Imperial', 'Butte', 'El Dorado', 'Siskiyou', 'Yuba', 'Calaveras', 'Merced', 'Mono', 'Inyo', 'Sutter', 'Colusa', 'Kings', 'Glenn', 'Tuolumne', 'Alpine', 'Plumas', 'Del Norte', 'Tehama', 'Lake', 'Mariposa', 'Trinity', 'Sierra', 'Lassen']
all_counties = []


flow_dict = {}
for i in data.iloc:

    county = i['region']
    all_counties.append(county)
    keys = i.keys()[6:]

    for key in keys:
        if key not in flow_dict:
            flow_dict[key] = {}
        flow_dict[key][county] = i[key] / 100

        
# county_info_path = 'data/county_basic_info/selected_data_norm.csv'
# county_info_data = pd.read_csv(county_info_path)
# county_dict = {}
# reg_keys = ['total_population', 'population_density_per_sqmi',
#             'percent_smokers', 'percent_adults_with_obesity',
#             'percent_excessive_drinking', 'percent_uninsured',
#             'percent_unemployed_CHR',
#             'violent_crime_rate','life_expectancy',
#             'percent_65_and_over', 'per_capita_income', 'percent_below_poverty']

# for ele in county_info_data.iloc:
#     county = ele['county']
#     if county not in all_counties:
#         continue
#     county_dict[county] = [ele[factor] for factor in reg_keys]
# sum_population = sum([county_dict[county][0] for county in county_dict])
# for county in county_dict:
#     county_dict[county].append(county_dict[county][0] / sum_population)

    
dump = True
if dump:
    case_path = 'data/covid_info/us-states.csv'
    case_data = pd.read_csv(case_path)
    case_dict = {}
    
    for i in tqdm(case_data.iloc):
        state = i['state']
        case = i['cases']
        death = i['deaths']
        date = i['date']
        date = date.replace('-', '/')
        date = date[6:] + '/' + date[:4] 
    
        county = i['state']
        if county not in all_counties or county == 'Unknown':
            continue
        if date not in case_dict:
            case_dict[date] = {c:[0,0] for c in all_counties}
            
        case_dict[date][county] = [case, death]
        
    with open('data/preprocessed/state_cases.pkl','wb') as f:
        pickle.dump(case_dict, f)

#print(case_dict)
with open('data/preprocessed/state_cases.pkl','rb') as f:
    case_dict = pickle.load(f)
county_case = {c:[] for c in all_counties}
for key, value in case_dict.items():
    for county, cd in value.items():
        county_case[county].append(cd[0])
for key, value in county_case.items():
    county_case[key] = [np.mean(value), np.max(value)]
for date, value in case_dict.items():
    for county, cd in value.items():
        mean, maximum = county_case[county]
        case_dict[date][county] = (np.array(cd) - mean) / maximum


root = 'data/google trend/States/'
search_dict = {}
start = 26
for file in os.listdir(root):
    path = os.path.join(root, file)
    county = file.replace(' Google Trend.csv','')
    
    if county not in all_counties:
        continue
    #scale = [county][-1]
    data = pd.read_csv(path)
    searches = []
    dates = []

    today = datetime.date(2020,1,19)
    for index,i in enumerate(data.iloc):
        if index < start: #2020-01-19
            continue

        value = i['Category: All categories']
        
        if value == '<1':
            value = 0
        else:
            value = float(value)
            #print(value)

        searches.append(value)
    
    new_search = []
    for i in range(len(searches)-1):
        
        add = [(searches[i]+(k+1) * (searches[i+1]-searches[i])/7) for k in range(6)]

        new_search.append(searches[i])
        new_search+=add
    new_search.append(searches[-1])
    new_search = np.array(new_search)
    new_search = new_search / 100
    


    new_search = [(i-np.min(new_search)) / np.mean(new_search) for i in new_search]


    for i in range(len(new_search)):
        date = str(today)
        date = date.replace('-', '/')
        date = date[6:] + '/' + date[:4]
        dates.append(date)
        today = today + datetime.timedelta(days=1)
        
    for date, search in zip(dates, new_search):
        if date not in search_dict:
            search_dict[date] = {c:0 for c in all_counties}
        search_dict[date][county] = search
'''
Normalize the following:
flow_dict
case_dict
search_dict
'''

# for date, value in case_dict.items():

#     value_mean = np.mean([value[i][0] for i in value])
    
#     value_min = np.min([value[i][0] for i in value])
    
#     value_max = np.max([value[i][0] for i in value])

#     value = {i:value[i]/value_max for i in value}

#     case_dict[date] = value

# =============================================================================
# print(flow_dict)
# for date, value in flow_dict.items():
#     
#     value = {i: value[i] * county_dict[i][-1] for i in value}
# 
#     value_mean = np.mean([value[i] for i in value])
#     
#     value_min = np.min([value[i] for i in value])
#     
#     value_max = np.max([value[i] for i in value])
#     print(value_max)
#     print([value[i] for i in value])
#     print('\n' * 4)
#     value = {i:value[i]/value_max for i in value}
#     print([value[i] for i in value])
#     flow_dict[date] = value
#     
# =============================================================================
    

#print(flow_dict)
# for date, value in search_dict.items():
    
#     #value = {i: value[i] * county_dict[i][-1] for i in value}

#     value_mean = np.mean([value[i] for i in value])
    
#     value_min = np.min([value[i] for i in value])
    
#     value_max = np.max([value[i] for i in value])
    
#     #print(value_max)
#     #print([value[i] for i in value])

#     value = {i:value[i]/value_max for i in value}
    
#     #print(value)

#     search_dict[date] = value
    
#print(search_dict)
#all_counties = []
print(len(flow_dict))
print(len(case_dict))
print(len(search_dict))
num_prev = 1
wd_w = np.array([0.1226, 0.1260, 0.1242, 0.1337, 0.1822, 0.1744, 0.1366]) * 5
for selected_county in all_counties:
    print(selected_county)
    save = {'X':[], 'x':[], 'C':[], 'c':[], 'weekday':[], 'flow':[]}
    '''
            self.X = tf.placeholder(tf.float32, [batch,k-1], 'X')
            self.C = tf.placeholder(tf.float32, [batch,k-1], 'C')
            self.x = tf.placeholder(tf.float32, [batch,1], 'x')
            self.c = tf.placeholder(tf.float32, [batch,1], 'c')
            self.gt = tf.placeholder(tf.float32, [batch,1], 'gt')
            self.wd = tf.placeholder(tf.float32, [batch,7], 'wd')
    '''

    for index, (date, flow_value) in enumerate(flow_dict.items()):
        
        
        if date not in case_dict:
            continue
        if date not in search_dict:
            continue

        if index < num_prev - 1:
            continue

        flow = [flow_value[selected_county]]
        c, C, x, X = [], [], [], []
        for i in range(num_prev):
            
            
            if i != 0:
                date = str(date).replace('-', '/')
                date = date[6:] + '/' + date[:4] 
            if date not in case_dict: 
                c.append(c_temp)
                C.append(C_temp)
                x.append(x_temp)
                X.append(X_temp)
                continue
            covid_value = case_dict[date]
            search_value = search_dict[date]
            split_date = date.split('/')

            # if int(split_date[0]) > 5 or int(split_date[0]) < 3:
            #     continue
            date = datetime.date(int(split_date[2]),int(split_date[0]),int(split_date[1]))
            wd = date.weekday()
            wd_weight = wd_w[wd]
            weekday = [0.0] * 7
            weekday[wd] = 1.0
            

            
            c_temp = covid_value[selected_county][0]
            C_temp = np.array([covid_value[key][0] for key in covid_value if key != selected_county]) * wd_weight
            x_temp = search_value[selected_county] 
            X_temp = np.array([search_value[key] for key in search_value if key != selected_county]) * wd_weight
            

            c.append(c_temp)
            C.append(C_temp)
            x.append(x_temp)
            X.append(X_temp)

            date = date - datetime.timedelta(days=1)
        C = np.transpose(C)
        X = np.transpose(X)

        for key in save:
            save[key].append(eval(key))
            #print(np.shape(eval(key)))

    save_path = f'data/preprocessed/input/States/{selected_county}_prev.pkl'
    with open(save_path,'wb') as f:
        pickle.dump(save,f)


    
    #for i in range(1,,len(searches)-1):
        
    
    #print(searches)
# sum_wd = sum([wd_w[i] for i in wd_w])
# wd_list = []
# for i in wd_w:
#     wd_list.append(wd_w[i] / sum_wd)
# print(wd_list)

    
    