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

root = 'data/google trend/California/'
search_dict = {}
start = 26
for file in os.listdir(root):
    path = os.path.join(root, file)
    county = file.split('.')[0]
    if county not in all_counties:
        print(county)
        continue
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
    searches = [(i-np.min(new_search)) / np.mean(new_search) for i in new_search]
    

    for i in range(len(new_search)):
        date = str(today)
        date = date.replace('-', '/')
        date = date[6:] + '/' + date[:4]
        dates.append(date)
        today = today + datetime.timedelta(days=1)
        
    for date, search in zip(dates, searches):

        if date not in search_dict:
            search_dict[date] = {c:0 for c in all_counties}
        search_dict[date][county] = search
        
selected_county = 'Los Angeles'
save = {'X':[], 'x':[], 'C':[], 'c':[], 'weekday':[], 'flow':[]}
'''
        self.X = tf.placeholder(tf.float32, [batch,k-1], 'X')
        self.C = tf.placeholder(tf.float32, [batch,k-1], 'C')
        self.x = tf.placeholder(tf.float32, [batch,1], 'x')
        self.c = tf.placeholder(tf.float32, [batch,1], 'c')
        self.gt = tf.placeholder(tf.float32, [batch,1], 'gt')
        self.wd = tf.placeholder(tf.float32, [batch,7], 'wd')
'''
metro = {'Bakersfield': ['Kern'], 'Chico-Redding':['Modoc', 'Trinity', 'Shasta', 'Tehama', 
         'Glenn', 'Butte'], 'Eureka': ['Humboldt'],
    'Fresno-Visalia': ['Kings', 'Tulare', 'Fresno', 'Madera', 'Mariposa', 'Merced'], 
    'Los Angeles': ['Ventura', 'Los Angeles', 'San Bernardino', 'Orange', 'Inyo'], 
    'Medford-Klamath Fall': ['Del Norte', 'Siskiyou'], 
    'Monterey-Salinas': ['Monterey', 'Santa Cruz', 'San Benito'], 
    'Palm Springs': ['Riverside'], 'Reno': ['Lassen', 'Mono', 'Alpine'], 
    'Sacramento-Stockton-Modesto': ['Colusa', 'Yolo', 'Sacra-Mento', 'Sutter', 'Yuba', 'Plumas', 'Sierra', 'Nevada', 'Placer', 'Amador','Calaveras', 'San Joaquin', 'Stanislaus', ], 
    'San Diego': ['San Diego'], 'San Francisco-Oakland-San Jose': ['Mendocina', 'Lake', 'Sondoma', 'Marin', 'San Francisco', 'San Mateo', 'Santa Clara', 'Alameda', 'Contra Costa', 'Soland'], 
    'SantaBArbara-Santa Maria- San Luis Obispo': ['San Luis Obispo', 'Santa Barbara'], 'Yuma AZ-EI Centro': ['Imperial']}

for date, flow_value in flow_dict.items():
    if date not in case_dict:
        continue
    if date not in search_dict:
        continue
    split_date = date.split('/')
    if int(split_date[0]) > 4 or int(split_date[0]) < 3:
        continue
    wd = datetime.date(int(split_date[2]),int(split_date[0]),int(split_date[1])).weekday()
    weekday = [[0.0] * 7]
    weekday[0][wd] = 1.0
    
    covid_value = case_dict[date]
    search_value = search_dict[date]
    
    x = [covid_value[selected_county][0]]
    X = [covid_value[key][0] for key in covid_value if key != selected_county]
    c = [search_value[selected_county]]
    C = [search_value[key] for key in search_value if key != selected_county]
    flow = [flow_value[selected_county]]
    
    for key in save:
        save[key].append(eval(key))
save_path = f'data/preprocessed/input_{selected_county}.pkl'
with open(save_path,'wb') as f:
    pickle.dump(save,f)
    
    
    
    #for i in range(1,,len(searches)-1):
        
    
    #print(searches)

    
    