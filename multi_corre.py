import pickle
import os
import numpy as np


root = 'data/preprocessed/input'
flow_dict = {}
search_dict = {}
case_dict = {}
counties = []
for file in os.listdir(root):
	path = os.path.join(root, file)
	with open(path, 'rb') as f:
		data = pickle.load(f)
	county = file.split('_')[0]
	counties.append(county)
	flow_dict[county] = np.reshape(data['flow'],(-1))
	search_dict[county] = np.reshape(data['x'], (-1))
	case_dict[county] = np.reshape(data['c'], (-1))

corrs = {c:0 for c in counties}
for c in counties:
	flow = flow_dict[c]
	temp = 0
	temp_c = ''
	for county in counties:
		search = search_dict[county]
		corr = np.corrcoef(search, flow)[0][1]
		if corr < temp:
			temp = corr
			temp_c = county
		corrs[county] += corr
	print(c, temp_c, temp, '\n')
		#print(county, corr[0][1], '\n')
# for county_1 in counties:
# 	flow = flow_dict[county]
# 	for county_2 in counties: