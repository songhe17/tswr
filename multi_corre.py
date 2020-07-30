import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy


def find_best_shift(shift,x,y):

	temp = 1000
	res = 0
	min_loss = 0
	param = 0
	for i in range(shift):
		if i != 0:
			sliced_x = deepcopy(x[:-i])
			sliced_y = deepcopy(y[i:])
		else:
			sliced_x = deepcopy(x)
			sliced_y = deepcopy(y)

		reg = LinearRegression().fit(sliced_x, sliced_y)
		predictions = reg.predict(sliced_x)
		loss = np.mean((sliced_y-predictions) ** 2)

		if loss < temp:
			temp = loss
			res = i
			min_loss = loss
			param = reg.coef_[0][0]
	return res, min_loss, param
skips = ['Hawaii', 'Guam', 'Virgin Islands', 'Puerto Rico', 'Alaska']
#skips = ['Tuolumne', 'Napa', 'El Dorado']
root = 'data/preprocessed/input/States'
flow_dict = {}
search_dict = {}
case_dict = {}
counties = []
for file in os.listdir(root):
	path = os.path.join(root, file)
	with open(path, 'rb') as f:
		data = pickle.load(f)
	county = file.split('_')[0]
	if county in skips:
		continue
	counties.append(county)
	flow_dict[county] = data['flow']
	search_dict[county] = data['x']
	case_dict[county] = data['c']

corrs = {c:0 for c in counties}
for c in counties:
	flow = flow_dict[c]
	temp = 1000
	temp_c = ''
	for county in counties:
		search = search_dict[county]
		shift, min_loss, param = find_best_shift(30, search, flow)
		print(c,county,shift)
		# reg = LinearRegression().fit(sliced_search, sliced_flow)
		# predictions = reg.predict(sliced_search)
		# loss = np.mean((flow-predictions) ** 2)
		# param = reg.coef_[0]

		#corr = np.corrcoef(search, flow)[0][1]
		corrs[county]+=param
print(corrs)
	# print(c, temp_c, temp, '\n')
	# 	print(county, corr[0][1], '\n')
# for county_1 in counties:
# 	flow = flow_dict[county]
# 	for county_2 in counties: