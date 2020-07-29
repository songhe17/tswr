import pickle
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
all_counties = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba']

with open('data/preprocessed/input/Fresno_prev.pkl','rb') as f:
	data = pickle.load(f)
	#print(data['x'])
# with open('results/Alameda_weights.pkl','rb') as f:
# 	data = pickle.load(f)
# 	print(data)


import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(data['X'])

x = data['x']
# x = np.reshape(x, (-1))
y = data['flow']
print(len(x))
'''
to prove there is a lag between amount of click and deduction of mobility
Under 7:
0 0.04385728112348498
[-0.19336408]

1 0.04120239139348476
[-0.20233767]

2 0.03742996764598766
[-0.21299374]

3 0.03397123499752404
[-0.2231714]

4 0.02973018003081361
[-0.23469219]

5 0.027128312185327467
[-0.24183219]

6 0.026781906998425323
[-0.24364233]

7 0.025757804639146654
[-0.24620355]

8 0.02507919140586344
[-0.24869603]

9 0.0247456632639142
[-0.25039273]

10 0.025790330995046854
[-0.24902684]

11 0.02684535583104381
[-0.24726781]

12 0.029022121226345076
[-0.24000856]

13 0.03308043522792044
[-0.23127749]

14 0.035798554593572604
[-0.22587767]

15 0.03920250511887235
[-0.21893886]

16 0.04273679022623469
[-0.21156781]

17 0.046277149061834964
[-0.20400066]

18 0.04959128559194082
[-0.19658527]

19 0.05367450082007029
[-0.18437659]
'''
print(len(x[:-1]), len(x))
losses = []
params = []
for i in range(20):
	if i != 0:
		sliced_x = deepcopy(x[:-i])
		sliced_y = deepcopy(y[i:])
	else:
		sliced_x = deepcopy(x)
		sliced_y = deepcopy(y)

	reg = LinearRegression().fit(sliced_x, sliced_y)
	predictions = reg.predict(sliced_x)
	loss = np.mean((sliced_y-predictions) ** 2)
	losses.append(loss)
	params.append(reg.coef_[0])
	# print(i,loss)
	# print(reg.coef_[0])
	# print()
	# plt.plot(predictions,marker='.')
	# plt.plot(sliced_y,marker='.')
	# plt.plot(sliced_x,marker='.')
	# plt.show()

plt.plot(losses)
#plt.plot(params)
plt.savefig('figures/losses_shift.png')
plt.show()



# with open('play_tab.csv', mode='w') as csv_file:
# 	keys = ['county', 'weights']
# 	writer = csv.DictWriter(csv_file, fieldnames=keys)
# 	writer.writeheader()
# 	for county, weight in zip(all_counties, reg.coef_[0]):
# 		writer.writerow({'county':county,'weights':weight})
	
	# for key in data:
	# 	print(key)
	# 	print(data[key])
# root = 'results'
# for file in os.listdir(root):
# 	path = os.path.join(root, file)
# 	with open(path, 'rb') as f:
# 		data = pickle.load(f)
# 	print(file)
# 	print(data)
# 	print('\n' * 4)
