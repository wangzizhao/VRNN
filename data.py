import csv
import numpy as np

def get_stock_data(stock_idx_name, time, n_train, n_test, Dx = 5):
	stock_file_name = "data/" + stock_idx_name + ".csv"
	with open(stock_file_name, 'r') as f:
		reader = csv.reader(f)
		data_list = list(reader)

	# remove the first row which is column names
	# remove the first column which is date
	# convert to float 32
	data = np.array([data_pt[1:1+Dx] for data_pt in data_list[1:]], dtype=np.float32)		

	obs_train = np.zeros((n_train, time, Dx))	# Open, High, Low, Close, Adj Close, Volume
	obs_test  = np.zeros((n_test,  time, Dx))

	# randomly pick some training and testing data
	for i in range(n_train + n_test):
		start = np.random.randint(low = 0, high = data.shape[0] - time)
		if i < n_train:
			obs_train[i] = data[start:start+time]
		else:
			obs_test[i-n_train] = data[start:start+time]

	return obs_train, obs_test
