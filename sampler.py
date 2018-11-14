import numpy as np

def generate_hidden_obs(time, Dz, Dx, z_0, f, g):
	"""
	generate hidden states and observation
	f: transition class with z_t = g.sample(z_t-1)
	g: emission class with x_t = g.sample(z_t)
	"""
	T = time
	Z = np.zeros((T, Dz))
	X = np.zeros((T, Dx))

	Z[0] = z_0
	X[0] = g.sample(z_0)
	for t in range(1,T):
		Z[t] = f.sample(Z[t-1])
		X[t] = g.sample(Z[t])
	return Z, X

def create_train_test_dataset(n_train, n_test, time, Dz, Dx, f, g, z_0 = None, lb = None, ub = None):
	hidden_train, obs_train = np.zeros((n_train, time, Dz)), np.zeros((n_train, time, Dx))
	hidden_test, obs_test = np.zeros((n_test, time, Dz)), np.zeros((n_test, time, Dx))

	if z_0 is None and (lb and ub) is None:
		assert False, 'must specify z_0 or (lb and ub)'

	for i in range(n_train+n_test):
		if z_0 is None:
			z_0 = np.random.uniform(low = lb, high = ub, size = Dz)
		hidden, obs = generate_hidden_obs(time, Dz, Dx, z_0, f, g)
		if i < n_train:
			hidden_train[i] = hidden
			obs_train[i] = obs
		else:
			hidden_test[i-n_train] = hidden
			obs_test[i-n_train] = obs

	return hidden_train, obs_train, hidden_test, obs_test