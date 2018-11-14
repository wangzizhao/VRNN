import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.contrib.integrate import odeint as tf_odeint

from transformation.base import transformation

class fhn_transformation(transformation):
	def transform(self, Z_prev):
		"""
		Integrates the fhn ODEs

		Input:
		fhn_params = a, b, c, I, dt
		a: the shape of the cubic parabola
		b: describes the kinetics of the recovery variable w
		c: describes the kinetics of the recovery variable
		I: input current
		dt: timestep

		Output
		Z = [V, w]
			V - membrane voltage
			w - recovery variable that mimics activation of an outward current
		"""
		a, b, c, I, dt = self.params

		def fhn_equation(Z, t, a, b, c, I):
			V, w = Z
			dVdt = V-V**3/3 - w + I
			dwdt = a*(b*V - c*w)
			return [dVdt, dwdt]

		t = np.arange(0, 2*dt, dt)
		Z = odeint(fhn_equation, Z_prev, t, args = (a, b, c, I))[1, :]

		return Z

# test code
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
	Dz = 2
	T = 20
	batch_size = 10

	fhn = fhn_transformation(fhn_params)

	Z = np.zeros((T, Dz))
	Z[0] = np.random.uniform(low = 0, high = 1, size = Dz)
	for t in range(1,T):
		Z[t] = fhn.transform(Z[t-1])

	plt.figure()
	plt.plot(Z[:, 0], Z[:, 1])
	plt.show()
