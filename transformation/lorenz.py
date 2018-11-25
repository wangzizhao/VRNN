import numpy as np
from scipy.integrate import odeint

from transformation.base import transformation


class lorenz_transformation(transformation):
    def transform(self, Z_prev):
        '''
        Integrates the lorenz ODEs
        '''
        sigma, rho, beta, dt = self.params

        def lorenz_equation(Z, t, sigma, rho, beta):
            x, y, z = Z

            xd = sigma * (y - x)
            yd = (rho - z) * x - y
            zd = x * y - beta * z

            return [xd, yd, zd]

        t = np.arange(0, 2 * dt, dt)
        Z = odeint(lorenz_equation, Z_prev, t, args=(sigma, rho, beta))[1, :]

        return Z


# test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lorenz_params = (10.0, 28.0, 8.0 / 3.0, 0.01)
    Dz = 3
    T = 1500
    batch_size = 10

    lorenz = lorenz_transformation(lorenz_params)

    Z = np.zeros((T, Dz))
    Z[0] = np.random.uniform(low=0, high=1, size=Dz)
    for t in range(1, T):
        Z[t] = lorenz.transform(Z[t - 1])

    plt.figure()
    plt.plot(Z[:, 0], Z[:, 1])
    plt.show()
