import numpy as np
import tensorflow as tf

from transformation.base import transformation


class linear_transformation(transformation):
    def transform(self, Z_prev):
        '''
        Integrates the Lorenz ODEs
        '''
        A = self.params
        return np.dot(A, Z_prev)


# test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    A = np.array([[0.99, 0.01], [0.01, 0.99]])
    Dz = 2
    T = 100
    batch_size = 10

    linear = linear_transformation(A)

    Z = np.zeros((T, Dz))
    Z[0] = np.random.uniform(low=0, high=1, size=Dz)
    for t in range(1, T):
        Z[t] = linear.transform(Z[t - 1])

    plt.figure()
    plt.plot(Z[:, 0], Z[:, 1])
    plt.show()
