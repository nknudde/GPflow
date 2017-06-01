from GPflow.rgp import RGP
from GPflow.ekernels import RBF
from GPflow.priors import Gamma
import numpy as np
import matplotlib.pyplot as plt

Lt = 2
t = np.array(np.linspace(0, 2.2 * np.pi, 100)[:, None])
Dt = t[1, 0] - t[0, 0]

kernels = []
for i in range(2):
    k = RBF(Lt, ARD=True, lengthscales=0.8)
    k.lengthscales.prior = Gamma(3., 1./3.)
    k.variance.prior = Gamma(3., 1./3.)
    kernels.append(k)
print('constructing')
m = RGP(kernels, np.sin(t), [70, 70], Lt, [1, 1])

print('optimizing')
m.optimize(disp=True, maxiter=50000)
print(m)
print('evaluating')

plt.plot(m.layers[0].X_mean.value)
plt.show()
plt.clf()
mu, var = m.predict_f(30)
print(var)
plt.plot(t, np.sin(t))
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, 60), (60, 1)), mu, 'b')
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, 60), (60, 1)), mu + 2 * np.sqrt(var), 'b--')
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, 60), (60, 1)), mu - 2 * np.sqrt(var), 'b--')
plt.show()
