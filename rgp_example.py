from GPflow.rgp import RGP
from GPflow.ekernels import RBF
from GPflow.priors import Gamma
import numpy as np
import matplotlib.pyplot as plt

n = 100

Lt = 2 # lag
H = 2 # layers
t = 5*np.array(np.linspace(0, 2.2 * np.pi, n)[:, None])
Dt = t[1, 0] - t[0, 0]

kernels = []
for i in range(H):
    k = RBF(Lt, ARD=True, lengthscales=0.8)
    k.lengthscales.prior = Gamma(3., 1./3.)
    k.variance.prior = Gamma(3., 1./3.)
    kernels.append(k)
print('constructing')
noise = 0 #np.random.normal(0,1e-3,size=(n,1))
m = RGP(kernels, np.sin(t)+noise, [70, 70], Lt, [1, 1])

print('optimizing')
m.optimize(disp=True, maxiter=50000)
print(m)
print('evaluating')

# todo: make it a parameter of predict_f
m.num_samples = 200

plt.plot(m.layers[0].X_mean.value)
plt.show()
plt.clf()
mu, var = m.predict_f()
print(var)
plt.plot(t, np.sin(t))
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, m.num_samples), (m.num_samples, 1)), mu, 'b')
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, m.num_samples), (m.num_samples, 1)), mu + 2 * np.sqrt(var), 'b--')
plt.plot(np.reshape(np.max(t) + Dt * np.arange(0, m.num_samples), (m.num_samples, 1)), mu - 2 * np.sqrt(var), 'b--')
plt.show()
