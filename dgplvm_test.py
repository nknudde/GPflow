from GPflow.dgplvm import BayesianDGPLVM
import GPflow
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

N = 20
kt = GPflow.ekernels.RBF(1)
kx = GPflow.ekernels.RBF(1)

t = np.arange(N)
t = np.reshape(t, (N, 1)) * .2
X = np.random.multivariate_normal(np.zeros(N), kt.compute_K(t, t), 1).T
Y = np.random.multivariate_normal(np.zeros(N), kx.compute_K(X, X), 2).T

X_mean = GPflow.gplvm.PCA_reduce(Y, 1)
X_mean = 0.01*(X_mean-np.mean(X_mean, axis=0))/np.std(X_mean, axis=0)
X_var = np.ones((N, 1))
m = BayesianDGPLVM(X_mean=[X_mean], X_var=[X_var], Y=Y, kern=kx, time=[t],
                   kern_t=kt, M=6)

m.optimize(maxiter=50000, disp=True)

# prediction
N = 500
tn = np.reshape(np.linspace(0, np.max(t) * 3, N), (N, 1))
new, var = m.predict_serie(tn)

ax = plt.subplot(1, 1, 1)
ax.plot(tn, new[:, 0] + 2 * np.sqrt(var[:, 0]), 'b--')
ax.plot(tn, new[:, 0] - 2 * np.sqrt(var[:, 0]), 'b--')
ax.plot(tn, new[:, 0], 'b')
ax.scatter(t[:, 0], Y[:, 0], c='b')
ax.plot(tn, new[:, 1] + 2 * np.sqrt(var[:, 1]), 'g--')
ax.plot(tn, new[:, 1] - 2 * np.sqrt(var[:, 1]), 'g--')
ax.plot(tn, new[:, 1], 'g')
ax.scatter(t[:, 0], Y[:, 1], c='g')
ax.set_xlabel('t')
ax.set_ylabel('f(t)')
ax.set_ylim(-3, 3)
plt.show()
