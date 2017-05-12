from .model import GPModel, Model, Parameterized
from .param import Param, DataHolder, AutoFlow
from .mean_functions import Zero
from .transforms import positive
from .likelihoods import Gaussian
from .conditionals import conditional
from .tf_wraps import eye
from numpy.random import randn
import numpy as np
from ._settings import settings
import tensorflow as tf

float_type = settings.dtypes.float_types


class RGP(Model):
    def __init__(self, Y, kern_em, kern_tr, input_dim, output_dim, Z):
        Model.__init__(self)


class Layer(Parameterized):
    def __init__(self, kernel):
        super(Layer, self).__init__()
        self.kern = kernel
        self.likelihood = Gaussian()

    def hankel(self, v, L):
        """
        v is the vector you want to Hankel
        L is the length of the time window
        """
        N = tf.shape(v)[0]
        D = tf.shape(v)[1]
        idx = tf.expand_dims(tf.range(0, N - L + 1), 1) + tf.expand_dims(tf.range(0, L), 0)
        l = tf.one_hot(idx, N, axis=-1, dtype=float_type)
        r = tf.reshape(tf.matmul(tf.reshape(l, [L * (N - L + 1), N]), v), [(N - L + 1), L * D])
        return r


class HiddenLayer(Layer):
    def __init__(self, kernel, N, Q, Lt, M):
        super(HiddenLayer, self).__init__(kernel)
        self.Z = Param(randn(M, 2 * Lt * Q))
        self.X_mean = Param(randn(N + Lt, Q))
        self.X_var = Param(np.ones(N + Lt, Q), transform=positive)

    def build_likelihood(self, Lt, Xm_m, Xm_v):
        N = tf.shape(self.X_mean)[0]
        D = tf.shape(self.X_mean)[1]
        M = tf.shape(self.Z)[0]
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        X_m = tf.concat([self.hankel(tf.slice(self.X_mean, [0, 0], [N - 1, -1]), Lt),
                         self.hankel(tf.slice(Xm_m, [1, 0], [N - 1, -1]), Lt)], 1)
        X_v = tf.concat([self.hankel(tf.slice(self.X_var, [0, 0], [N - 1, -1]), Lt),
                         self.hankel(tf.slice(Xm_v, [1, 0], [N - 1, -1]), Lt)], 1)
        X_mo = tf.slice(self.X_mean, [Lt, 0], [-1, -1])
        X_vo = tf.slice(self.X_var, [Lt, 0], [-1, -1])
        X_mb = tf.slice(self.X_mean, [0, 0], [Lt, -1])
        X_vb = tf.slice(self.X_var, [0, 0], [Lt, -1])
        Kuu = self.kern.K(self.Z) + 1E-6 * eye(M)
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_m, X_v), 0)
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(M)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, X_mo), lower=True) / sigma
        ent = 0.5 * tf.reduce_sum(tf.log(X_vo)) + (N - Lt) * D * .5 * tf.log(2. * np.pi)
        ent2 = -Lt * D * tf.log(2 * np.pi) - 0.5 * tf.reduce_sum((tf.square(X_mb) + X_vb))

        bound = -.5 * (N - Lt) * D * tf.log(2 * np.pi * self.likelihood.variance)
        bound += -.5 / sigma2 * (tf.reduce_sum(X_vo) + tf.reduce_sum(tf.square(X_mo)))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.diag_part(AAT)))
        bound += -0.5 * D * log_det_B
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += ent
        bound += ent2
        return bound, self.X_mean, self.X_var


class InputLayer(Layer):
    def __init__(self, kernel, N, Q, Lt, M):
        super(InputLayer, self).__init__(kernel)
        self.Z = Param(randn(M, Lt * Q))
        self.X_mean = Param(randn(N + Lt, Q))
        self.X_var = Param(np.ones(N + Lt, Q), transform=positive)

    def build_likelihood(self, Lt):
        N = tf.shape(self.X_mean)[0]
        D = tf.shape(self.X_mean)[1]
        M = tf.shape(self.Z)[0]
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        X_m = self.hankel(tf.slice(self.X_mean, [0, 0], [N - 1, -1]), Lt)
        X_v = self.hankel(tf.slice(self.X_var, [0, 0], [N - 1, -1]), Lt)
        X_mo = tf.slice(self.X_mean, [Lt, 0], [-1, -1])
        X_vo = tf.slice(self.X_var, [Lt, 0], [-1, -1])
        X_mb = tf.slice(self.X_mean, [0, 0], [Lt, -1])
        X_vb = tf.slice(self.X_var, [0, 0], [Lt, -1])
        Kuu = self.kern.K(self.Z) + 1E-6 * eye(M)
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_m, X_v), 0)
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(M)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, X_mo), lower=True) / sigma
        ent = 0.5 * tf.reduce_sum(tf.log(X_vo)) + (N - Lt) * D * .5 * tf.log(2. * np.pi)
        ent2 = -Lt * D * tf.log(2 * np.pi) - 0.5 * tf.reduce_sum((tf.square(X_mb) + X_vb))

        bound = -.5 * (N - Lt) * D * tf.log(2 * np.pi * self.likelihood.variance)
        bound += -.5 / sigma2 * (tf.reduce_sum(X_vo) + tf.reduce_sum(tf.square(X_mo)))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.diag_part(AAT)))
        bound += -0.5 * D * log_det_B
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += ent
        bound += ent2
        return bound, self.X_mean, self.X_var


class OutputLayer(Layer):
    def __init__(self, kernel, N, Q, Lt, M, Y):
        super(OutputLayer, self).__init__(kernel, N, Q, Lt)
        self.Z = Param(randn(M, Lt * Q))
        self.Y = DataHolder(Y)

    def build_likelihood(self, Lt, Xm_m, Xm_v):
        N = tf.shape(self.X_mean)[0]
        D = tf.shape(self.X_mean)[1]
        M = tf.shape(self.Z)[0]
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        X_m = self.hankel(tf.slice(Xm_m, [1, 0], [N - 1, -1]), Lt)
        X_v = self.hankel(tf.slice(Xm_v, [1, 0], [N - 1, -1]), Lt)

        Kuu = self.kern.K(self.Z) + 1E-6 * eye(M)
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_m, X_v), 0)
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(M)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        bound = -.5 * (N - Lt) * D * tf.log(2 * np.pi * self.likelihood.variance)
        bound += -.5 / sigma2 * (tf.reduce_sum(tf.square(self.Y)))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.diag_part(AAT)))
        bound += -0.5 * D * log_det_B
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        return bound
