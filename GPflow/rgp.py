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
    def __init__(self, kernels, Y, Ms, Lt, Qs):
        Model.__init__(self)
        layers = []
        assert len(kernels) == len(Ms)
        assert len(Ms) == len(Qs)
        assert len(kernels) >= 2
        N = np.shape(Y)[0]
        H = len(kernels)
        self.N = N
        self.Lt = Lt

        layers.append(InputLayer(kernels[0], N, Qs[0], Lt, Ms[0]))
        for i in range(1, H - 1):
            layers.append(HiddenLayer(kernels[i], N, Qs[i], Lt, Ms[i]))
        layers.append(OutputLayer(kernels[-1], N, Qs[-1], Lt, Ms[-1], Y))
        self.layers = layers

    def build_likelihood(self):
        b, Xm, Xv = self.layers[0].build_likelihood(self.Lt)
        for i in range(1, len(self.layers) - 1):
            b, Xm, Xv = self.layers[i].build_likelihood(self.Lt, Xm, Xv)
        return self.layers[-1].build_likelihood(self.Lt, Xm, Xv)

    @AutoFlow((tf.int32, []))
    def predict_f(self, num_samples):
        X_m = []
        X_v = []
        xm, xv, xmm, xvm = self.layers[0].predict_x(self.N, self.Lt, init=True)
        X_m.append(xm)
        X_v.append(xv)
        for i in range(1, len(self.layers) - 1):
            xm, xv, xmm, xvm = self.layers[i].predict_x(self.N, self.Lt, xm, xv, xmm, xvm, init=True)
            X_m.append(xm)
            X_v.append(xv)
        Ym, Yv = self.layers[-1].predict_x(self.N, self.Lt, xm, xv, xmm, xvm)
        for i in range(num_samples - 1):
            xm, xv, xmm, xvm = self.layers[0].predict_x(self.N, self.Lt, X_cms=X_m[0], X_cvs=X_v[0])
            X_m[0] = xm
            X_v[0] = xv
            for i in range(1, len(self.layers) - 1):
                xm, xv, xmm, xvm = self.layers[i].predict_x(self.N, self.Lt, xm, xv, xmm, xvm, X_cms=X_m[i],
                                                            X_cvs=X_v[i])
                X_m.append(xm)
                X_v.append(xv)
            ym, yv = self.layers[-1].predict_x(self.N, self.Lt, xm, xv, xmm, xvm)
            Ym = tf.concat([Ym, ym], 0)
            Yv = tf.concat([Yv, yv], 0)
        return Ym, Yv


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

    def predict_x(self, N, Lt, X_pms, X_pvs, Xp_m, Xp_v, init=False, X_cms=None, X_cvs=None):
        if init:
            X_cms = tf.slice(self.X_mean, [N - Lt, 0], [-1, -1])
            X_cvs = tf.slice(self.X_var, [N - Lt, 0], [-1, -1])
        X_m = tf.concat([self.hankel(tf.slice(self.X_mean, [0, 0], [N - 1, -1]), Lt),
                         self.hankel(tf.slice(Xp_m, [1, 0], [N - 1, -1]), Lt)], 1)
        X_v = tf.concat([self.hankel(tf.slice(self.X_var, [0, 0], [N - 1, -1]), Lt),
                         self.hankel(tf.slice(Xp_v, [1, 0], [N - 1, -1]), Lt)], 1)

        Xrms = tf.concat([X_cms, X_pms], 1)
        Xrvs = tf.concat([X_cvs, X_pvs], 1)
        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        psi0star = tf.reduce_sum(self.kern.eKdiag(Xrms, Xrvs), 0)  # N*
        psi1star = self.kern.eKxz(self.Z, Xrms, Xrvs)  # N* x M
        psi2star = tf.reduce_sum(self.kern.eKzxKxz(self.Z, Xrms, Xrvs), 0)
        X_mo = tf.slice(self.X_m, [Lt, 0], [-1, -1])

        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6  # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)  # M x M

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma  # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)  # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)  # M x M
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, X_mo), lower=True) / sigma  # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)

        # All of these: M x M
        tmp4 = tf.matmul(tmp1, tmp1, transpose_b=True)
        tmp5 = tf.matrix_triangular_solve(L, tf.transpose(tf.matrix_triangular_solve(L, psi2star)))
        tmp6 = tf.matrix_triangular_solve(LB, tf.transpose(tf.matrix_triangular_solve(LB, tmp5)))

        TT = tf.reduce_sum(tf.trace(tmp5 - tmp6))  # 1
        diagonals = psi0star - TT  # 1
        covar1 = tf.diag_part(tf.matmul(c, tf.matmul(tmp6 - tmp4, c), transpose_a=True))  # p x p
        covar = tf.expand_dims(covar1 + diagonals, 1)
        X_cms = tf.slice(tf.concat([X_m, mean], 0), [1, 0], [-1, -1])
        X_cvs = tf.slice(tf.concat([X_m, covar], 0), [1, 0], [-1, -1])
        return X_cms, X_cvs, self.X_mean, self.X_var


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

    def predict_x(self, N, Lt, init=False, X_cms=None, X_cvs=None):
        if init:
            X_cms = tf.slice(self.X_mean, [N - Lt, 0], [-1, -1])
            X_cvs = tf.slice(self.X_var, [N - Lt, 0], [-1, -1])
        X_m = self.hankel(tf.slice(self.X_mean, [0, 0], [N - 1, -1]), Lt)
        X_v = self.hankel(tf.slice(self.X_var, [0, 0], [N - 1, -1]), Lt)

        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        psi0star = tf.reduce_sum(self.kern.eKdiag(X_cms, X_cvs), 0)  # N*
        psi1star = self.kern.eKxz(self.Z, X_cms, X_cvs)  # N* x M
        psi2star = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_cms, X_cvs), 0)
        X_mo = tf.slice(self.X_m, [Lt, 0], [-1, -1])

        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6  # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)  # M x M

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma  # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)  # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)  # M x M
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, X_mo), lower=True) / sigma  # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)

        # All of these: M x M
        tmp4 = tf.matmul(tmp1, tmp1, transpose_b=True)
        tmp5 = tf.matrix_triangular_solve(L, tf.transpose(tf.matrix_triangular_solve(L, psi2star)))
        tmp6 = tf.matrix_triangular_solve(LB, tf.transpose(tf.matrix_triangular_solve(LB, tmp5)))

        TT = tf.reduce_sum(tf.trace(tmp5 - tmp6))  # 1
        diagonals = psi0star - TT  # 1
        covar1 = tf.diag_part(tf.matmul(c, tf.matmul(tmp6 - tmp4, c), transpose_a=True))  # p x p
        covar = tf.expand_dims(covar1 + diagonals, 1)
        X_cms = tf.slice(tf.concat([X_m, mean], 0), [1, 0], [-1, -1])
        X_cvs = tf.slice(tf.concat([X_m, covar], 0), [1, 0], [-1, -1])
        return X_cms, X_cvs


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

    def predict_x(self, N, Lt, X_pms, X_pvs, Xp_m, Xp_v):
        X_m = self.hankel(tf.slice(Xp_m, [1, 0], [N - 1, -1]), Lt)
        X_v = self.hankel(tf.slice(Xp_v, [1, 0], [N - 1, -1]), Lt)

        Xrms = X_pms
        Xrvs = X_pvs
        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, X_m, X_v)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_m, X_v), 0)
        psi0star = tf.reduce_sum(self.kern.eKdiag(Xrms, Xrvs), 0)  # N*
        psi1star = self.kern.eKxz(self.Z, Xrms, Xrvs)  # N* x M
        psi2star = tf.reduce_sum(self.kern.eKzxKxz(self.Z, Xrms, Xrvs), 0)
        X_mo = tf.slice(self.X_m, [Lt, 0], [-1, -1])

        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * 1e-6  # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)  # M x M

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma  # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)  # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)  # M x M
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma  # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)

        # All of these: M x M
        tmp4 = tf.matmul(tmp1, tmp1, transpose_b=True)
        tmp5 = tf.matrix_triangular_solve(L, tf.transpose(tf.matrix_triangular_solve(L, psi2star)))
        tmp6 = tf.matrix_triangular_solve(LB, tf.transpose(tf.matrix_triangular_solve(LB, tmp5)))

        TT = tf.reduce_sum(tf.trace(tmp5 - tmp6))  # 1
        diagonals = psi0star - TT  # 1
        covar1 = tf.diag_part(tf.matmul(c, tf.matmul(tmp6 - tmp4, c), transpose_a=True))  # p x p
        covar = tf.expand_dims(covar1 + diagonals, 1)
        return mean, covar