import tensorflow as tf
from tensorflow import keras
import numpy as np
from . import Loss


class Non_zero_params(Loss):
    def __init__(self, loss_name, parameters, epsilon=1e-7, alpha=1, print_precision=".5f"):
        super().__init__(name=f"non_zero_{loss_name}", print_precision=print_precision)
        """ Create a loss object that keeps the parameters from their lower bound.

            It use the function f(x) = epsilon/x^alpha to control the loss and its
            derivative.

        Args:
            loss_name:        str. The name of the corresponding loss class
                              that the list of parameters belongs.
            parameters:       A list of tf.Variable that belongs to another
                              loss class.
            epsilon:          A flaot hyper-parameter that control the loss value and
                              its gradient. As a rule of thumb, a value of one
                              order of magnitude smaller than the lower bound of
                              the parameter(s) is a good choise. Defaults to 1e-7.
            alpha:            An integer power of algebraic decay. Defaults to 1.
            print_precision:  F-string prining format for outputs. Defaults to ".5f".
        """
        self.parameters = parameters
        self.epsilon = epsilon
        self.alpha = alpha

    @tf.function
    def loss(self, pinn, x):
        def f(x):
            # the small number epsilon*1e-3 prevents division by zero
            return self.epsilon / (x + self.epsilon * 1e-3) ** self.alpha

        params = tf.stack(self.parameters)
        return tf.reduce_sum(f(params))


class ASDM(Loss):
    def __init__(
        self,
        dtype,
        init_value=10.0,
        D_a=None,
        D_s=None,
        sigma_a=None,
        sigma_s=None,
        mu_a=None,
        rho_a=None,
        rho_s=None,
        kappa_a=None,
        print_precision=".5f",
    ):
        """ASDM PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        """

        super().__init__(name="Loss_ASDM", print_precision=print_precision)

        self._trainables_ = ()
        if D_a is None:
            self.D_a = tf.Variable(
                [init_value], dtype=dtype, name="D_a", constraint=lambda z: tf.clip_by_value(z, 1e-6, 1e10)
            )
            self._trainables_ += (self.D_a,)
        else:
            self.D_a = tf.constant(D_a, dtype=dtype, name="D_a")

        if D_s is None:
            self.D_s = tf.Variable(
                [init_value], dtype=dtype, name="D_s", constraint=lambda z: tf.clip_by_value(z, 1e-6, 1e10)
            )
            self._trainables_ += (self.D_s,)
        else:
            self.D_s = tf.constant(D_s, dtype=dtype, name="D_s")

        if sigma_a is None:
            self.sigma_a = tf.Variable(
                [init_value], dtype=dtype, name="sigma_a", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.sigma_a,)
        else:
            self.sigma_a = tf.constant(sigma_a, dtype=dtype, name="sigma_a")

        if sigma_s is None:
            self.sigma_s = tf.Variable(
                [init_value], dtype=dtype, name="sigma_s", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.sigma_s,)
        else:
            self.sigma_s = tf.constant(sigma_s, dtype=dtype, name="sigma_s")

        if mu_a is None:
            self.mu_a = tf.Variable(
                [init_value], dtype=dtype, name="mu_a", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.mu_a,)
        else:
            self.mu_a = tf.constant(mu_a, dtype=dtype, name="mu_a")

        if rho_a is None:
            self.rho_a = tf.Variable(
                [init_value], dtype=dtype, name="rho_a", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.rho_a,)
        else:
            self.rho_a = tf.constant(rho_a, dtype=dtype, name="rho_a")

        if rho_s is None:
            self.rho_s = tf.Variable(
                [init_value], dtype=dtype, name="rho_s", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.rho_s,)
        else:
            self.rho_s = tf.constant(rho_s, dtype=dtype, name="rho_s")

        if kappa_a is None:
            self.kappa_a = tf.Variable(
                [init_value], dtype=dtype, name="kappa_a", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.kappa_a,)
        else:
            self.kappa_a = tf.constant(kappa_a, dtype=dtype, name="kappa_a")

    @tf.function
    def loss(self, pinn, x):
        outputs = pinn(x)
        p1, p2 = pinn.gradients(x, outputs)

        a = outputs[:, 0]
        s = outputs[:, 1]

        # a_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # a_y = tf.cast(p1[0][:, 1], pinn.dtype)
        a_t = tf.cast(p1[0][:, 2], pinn.dtype)

        a_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        a_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # s_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # s_y = tf.cast(p1[1][:, 1], pinn.dtype)
        s_t = tf.cast(p1[1][:, 2], pinn.dtype)

        s_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        s_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_a = self.D_a
        D_s = self.D_s
        sigma_a = self.sigma_a
        sigma_s = self.sigma_s
        mu_a = self.mu_a
        rho_a = self.rho_a
        rho_s = self.rho_s
        kappa_a = self.kappa_a

        f = a * a * s / (1.0 + kappa_a * a * a)
        f_a = a_t - D_a * (a_xx + a_yy) - rho_a * f + mu_a * a - sigma_a
        f_s = s_t - D_s * (s_xx + s_yy) + rho_s * f - sigma_s

        return outputs, f_a, f_s


class Schnakenberg(Loss):
    def __init__(
        self, dtype, init_value=10.0, D_u=None, D_v=None, c_0=None, c_1=None, c_2=None, c_3=None, print_precision=".5f"
    ):
        super().__init__(name="Loss_Schnakenberg", print_precision=print_precision)

        self._trainables_ = ()
        if D_u is None:
            self.D_u = tf.Variable(
                [init_value], dtype=dtype, name="D_u", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.D_u,)
        else:
            self.D_u = tf.constant(D_u, dtype=dtype, name="D_u")

        if D_v is None:
            self.D_v = tf.Variable(
                [init_value], dtype=dtype, name="D_v", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.D_v,)
        else:
            self.D_v = tf.constant(D_v, dtype=dtype, name="D_v")

        if c_0 is None:
            self.c_0 = tf.Variable(
                [init_value], dtype=dtype, name="c_0", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_0,)
        else:
            self.c_0 = tf.constant(c_0, dtype=dtype, name="c_0")

        if c_1 is None:
            self.c_1 = tf.Variable(
                [init_value], dtype=dtype, name="c_1", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_1,)
        else:
            self.c_1 = tf.constant(c_1, dtype=dtype, name="c_1")

        if c_2 is None:
            self.c_2 = tf.Variable(
                [init_value], dtype=dtype, name="c_2", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_2,)
        else:
            self.c_2 = tf.constant(c_2, dtype=dtype, name="c_2")

        if c_3 is None:
            self.c_3 = tf.Variable(
                [init_value], dtype=dtype, name="c_3", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_3,)
        else:
            self.c_3 = tf.constant(c_3, dtype=dtype, name="c_3")

    @tf.function
    def loss(self, pinn, x):
        outputs = pinn(x)
        p1, p2 = pinn.gradients(x, outputs)

        u = outputs[:, 0]
        v = outputs[:, 1]

        # u_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # u_y = tf.cast(p1[0][:, 1], pinn.dtype)
        u_t = tf.cast(p1[0][:, 2], pinn.dtype)

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # v_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # v_y = tf.cast(p1[1][:, 1], pinn.dtype)
        v_t = tf.cast(p1[1][:, 2], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_u = self.D_u
        D_v = self.D_v
        c_0 = self.c_0
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3

        u2v = u * u * v
        f_u = u_t - D_u * (u_xx + u_yy) - c_1 + c_0 * u - c_3 * u2v
        f_v = v_t - D_v * (v_xx + v_yy) - c_2 + c_3 * u2v

        return outputs, f_u, f_v


class FitzHugh_Nagumo(Loss):
    def __init__(
        self,
        dtype,
        init_value=10.0,
        D_u=None,
        D_v=None,
        alpha=None,
        epsilon=None,
        mu=None,
        c_1=None,
        c_2=None,
        print_precision=".5f",
    ):
        super().__init__(name="FitzHugh_Nagumo", print_precision=print_precision)

        self._trainables_ = ()
        if D_u is None:
            self.D_u = tf.Variable(
                [init_value], dtype=dtype, name="D_u", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.D_u,)
        else:
            self.D_u = tf.constant(D_u, dtype=dtype, name="D_u")

        if D_v is None:
            self.D_v = tf.Variable(
                [init_value], dtype=dtype, name="D_v", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.D_v,)
        else:
            self.D_v = tf.constant(D_v, dtype=dtype, name="D_v")

        if alpha is None:
            self.alpha = tf.Variable(
                [init_value], dtype=dtype, name="alpha", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.alpha,)
        else:
            self.alpha = tf.constant(alpha, dtype=dtype, name="alpha")

        if epsilon is None:
            self.epsilon = tf.Variable(
                [init_value], dtype=dtype, name="epsilon", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.epsilon,)
        else:
            self.epsilon = tf.constant(epsilon, dtype=dtype, name="epsilon")

        if mu is None:
            self.mu = tf.Variable(
                [init_value], dtype=dtype, name="mu", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.mu,)
        else:
            self.mu = tf.constant(mu, dtype=dtype, name="mu")

        if c_1 is None:
            self.c_1 = tf.Variable(
                [init_value], dtype=dtype, name="c_1", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_1,)
        else:
            self.c_1 = tf.constant(c_1, dtype=dtype, name="c_1")

        if c_2 is None:
            self.c_2 = tf.Variable(
                [init_value], dtype=dtype, name="c_2", constraint=lambda z: tf.clip_by_value(z, 0, 1e10)
            )
            self._trainables_ += (self.c_2,)
        else:
            self.c_2 = tf.constant(c_2, dtype=dtype, name="c_2")

    @tf.function
    def loss(self, pinn, x):
        outputs = pinn(x)
        p1, p2 = pinn.gradients(x, outputs)

        u = outputs[:, 0]
        v = outputs[:, 1]

        # u_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # u_y = tf.cast(p1[0][:, 1], pinn.dtype)
        u_t = tf.cast(p1[0][:, 2], pinn.dtype)

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # v_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # v_y = tf.cast(p1[1][:, 1], pinn.dtype)
        v_t = tf.cast(p1[1][:, 2], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_u = self.D_u
        D_v = self.D_v
        alpha = self.alpha
        epsilon = self.epsilon
        mu = self.mu
        c_1 = self.c_1
        c_2 = self.c_2

        f_u = u_t - D_u * (u_xx + u_yy) + alpha * u - epsilon * v
        f_v = v_t - D_v * (v_xx + v_yy) + c_1 * u - mu * v - c_2 * v * v * v

        return outputs, f_u, f_v


#   The following are wrapper classes to turn the usual two node losses
#   compatible to multi nodes version


class ASDM_multi(ASDM):
    def __init__(
        self,
        dtype,
        init_value=10.0,
        D_a=None,
        D_s=None,
        sigma_a=None,
        sigma_s=None,
        mu_a=None,
        rho_a=None,
        rho_s=None,
        kappa_a=None,
        print_precision=".5f",
    ):
        super().__init__(dtype, init_value, D_a, D_s, sigma_a, sigma_s, mu_a, rho_a, rho_s, kappa_a, print_precision)

    @tf.function
    def loss(self, pinn, x):
        outputs, f_a, f_s = super().loss(pinn, x)
        return outputs, tf.concat([tf.expand_dims(f_a, axis=1), tf.expand_dims(f_s, axis=1)], axis=1)


class Schnakenberg_multi(Schnakenberg):
    def __init__(
        self, dtype, init_value=10.0, D_u=None, D_v=None, c_0=None, c_1=None, c_2=None, c_3=None, print_precision=".5f"
    ):
        super().__init__(dtype, init_value, D_u, D_v, c_0, c_1, c_2, c_3, print_precision)

    @tf.function
    def loss(self, pinn, x):
        outputs, f_u, f_v = super().loss(pinn, x)
        return outputs, tf.concat([tf.expand_dims(f_u, axis=1), tf.expand_dims(f_v, axis=1)], axis=1)


class FitzHugh_Nagumo_multi(FitzHugh_Nagumo):
    def __init__(
        self, dtype, init_value=10.0, D_u=None, D_v=None, alpha=None, epsilon=None, mu=None, print_precision=".5f"
    ):
        super().__init__(dtype, init_value, D_u, D_v, alpha, epsilon, mu, print_precision)

    @tf.function
    def loss(self, pinn, x):
        outputs, f_u, f_v = super().loss(pinn, x)
        return outputs, tf.concat([tf.expand_dims(f_u, axis=1), tf.expand_dims(f_v, axis=1)], axis=1)
