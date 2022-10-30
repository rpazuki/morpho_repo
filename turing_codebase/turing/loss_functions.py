from enum import Enum
import tensorflow as tf
from tensorflow import keras
import numpy as np
from . import PDE_Residual
from . import Loss


class Parameter_Type(Enum):
    CONSTANT = 1
    VARIABLE = 2
    INPUT = 3


def clip_by_value(z):
    return tf.clip_by_value(z, 1e-6, 1e10)


class PDE_Parameter:
    def __init__(self, name, parameter_type: Parameter_Type, value=1.0, index=-1, dtype=tf.float32):
        self.name = name
        self.parameter_type = parameter_type
        self.dtype = dtype
        self.value = value
        self.index = index
        self.trainable = ()

    def build(self):
        if self.parameter_type == Parameter_Type.CONSTANT:
            self.tf_var = tf.constant(self.value, dtype=self.dtype, name=self.name)
        elif self.parameter_type == Parameter_Type.VARIABLE:
            self.tf_var = tf.Variable([self.value], dtype=self.dtype, name=self.name, constraint=clip_by_value)
            self.trainable = (self.tf_var,)
        else:  # INPUT
            self.tf_var = None

        return self

    def get_value(self, input):
        if self.parameter_type == Parameter_Type.CONSTANT:
            return self.tf_var
        elif self.parameter_type == Parameter_Type.VARIABLE:
            return self.tf_var
        else:  # INPUT, self.index + 3 starts after (x, y, t)
            return input[:, self.index + 3]

    def set_value(self, value):
        if self.parameter_type == Parameter_Type.CONSTANT:
            self.value = value
            try:
                self.tf_var.assign(value)
            except AttributeError:
                self.tf_var = tf.constant(self.value, dtype=self.dtype, name=self.name)
        elif self.parameter_type == Parameter_Type.VARIABLE:
            self.value = value
            try:
                self.tf_var[0].assign(value)
            except AttributeError:
                self.tf_var = tf.Variable([self.value], dtype=self.dtype, name=self.name, constraint=clip_by_value)
        # else:  do nothing


class L2(Loss):
    def __init__(self):
        super().__init__(None)

    def norm(self, x, axis=None):
        return tf.reduce_mean(tf.square(x), axis=axis)


class L_Inf(Loss):
    def __init__(self):
        super().__init__(None)

    def norm(self, x, axis=None):
        return tf.reduce_max(tf.abs(x), axis=axis)


class Non_zero_params(PDE_Residual):
    def __init__(self, loss_name, parameters, epsilon=1e-7, alpha=1, print_precision=".5f", **kwargs):
        super().__init__(name=f"non_zero_{loss_name}", print_precision=print_precision, **kwargs)
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
    def residual(self, pinn, x):
        def f(x):
            # the small number epsilon*1e-3 prevents division by zero
            return self.epsilon / (x + self.epsilon * 1e-3) ** self.alpha

        params = tf.stack(self.parameters)
        return tf.reduce_sum(f(params))


class ASDM(PDE_Residual):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        sigma_u: PDE_Parameter,
        sigma_v: PDE_Parameter,
        mu_u: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        print_precision=".5f",
    ):
        """ASDM PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        """

        super().__init__(name="Loss_ASDM", print_precision=print_precision)

        self._trainables_ = ()

        self.D_u = D_u.build()
        self._trainables_ += D_u.trainable
        self.D_v = D_v.build()
        self._trainables_ += D_v.trainable
        self.sigma_u = sigma_u.build()
        self._trainables_ += sigma_u.trainable
        self.sigma_v = sigma_v.build()
        self._trainables_ += sigma_v.trainable
        self.mu_u = mu_u.build()
        self._trainables_ += mu_u.trainable
        self.rho_u = rho_u.build()
        self._trainables_ += rho_u.trainable
        self.rho_v = rho_v.build()
        self._trainables_ += rho_v.trainable
        self.kappa_u = kappa_u.build()
        self._trainables_ += kappa_u.trainable

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
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

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        sigma_u = self.sigma_u.get_value(x)
        sigma_v = self.sigma_v.get_value(x)
        mu_u = self.mu_u.get_value(x)
        rho_u = self.rho_u.get_value(x)
        rho_v = self.rho_v.get_value(x)
        kappa_u = self.kappa_u.get_value(x)

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = u_t - D_u * (u_xx + u_yy) - rho_u * f + mu_u * u - sigma_u
        f_v = v_t - D_v * (v_xx + v_yy) + rho_v * f - sigma_v

        return outputs, f_u, f_v


class Schnakenberg(PDE_Residual):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        c_0: PDE_Parameter,
        c_1: PDE_Parameter,
        c_2: PDE_Parameter,
        c_3: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(name="Loss_Schnakenberg", print_precision=print_precision)

        self._trainables_ = ()
        self.D_u = D_u.build()
        self._trainables_ += D_u.trainable
        self.D_v = D_v.build()
        self._trainables_ += D_v.trainable
        self.c_0 = c_0.build()
        self._trainables_ += c_0.trainable
        self.c_1 = c_1.build()
        self._trainables_ += c_1.trainable
        self.c_2 = c_2.build()
        self._trainables_ += c_2.trainable
        self.c_3 = c_3.build()
        self._trainables_ += c_3.trainable

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
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

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        c_0 = self.c_0.get_value(x)
        c_1 = self.c_1.get_value(x)
        c_2 = self.c_2.get_value(x)
        c_3 = self.c_3.get_value(x)

        u2v = u * u * v
        f_u = u_t - D_u * (u_xx + u_yy) - c_1 + c_0 * u - c_3 * u2v
        f_v = v_t - D_v * (v_xx + v_yy) - c_2 + c_3 * u2v

        return outputs, f_u, f_v


class FitzHugh_Nagumo(PDE_Residual):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        b: PDE_Parameter,
        gamma: PDE_Parameter,
        mu: PDE_Parameter,
        sigma: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(name="FitzHugh_Nagumo", print_precision=print_precision)

        self._trainables_ = ()

        self.D_u = D_u.build()
        self._trainables_ += D_u.trainable
        self.D_v = D_v.build()
        self._trainables_ += D_v.trainable
        self.b = b.build()
        self._trainables_ += b.trainable
        self.gamma = gamma.build()
        self._trainables_ += gamma.trainable
        self.mu = mu.build()
        self._trainables_ += mu.trainable
        self.sigma = sigma.build()
        self._trainables_ += sigma.trainable

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
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

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        b = self.b.get_value(x)
        gamma = self.gamma.get_value(x)
        mu = self.mu.get_value(x)
        sigma = self.sigma.get_value(x)

        f_u = u_t - D_u * (u_xx + u_yy) - mu * u + u * u * u + v - sigma
        f_v = v_t - D_v * (v_xx + v_yy) - b * u + gamma * v

        return outputs, f_u, f_v


#   The following are wrapper classes to turn the usual losses
#   to steady version (i.e. no time, or just one snapshot)
class Brusselator(PDE_Residual):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        A: PDE_Parameter,
        B: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(name="Brusselator", print_precision=print_precision)

        self._trainables_ = ()
        #
        self.D_u = D_u.build()
        self._trainables_ += D_u.trainable
        self.D_v = D_v.build()
        self._trainables_ += D_v.trainable
        self.A = A.build()
        self._trainables_ += A.trainable
        self.B = B.build()
        self._trainables_ += B.trainable

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
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

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        A = self.A.get_value(x)
        B = self.B.get_value(x)

        u2v = u * u * v

        f_u = u_t - D_u * (u_xx + u_yy) - A + (B + 1) * u - u2v
        f_v = v_t - D_v * (v_xx + v_yy) - B * u + u2v

        return outputs, f_u, f_v


class Circuit2_variant5716(PDE_Residual):
    def __init__(
        self,
        D_A: PDE_Parameter,
        D_B: PDE_Parameter,
        b_A: PDE_Parameter,
        b_B: PDE_Parameter,
        b_C: PDE_Parameter,
        b_D: PDE_Parameter,
        b_E: PDE_Parameter,
        b_F: PDE_Parameter,
        V_A: PDE_Parameter,
        V_B: PDE_Parameter,
        V_C: PDE_Parameter,
        V_D: PDE_Parameter,
        V_E: PDE_Parameter,
        V_F: PDE_Parameter,
        k_AA: PDE_Parameter,
        k_BD: PDE_Parameter,
        k_CE: PDE_Parameter,
        k_DA: PDE_Parameter,
        k_EB: PDE_Parameter,
        k_EE: PDE_Parameter,
        k_FE: PDE_Parameter,
        mu_A: PDE_Parameter,
        mulv_A: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(name="Circuit2_variant5716", print_precision=print_precision)

        self._trainables_ = ()

        self.add_trainable(D_A, "D_A")
        self.add_trainable(D_B, "D_B")
        self.add_trainable(b_A, "b_A")
        self.add_trainable(b_B, "b_B")
        self.add_trainable(b_C, "b_C")
        self.add_trainable(b_D, "b_D")
        self.add_trainable(b_E, "b_E")
        self.add_trainable(b_F, "b_F")
        self.add_trainable(V_A, "V_A")
        self.add_trainable(V_B, "V_B")
        self.add_trainable(V_C, "V_C")
        self.add_trainable(V_D, "V_D")
        self.add_trainable(V_E, "V_E")
        self.add_trainable(V_F, "V_F")
        self.add_trainable(k_AA, "k_AA")
        self.add_trainable(k_BD, "k_BD")
        self.add_trainable(k_CE, "k_CE")
        self.add_trainable(k_DA, "k_DA")
        self.add_trainable(k_EB, "k_EB")
        self.add_trainable(k_EE, "k_EE")
        self.add_trainable(k_FE, "k_FE")
        self.add_trainable(mu_A, "mu_A")
        self.add_trainable(mulv_A, "mulv_A")

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        p1, p2 = pinn.gradients(x, outputs)

        A = outputs[:, 0]
        B = outputs[:, 1]
        C = outputs[:, 2]
        D = outputs[:, 3]
        E = outputs[:, 4]
        F = outputs[:, 5]

        # p1 = [tf.gradients(outputs[:, i], x)[0] for i in range(outputs.shape[1])]

        # A_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # A_y = tf.cast(p1[0][:, 1], pinn.dtype)
        A_t = tf.cast(p1[0][:, 2], pinn.dtype)

        # A_xx = tf.cast(tf.gradients(A_x, x)[0][:, 0], pinn.dtype)
        # A_yy = tf.cast(tf.gradients(A_y, x)[0][:, 1], pinn.dtype)
        A_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        A_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # B_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # B_y = tf.cast(p1[1][:, 1], pinn.dtype)
        B_t = tf.cast(p1[1][:, 2], pinn.dtype)

        # B_xx = tf.cast(tf.gradients(B_x, x)[0][:, 0], pinn.dtype)
        # B_yy = tf.cast(tf.gradients(B_y, x)[0][:, 1], pinn.dtype)
        B_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        B_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        C_t = tf.cast(p1[2][:, 2], pinn.dtype)
        D_t = tf.cast(p1[3][:, 2], pinn.dtype)
        E_t = tf.cast(p1[4][:, 2], pinn.dtype)
        F_t = tf.cast(p1[5][:, 2], pinn.dtype)

        D_A = self.D_A.get_value(x)
        D_B = self.D_B.get_value(x)
        b_A = self.b_A.get_value(x)
        b_B = self.b_B.get_value(x)
        b_C = self.b_C.get_value(x)
        b_D = self.b_D.get_value(x)
        b_E = self.b_E.get_value(x)
        b_F = self.b_F.get_value(x)
        V_A = self.V_A.get_value(x)
        V_B = self.V_B.get_value(x)
        V_C = self.V_C.get_value(x)
        V_D = self.V_D.get_value(x)
        V_E = self.V_E.get_value(x)
        V_F = self.V_F.get_value(x)
        k_AA = self.k_AA.get_value(x)
        k_BD = self.k_BD.get_value(x)
        k_CE = self.k_CE.get_value(x)
        k_DA = self.k_DA.get_value(x)
        k_EB = self.k_EB.get_value(x)
        k_EE = self.k_EE.get_value(x)
        k_FE = self.k_FE.get_value(x)
        mu_A = self.mu_A.get_value(x)
        mulv_A = self.mulv_A.get_value(x)

        def noncompetitiveact(U, km, n=2):
            act = ((U / (km + 1e-20)) ** (n)) / (1 + (U / (km + 1e-20)) ** (n))
            return act

        def noncompetitiveinh(U, km, n=2):
            inh = 1 / (1 + (U / (km + 1e-20)) ** (n))
            return inh

        f_A = A_t - D_A * (A_xx + A_yy) - b_A - V_A * noncompetitiveinh(D, k_DA) + mu_A * A
        f_B = B_t - D_B * (B_xx + B_yy) - b_B - V_B * noncompetitiveact(A, k_AA) * noncompetitiveinh(E, k_EB) + mu_A * B
        f_C = C_t - b_C - V_C * noncompetitiveinh(D, k_DA) + mulv_A * C
        f_D = D_t - b_D - V_D * noncompetitiveact(B, k_BD) + mulv_A * D
        f_E = (
            E_t
            - b_E
            - V_E * noncompetitiveinh(C, k_CE) * noncompetitiveinh(F, k_FE) * noncompetitiveact(E, k_EE)
            + mulv_A * E
        )
        f_F = F_t - b_F - V_F * noncompetitiveact(B, k_BD) + mulv_A * F

        return outputs, f_A, f_B, f_C, f_D, f_E, f_F


class Circuit3954(PDE_Residual):
    def __init__(
        self,
        D: PDE_Parameter,
        b_A: PDE_Parameter,
        b_B: PDE_Parameter,
        b_C: PDE_Parameter,
        b_D: PDE_Parameter,
        b_E: PDE_Parameter,
        b_F: PDE_Parameter,
        mu_U: PDE_Parameter,
        mu_V: PDE_Parameter,
        mu_B: PDE_Parameter,
        mu_C: PDE_Parameter,
        mu_D: PDE_Parameter,
        mu_E: PDE_Parameter,
        mu_F: PDE_Parameter,
        mu_aTc: PDE_Parameter,
        K_AB: PDE_Parameter,
        K_BD: PDE_Parameter,
        K_CE: PDE_Parameter,
        K_DA: PDE_Parameter,
        K_EB: PDE_Parameter,
        K_EE: PDE_Parameter,
        K_FE: PDE_Parameter,
        K_aTc: PDE_Parameter,
        n: PDE_Parameter,
        n_aTc: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(name="Circuit2_variant5716", print_precision=print_precision)

        self._trainables_ = ()

        self.add_trainable(D, "D")
        self.add_trainable(b_A, "b_A")
        self.add_trainable(b_B, "b_B")
        self.add_trainable(b_C, "b_C")
        self.add_trainable(b_D, "b_D")
        self.add_trainable(b_E, "b_E")
        self.add_trainable(b_F, "b_F")
        self.add_trainable(mu_U, "mu_U")
        self.add_trainable(mu_V, "mu_V")
        self.add_trainable(mu_B, "mu_B")
        self.add_trainable(mu_C, "mu_C")
        self.add_trainable(mu_D, "mu_D")
        self.add_trainable(mu_E, "mu_E")
        self.add_trainable(mu_F, "mu_F")
        self.add_trainable(mu_aTc, "mu_aTc")
        self.add_trainable(K_AB, "K_AB")
        self.add_trainable(K_BD, "K_BD")
        self.add_trainable(K_CE, "K_CE")
        self.add_trainable(K_DA, "K_DA")
        self.add_trainable(K_EB, "K_EB")
        self.add_trainable(K_EE, "K_EE")
        self.add_trainable(K_FE, "K_FE")
        self.add_trainable(K_FE, "K_FE")
        self.add_trainable(K_aTc, "K_aTc")
        self.add_trainable(n_aTc, "n_aTc")
        self.add_trainable(n, "n")

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        p1, p2 = pinn.gradients(x, outputs)

        U = outputs[:, 0]
        V = outputs[:, 1]
        A = outputs[:, 2]
        B = outputs[:, 3]
        C = outputs[:, 4]
        D = outputs[:, 5]
        E = outputs[:, 6]
        F = outputs[:, 7]
        aTc = outputs[:, 8]

        U_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        U_yy = tf.cast(p2[0][:, 1], pinn.dtype)
        U_t = tf.cast(p1[0][:, 2], pinn.dtype)

        V_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        V_yy = tf.cast(p2[1][:, 1], pinn.dtype)
        V_t = tf.cast(p1[1][:, 2], pinn.dtype)

        A_t = tf.cast(p1[2][:, 2], pinn.dtype)
        B_t = tf.cast(p1[3][:, 2], pinn.dtype)
        C_t = tf.cast(p1[4][:, 2], pinn.dtype)
        D_t = tf.cast(p1[5][:, 2], pinn.dtype)
        E_t = tf.cast(p1[6][:, 2], pinn.dtype)
        F_t = tf.cast(p1[7][:, 2], pinn.dtype)
        aTc_t = tf.cast(p1[9][:, 2], pinn.dtype)

        D = self.D.get_value(x)
        b_A = self.b_A.get_value(x)
        b_B = self.b_B.get_value(x)
        b_C = self.b_C.get_value(x)
        b_D = self.b_D.get_value(x)
        b_E = self.b_E.get_value(x)
        b_F = self.b_F.get_value(x)
        mu_U = self.mu_U.get_value(x)
        mu_V = self.mu_V.get_value(x)
        mu_B = self.mu_B.get_value(x)
        mu_C = self.mu_C.get_value(x)
        mu_D = self.mu_D.get_value(x)
        mu_E = self.mu_E.get_value(x)
        mu_F = self.mu_F.get_value(x)
        mu_aTc = self.mu_aTc.get_value(x)
        K_AB = self.K_AB.get_value(x)
        K_BD = self.K_BD.get_value(x)
        K_CE = self.K_CE.get_value(x)
        K_DA = self.K_DA.get_value(x)
        K_EB = self.K_EB.get_value(x)
        K_EE = self.K_EE.get_value(x)
        K_FE = self.K_FE.get_value(x)
        K_aTc = self.K_aTc.get_value(x)
        n_aTc = self.n_aTc.get_value(x)
        n = self.n.get_value(x)

        def activate(Concentration, K, power=n):
            act = 1 / (1 + (K / (Concentration + 1e-20)) ** power)
            return act

        def inhibit(Concentration, K, power=n):
            inh = 1 / (1 + (Concentration / (K + 1e-20)) ** power)
            return inh

        K_CE_star = K_CE * inhibit(aTc, K_aTc, n_aTc)

        f_U = -U_t + A - mu_U * U + (U_xx + U_yy)
        f_V = -V_t + B - mu_V * V + D * (V_xx + V_yy)
        f_A = -A_t + b_A**2 + b_A * inhibit(D, K_DA) - A
        f_B = -B_t + mu_B * (b_B**2 + b_B * activate(U, K_AB) * inhibit(E, K_EB) - B)
        f_C = -C_t + mu_C * (b_C**2 + b_C * inhibit(D, K_DA) - C)
        f_D = -D_t + mu_D * (b_D**2 + b_D * activate(V, K_BD) - D)
        f_E = -E_t + mu_E * (b_E**2 + b_E * inhibit(C, K_CE_star) * inhibit(F, K_FE) * activate(E, K_EE) - E)
        f_F = -F_t + mu_F * (b_F**2 + b_F * activate(V, K_BD) - F)
        f_Atc = -aTc_t - mu_aTc * aTc

        return outputs, f_U, f_V, f_A, f_B, f_C, f_D, f_E, f_F, f_Atc


class ASDM_steady(ASDM):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        sigma_u: PDE_Parameter,
        sigma_v: PDE_Parameter,
        mu_u: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(D_u, D_v, sigma_u, sigma_v, mu_u, rho_u, rho_v, kappa_u, print_precision)

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        _, p2 = pinn.gradients(x, outputs)

        u = outputs[:, 0]
        v = outputs[:, 1]

        # a_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # a_y = tf.cast(p1[0][:, 1], pinn.dtype)
        # a_t = tf.cast(p1[0][:, 2], pinn.dtype)

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # s_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # s_y = tf.cast(p1[1][:, 1], pinn.dtype)
        # s_t = tf.cast(p1[1][:, 2], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        sigma_u = self.sigma_u.get_value(x)
        sigma_v = self.sigma_v.get_value(x)
        mu_u = self.mu_u.get_value(x)
        rho_u = self.rho_u.get_value(x)
        rho_v = self.rho_v.get_value(x)
        kappa_u = self.kappa_u.get_value(x)

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = -D_u * (u_xx + u_yy) - rho_u * f + mu_u * u - sigma_u
        f_v = -D_v * (v_xx + v_yy) + rho_v * f - sigma_v

        return outputs, f_u, f_v


class Schnakenberg_steady(Schnakenberg):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        c_0: PDE_Parameter,
        c_1: PDE_Parameter,
        c_2: PDE_Parameter,
        c_3: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(D_u, D_v, c_0, c_1, c_2, c_3, print_precision)

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        _, p2 = pinn.gradients(x, outputs)

        u = outputs[:, 0]
        v = outputs[:, 1]

        # u_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # u_y = tf.cast(p1[0][:, 1], pinn.dtype)
        # u_t = tf.cast(p1[0][:, 2], pinn.dtype)

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # v_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # v_y = tf.cast(p1[1][:, 1], pinn.dtype)
        # v_t = tf.cast(p1[1][:, 2], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        c_0 = self.c_0.get_value(x)
        c_1 = self.c_1.get_value(x)
        c_2 = self.c_2.get_value(x)
        c_3 = self.c_3.get_value(x)

        u2v = u * u * v
        f_u = -D_u * (u_xx + u_yy) - c_1 + c_0 * u - c_3 * u2v
        f_v = -D_v * (v_xx + v_yy) - c_2 + c_3 * u2v

        return outputs, f_u, f_v


class FitzHugh_Nagumo_steady(FitzHugh_Nagumo):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        b: PDE_Parameter,
        gamma: PDE_Parameter,
        mu: PDE_Parameter,
        sigma: PDE_Parameter,
        print_precision=".5f",
    ):
        super().__init__(D_u, D_v, b, gamma, mu, sigma, print_precision)

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        _, p2 = pinn.gradients(x, outputs)

        u = outputs[:, 0]
        v = outputs[:, 1]

        # u_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # u_y = tf.cast(p1[0][:, 1], pinn.dtype)
        # u_t = tf.cast(p1[0][:, 2], pinn.dtype)

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # v_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # v_y = tf.cast(p1[1][:, 1], pinn.dtype)
        # v_t = tf.cast(p1[1][:, 2], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        b = self.b.get_value(x)
        gamma = self.gamma.get_value(x)
        mu = self.mu.get_value(x)
        sigma = self.sigma.get_value(x)

        f_u = -D_u * (u_xx + u_yy) - mu * u + u * u * u + v - sigma
        f_v = -D_v * (v_xx + v_yy) - b * u + gamma * v

        return outputs, f_u, f_v
