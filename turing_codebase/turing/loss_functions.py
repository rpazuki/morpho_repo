import tensorflow as tf
from tensorflow import keras
import numpy as np
from .tf_utils import Parameter_Type
from .tf_utils import Loss_Grad_Type
from .tf_utils import PDE_Parameter
from .pinns import Loss
from .tf_utils import clip_by_value, clip_by_value_zero_lb
from .pinns import Norm


class L2(Norm):
    def __init__(self):
        super().__init__(None)

    def reduce_norm(self, tupled_x, axis=None):
        # All loss object return their tensor as
        # Rank one or more tuples
        return tf.stack([tf.reduce_mean(tf.square(item), axis=axis) for item in tupled_x], axis=0)


class L_Inf(Norm):
    def __init__(self):
        super().__init__(None)

    def reduce_norm(self, tupled_x, axis=None):
        # All loss object return their tensor as
        # Rank one or more tuple, so, we loop over them
        # to get the norms
        return tf.stack([tf.reduce_max(tf.abs(item), axis=axis) for item in tupled_x], axis=0)


class Observation_Loss(Loss):
    def __init__(
        self,
        layers,
        loss_grad_type=Loss_Grad_Type.PINN,
        residual_ret_names=None,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Observation_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=layers[-1],
            residual_ret_names=tuple(["obs " + chr(ord("u") + i) for i in range(layers[-1])])
            if residual_ret_names is None
            else residual_ret_names,
            print_precision=print_precision,
            **kwargs,
        )
        self.layers = layers
        self.input_dim = layers[0]

    @tf.function
    def residual(self, pinn, x):
        output = pinn.net(x[:, : self.input_dim])
        y = x[:, self.input_dim :]
        diff = output - y
        # return tuple([diff[:, i] / tf.math.reduce_std(output[:, i]) for i in range(self.residual_ret_num)])
        return tuple([diff[:, i] for i in range(self.residual_ret_num)])


class Scaled_Output_Loss(Loss):
    def __init__(
        self,
        dtype,
        layers,
        scales=None,
        loss_grad_type=Loss_Grad_Type.PINN,
        residual_ret_names=None,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Observation_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=layers[-1],
            residual_ret_names=tuple([" " + chr(ord("A") + i) for i in range(layers[-1])])
            if residual_ret_names is None
            else residual_ret_names,
            print_precision=print_precision,
            **kwargs,
        )
        self.layers = layers
        self.input_dim = layers[0]
        if scales is not None:
            assert len(scales) == self.residual_ret_num
            self.scales = [tf.constant(s, dtype=dtype) for s in scales]
        else:
            self.scales = [tf.constant(1.0, dtype=dtype) for _ in range(self.residual_ret_num)]

    @tf.function
    def residual(self, pinn, x):
        output = pinn.net(x[:, : self.input_dim])
        y = x[:, self.input_dim :]
        # diff = output - y
        # return tuple(
        #     [
        #         (output[:, i] - self.scales[i] * y[:, i]) / tf.math.reduce_std(output[:, i])
        #         for i in range(self.residual_ret_num)
        #     ]
        # )
        return tuple([(output[:, i] - self.scales[i] * y[:, i]) for i in range(self.residual_ret_num)])


class Derivatives_Loss(Loss):
    def __init__(
        self,
        dtype,
        Ds=[1.0, 1.0],
        loss_grad_type=Loss_Grad_Type.PINN,
        regularise=True,
        input_dim: int = 3,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Derivatives_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=6,
            residual_ret_names=("u_xx", "u_yy", "u_t", "v_xx", "v_yy", "v_t"),
            print_precision=print_precision,
            **kwargs,
        )
        self.input_dim = input_dim
        self.Ds = [tf.constant(Ds[i], dtype=dtype) for i in range(len(Ds))]

    @tf.function
    def residual(self, pinn, x):
        inputs = x[:, : self.input_dim]
        (_, _, u_t, u_xx, u_yy, _, v_t, v_xx, v_yy) = self.derivatives(pinn, inputs)
        u_xx_obs = x[:, self.input_dim : self.input_dim + 1]
        u_yy_obs = x[:, self.input_dim + 1 : self.input_dim + 2]
        u_t_obs = x[:, self.input_dim + 2 : self.input_dim + 3]
        v_xx_obs = x[:, self.input_dim + 3 : self.input_dim + 4]
        v_yy_obs = x[:, self.input_dim + 4 : self.input_dim + 5]
        v_t_obs = x[:, self.input_dim + 5 : self.input_dim + 6]

        return (
            (self.Ds[0] * u_xx - u_xx_obs),  # / tf.math.reduce_std(u_xx_obs),
            (self.Ds[0] * u_yy - u_yy_obs),  # / tf.math.reduce_std(u_yy_obs),
            (u_t - u_t_obs),  # /tf.math.reduce_std(u_t_obs),
            (self.Ds[1] * v_xx - v_xx_obs),  # / tf.math.reduce_std(v_xx_obs),
            (self.Ds[1] * v_yy - v_yy_obs),  # / tf.math.reduce_std(v_yy_obs),
            (v_t - v_t_obs),  # /tf.math.reduce_std(v_t_obs),
        )


class Observation_And_Derivatives_Loss(Loss):
    def __init__(
        self,
        dtype,
        Ds=[1.0, 1.0],
        loss_grad_type=Loss_Grad_Type.PINN,
        regularise=True,
        input_dim: int = 3,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Derivatives_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=8,
            residual_ret_names=("obs u", "obs v", "u_xx", "u_yy", "u_t", "v_xx", "v_yy", "v_t"),
            print_precision=print_precision,
            **kwargs,
        )
        self.input_dim = input_dim
        self.Ds = [tf.constant(Ds[i], dtype=dtype) for i in range(len(Ds))]

    @tf.function
    def residual(self, pinn, x):
        inputs = x[:, : self.input_dim]
        (_, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, inputs)
        u_obs = x[:, self.input_dim : self.input_dim + 1]
        v_obs = x[:, self.input_dim + 1 : self.input_dim + 2]
        u_xx_obs = x[:, self.input_dim + 2 : self.input_dim + 3]
        u_yy_obs = x[:, self.input_dim + 3 : self.input_dim + 4]
        u_t_obs = x[:, self.input_dim + 4 : self.input_dim + 5]
        v_xx_obs = x[:, self.input_dim + 5 : self.input_dim + 6]
        v_yy_obs = x[:, self.input_dim + 6 : self.input_dim + 7]
        v_t_obs = x[:, self.input_dim + 7 : self.input_dim + 8]

        return (
            u - u_obs,
            v - v_obs,
            self.Ds[0] * u_xx - u_xx_obs,
            self.Ds[0] * u_yy - u_yy_obs,
            u_t - u_t_obs,
            self.Ds[1] * v_xx - v_xx_obs,
            self.Ds[1] * v_yy - v_yy_obs,
            v_t - v_t_obs,
        )


class Periodic_Boundary_Condition(Loss):
    def __init__(
        self,
        loss_grad_type=Loss_Grad_Type.PINN,
        regularise=True,
        input_dim=3,
        print_precision=".5f",
        **kwargs,
    ):
        """ """

        super().__init__(
            name="Periodic_Boundary_Condition",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=1,
            residual_ret_names=("periodic boundary",),
            print_precision=print_precision,
            **kwargs,
        )
        self.input_dim = input_dim

    @tf.function
    def residual(self, pinn, x):
        y1 = pinn.net(x[:, : self.input_dim])
        y2 = pinn.net(x[:, self.input_dim :])
        return (y1 - y2,)


class Diffusion_point_Loss(Loss):
    def __init__(
        self,
        Ds,
        dtype,
        loss_grad_type=Loss_Grad_Type.PINN,
        regularise=True,
        input_dim: int = 3,
        print_precision=".5f",
        **kwargs,
    ):
        """ """

        super().__init__(
            name="Diffusion_Point_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("diff u", "diff v"),
            print_precision=print_precision,
            **kwargs,
        )
        self.Ds = [tf.constant(Ds[i], dtype=dtype) for i in range(len(Ds))]
        self.input_dim = input_dim

    @tf.function
    def residual(self, pinn, x):
        x_center = x[:, : self.input_dim]
        (_, _, u_xx, u_yy, _, v_xx, v_yy) = self.derivatives_steady(pinn, x_center)

        diff_u = self.Ds[0] * (u_xx + u_yy)
        diff_v = self.Ds[1] * (v_xx + v_yy)
        obs_diff = x[:, self.input_dim :]
        # scale the observation by the grid step size
        # Note that the PINN's diffusion is scaled on differentiation time
        # diff_u_v = tf.stack([diff_u, diff_v], axis=1) - obs_diff * self.dxdy
        # scale both by diffusion constants
        # diff_u_v = tf.stack([diff_u_v[:, i] * self.Ds[i] for i in range(diff_u_v.shape[1])], axis=1)
        # diff_u_v = tf.stack([diff_u, diff_v], axis=1) - obs_diff
        diff_u = diff_u - obs_diff[:, 0]
        diff_v = diff_v - obs_diff[:, 1]
        return (diff_u, diff_v)


class Diffusion_Loss(Loss):
    def __init__(
        self,
        ns,
        Ls,
        Ds,
        dtype,
        loss_grad_type=Loss_Grad_Type.PINN,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """ """

        super().__init__(
            name="Diffusion_Loss",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("diff u", "diff v"),
            print_precision=print_precision,
            **kwargs,
        )
        self.dxdy = tf.constant(np.prod([ns[i] / Ls[i] for i in range(len(ns))]), dtype=dtype)
        self.Ds = [tf.constant(Ds[i], dtype=dtype) for i in range(len(Ds))]

    @tf.function
    def residual(self, pinn, x):
        x_center = pinn.net(x[:, :3])
        x_left = pinn.net(tf.concat([x[:, 3:4], x[:, 1:3]], axis=1))
        x_right = pinn.net(tf.concat([x[:, 4:5], x[:, 1:3]], axis=1))
        y_up = pinn.net(tf.concat([x[:, 0:1], x[:, 5:6], x[:, 2:3]], axis=1))
        y_bottom = pinn.net(tf.concat([x[:, 0:1], x[:, 6:7], x[:, 2:3]], axis=1))

        pinn_diff = self.dxdy * (x_left + x_right + y_up + y_bottom - 4.0 * x_center)
        pinn_diff = tf.stack([pinn_diff[:, i] * self.Ds[i] for i in range(pinn_diff.shape[1])], axis=1)

        obs_diff = x[:, 7:]
        # diff_u_v = pinn_diff - obs_diff
        diff_u = pinn_diff[:, :1] - obs_diff[:, :1]
        diff_v = pinn_diff[:, 1:] - obs_diff[:, 1:]
        return (diff_u, diff_v)


class Non_zero_params(Loss):
    def __init__(
        self,
        loss_name,
        parameters,
        loss_grad_type=Loss_Grad_Type.PARAMETER,
        regularise=False,
        epsilon=1e-7,
        alpha=1,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name=f"non_zero_{loss_name}",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=len(parameters),
            print_precision=print_precision,
            **kwargs,
        )
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

        # params = tf.stack(self.parameters)
        return tuple([f(p) for p in self.parameters])


class Koch_Meinhard(Loss):
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
        alpha_u=1.0,
        alpha_v=1.0,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """Koch_Meinhard PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        alpha_u and alpha_v are scales that we use to normalise the u and v.
        """

        super().__init__(
            name="Loss_Koch_Meinhard",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

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

        self.alpha_u = alpha_u
        self.alpha_v = alpha_v

    @tf.function
    def residual(self, pinn, x):

        (y, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        sigma_u = self.sigma_u.get_value(x) / self.alpha_u
        sigma_v = self.sigma_v.get_value(x) / self.alpha_v
        mu_u = self.mu_u.get_value(x)
        rho_u = self.rho_u.get_value(x) * self.alpha_u * self.alpha_v
        rho_v = self.rho_v.get_value(x) * self.alpha_u * self.alpha_u
        kappa_u = self.kappa_u.get_value(x) * self.alpha_u * self.alpha_u

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = u_t - D_u * (u_xx + u_yy) - rho_u * f + mu_u * u - sigma_u
        f_v = v_t - D_v * (v_xx + v_yy) + rho_v * f - sigma_v

        # return (f_u / tf.math.reduce_std(f_u), f_v / tf.math.reduce_std(f_v))
        return (f_u, f_v)


class Koch_Meinhard_output_as_Der(Loss):
    def __init__(
        self,
        dtype,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        sigma_u: PDE_Parameter,
        sigma_v: PDE_Parameter,
        mu_u: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        outputs_scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """Koch_Meinhard PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        alpha_u and alpha_v are scales that we use to normalise the u and v.
        """

        super().__init__(
            name="Loss_Koch_Meinhard",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

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

        assert len(outputs_scales) == 8
        self.outputs_scales = [tf.constant(i, dtype=dtype) for i in outputs_scales]

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        u, v, u_xx, u_yy, u_t, v_xx, v_yy, v_t = (
            outputs[:, 0] * self.outputs_scales[0],
            outputs[:, 1] * self.outputs_scales[1],
            outputs[:, 2] * self.outputs_scales[2],
            outputs[:, 3] * self.outputs_scales[3],
            outputs[:, 4] * self.outputs_scales[4],
            outputs[:, 5] * self.outputs_scales[5],
            outputs[:, 6] * self.outputs_scales[6],
            outputs[:, 7] * self.outputs_scales[7],
        )

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

        return (f_u, f_v)


class Koch_Meinhard_Dimensionless_output_as_Der(Loss):
    def __init__(
        self,
        dtype,
        D: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        outputs_scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """Koch_Meinhard PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        alpha_u and alpha_v are scales that we use to normalise the u and v.
        """

        super().__init__(
            name="Loss_Koch_Meinhard",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

        self._trainables_ = ()

        self.D = D.build()
        self._trainables_ += D.trainable
        self.rho_u = rho_u.build()
        self._trainables_ += rho_u.trainable
        self.rho_v = rho_v.build()
        self._trainables_ += rho_v.trainable
        self.kappa_u = kappa_u.build()
        self._trainables_ += kappa_u.trainable

        assert len(outputs_scales) == 8
        self.outputs_scales = [tf.constant(i, dtype=dtype) for i in outputs_scales]

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        u, v, u_xx, u_yy, u_t, v_xx, v_yy, v_t = (
            outputs[:, 0] * self.outputs_scales[0],
            outputs[:, 1] * self.outputs_scales[1],
            outputs[:, 2] * self.outputs_scales[2],
            outputs[:, 3] * self.outputs_scales[3],
            outputs[:, 4] * self.outputs_scales[4],
            outputs[:, 5] * self.outputs_scales[5],
            outputs[:, 6] * self.outputs_scales[6],
            outputs[:, 7] * self.outputs_scales[7],
        )

        D = self.D.get_value(x)
        rho_u = self.rho_u.get_value(x)
        rho_v = self.rho_v.get_value(x)
        kappa_u = self.kappa_u.get_value(x)

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = u_t - (u_xx + u_yy) - rho_u * f + u - 1
        f_v = v_t - D * (v_xx + v_yy) + rho_v * f - 1

        return (f_u, f_v)


class Koch_Meinhard_Dimensionless_steady_output_as_Der(Loss):
    def __init__(
        self,
        dtype,
        D: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        outputs_scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """Koch_Meinhard PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        alpha_u and alpha_v are scales that we use to normalise the u and v.
        """

        super().__init__(
            name="Loss_Koch_Meinhard",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

        self._trainables_ = ()

        self.D = D.build()
        self._trainables_ += D.trainable
        self.rho_u = rho_u.build()
        self._trainables_ += rho_u.trainable
        self.rho_v = rho_v.build()
        self._trainables_ += rho_v.trainable
        self.kappa_u = kappa_u.build()
        self._trainables_ += kappa_u.trainable

        assert len(outputs_scales) == 6
        self.outputs_scales = [tf.constant(i, dtype=dtype) for i in outputs_scales]

    @tf.function
    def residual(self, pinn, x):
        outputs = pinn.net(x)
        u, v, u_xx, u_yy, v_xx, v_yy = (
            outputs[:, 0] * self.outputs_scales[0],
            outputs[:, 1] * self.outputs_scales[1],
            outputs[:, 2] * self.outputs_scales[2],
            outputs[:, 3] * self.outputs_scales[3],
            outputs[:, 4] * self.outputs_scales[4],
            outputs[:, 5] * self.outputs_scales[5],
        )

        D = self.D.get_value(x)
        rho_u = self.rho_u.get_value(x)
        rho_v = self.rho_v.get_value(x)
        kappa_u = self.kappa_u.get_value(x)

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = -(u_xx + u_yy) - rho_u * f + u - 1
        f_v = -D * (v_xx + v_yy) + rho_v * f - 1

        return (f_u, f_v)


# - $\frac{\partial u^*}{\partial t^*} =  (\partial_{x^* x^*} + \partial_{y^* y^*}) u^*
# + \rho^*_u \frac{(u^*)^2 v^*}{1 + \kappa_u (u^*)^2} - u^* + 1$
# - $\frac{\partial v^*}{\partial t^*} =  D (\partial_{x^* x^*} + \partial_{y^* y^*}) v^*
# - \rho^*_v \frac{(u^*)^2 v^*}{1 + \kappa_u^* (u^*)^2} + 1$

# - $  u = (\sigma_u/\mu_u) u^*$
# - $  v = (\sigma_v/\mu_u) v^*$
# - $  t = t^*/ \mu_u$
# - $  x = \sqrt{\frac{D_u}{\mu_u}} x^*$
# - $  y = \sqrt{\frac{D_u}{\mu_u}} y^*$
# - $  D = \frac{D_v}{D_u}$
# - $  \rho_u = (\frac{mu_u^3}{\sigma_u \sigma_v}) \rho^*_u$
# - $  \rho_v = (\frac{mu_u^3}{\sigma_u^2}) \rho^*_v$
# - $  \kappa_u = (\frac{mu_u^2}{\sigma_u^2}) \kappa_u^*$
#
class Koch_Meinhard_Dimensionless(Loss):
    def __init__(
        self,
        D: PDE_Parameter,
        rho_u: PDE_Parameter,
        rho_v: PDE_Parameter,
        kappa_u: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        """Koch_Meinhard PDE loss

        if the parameter is None, it becomes traiable with initial value set as init_vale,
        otherwise, it will be a constant
        alpha_u and alpha_v are scales that we use to normalise the u and v.
        """

        super().__init__(
            name="Loss_Koch_Meinhard",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

        self._trainables_ = ()

        self.D = D.build()
        self._trainables_ += D.trainable
        self.rho_u = rho_u.build()
        self._trainables_ += rho_u.trainable
        self.rho_v = rho_v.build()
        self._trainables_ += rho_v.trainable
        self.kappa_u = kappa_u.build()
        self._trainables_ += kappa_u.trainable

    @tf.function
    def residual(self, pinn, x):

        (_, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, x)

        D = self.D.get_value(x)
        rho_u = self.rho_u.get_value(x)
        rho_v = self.rho_v.get_value(x)
        kappa_u = self.kappa_u.get_value(x)

        f = u * u * v / (1.0 + kappa_u * u * u)
        f_u = u_t - (u_xx + u_yy) - rho_u * f + u - 1
        f_v = v_t - D * (v_xx + v_yy) + rho_v * f - 1

        return (f_u, f_v)


class Schnakenberg(Loss):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        c_0: PDE_Parameter,
        c_1: PDE_Parameter,
        c_2: PDE_Parameter,
        c_3: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Loss_Schnakenberg",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

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
        (y, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        c_0 = self.c_0.get_value(x)
        c_1 = self.c_1.get_value(x)
        c_2 = self.c_2.get_value(x)
        c_3 = self.c_3.get_value(x)

        u2v = u * u * v
        f_u = u_t - D_u * (u_xx + u_yy) - c_1 + c_0 * u - c_3 * u2v
        f_v = v_t - D_v * (v_xx + v_yy) - c_2 + c_3 * u2v

        return (f_u, f_v)


class FitzHugh_Nagumo(Loss):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        b: PDE_Parameter,
        gamma: PDE_Parameter,
        mu: PDE_Parameter,
        sigma: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="FitzHugh_Nagumo",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

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
        (y, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        b = self.b.get_value(x)
        gamma = self.gamma.get_value(x)
        mu = self.mu.get_value(x)
        sigma = self.sigma.get_value(x)

        f_u = u_t - D_u * (u_xx + u_yy) - mu * u + u * u * u + v - sigma
        f_v = v_t - D_v * (v_xx + v_yy) - b * u + gamma * v

        return (f_u, f_v)


#   The following are wrapper classes to turn the usual losses
#   to steady version (i.e. no time, or just one snapshot)
class Brusselator(Loss):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        A: PDE_Parameter,
        B: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Brusselator",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=2,
            residual_ret_names=("res u", "res v"),
            print_precision=print_precision,
            **kwargs,
        )

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
        (y, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy) = self.derivatives(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        A = self.A.get_value(x)
        B = self.B.get_value(x)

        u2v = u * u * v

        f_u = u_t - D_u * (u_xx + u_yy) - A + (B + 1) * u - u2v
        f_v = v_t - D_v * (v_xx + v_yy) - B * u + u2v

        return (f_u, f_v)


class Circuit2_variant5716(Loss):
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
        k_AB: PDE_Parameter,
        k_BD: PDE_Parameter,
        k_CE: PDE_Parameter,
        k_DA: PDE_Parameter,
        k_EB: PDE_Parameter,
        k_EE: PDE_Parameter,
        k_FE: PDE_Parameter,
        mu_ASV: PDE_Parameter,
        mu_lvA: PDE_Parameter,
        nab: PDE_Parameter,
        nbd: PDE_Parameter,
        nce: PDE_Parameter,
        nda: PDE_Parameter,
        nfe: PDE_Parameter,
        neb: PDE_Parameter,
        nee: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        masked=False,
        **kwargs,
    ):
        super().__init__(
            name="Circuit2_variant5716",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=6,
            residual_ret_names=("res A", "res B", "res C", "res D", "res E", "res F"),
            print_precision=print_precision,
            **kwargs,
        )

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
        self.add_trainable(k_AB, "k_AB")
        self.add_trainable(k_BD, "k_BD")
        self.add_trainable(k_CE, "k_CE")
        self.add_trainable(k_DA, "k_DA")
        self.add_trainable(k_EB, "k_EB")
        self.add_trainable(k_EE, "k_EE")
        self.add_trainable(k_FE, "k_FE")
        self.add_trainable(mu_ASV, "mu_ASV")
        self.add_trainable(mu_lvA, "mu_lvA")
        self.add_trainable(nab, "nab")
        self.add_trainable(nbd, "nbd")
        self.add_trainable(nce, "nce")
        self.add_trainable(nda, "nda")
        self.add_trainable(nfe, "nfe")
        self.add_trainable(neb, "neb")
        self.add_trainable(nee, "nee")
        self.masked = masked

    @tf.function
    def residual(self, pinn, x):
        if self.masked is None:
            # outputs = pinn.net(x)
            # p1, p2 = pinn.gradients(x, outputs)
            x_v = x
        else:
            x_v = x[:, 0:3]
            mask = x[:, 3]
            # outputs = pinn.net(x_v)
            # p1, p2 = pinn.gradients(x_v, outputs)
        (y, A, B, C, D, E, F, A_t, B_t, C_t, D_t, E_t, F_t, A_xx, B_xx, A_yy, B_yy) = self.derivatives_multi_nodes(
            pinn, x_v
        )

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
        k_AB = self.k_AB.get_value(x)
        k_BD = self.k_BD.get_value(x)
        k_CE = self.k_CE.get_value(x)
        k_DA = self.k_DA.get_value(x)
        k_EB = self.k_EB.get_value(x)
        k_EE = self.k_EE.get_value(x)
        k_FE = self.k_FE.get_value(x)
        mu_ASV = self.mu_ASV.get_value(x)
        mu_lvA = self.mu_lvA.get_value(x)
        nab = self.nab.get_value(x)
        nbd = self.nbd.get_value(x)
        nce = self.nce.get_value(x)
        nda = self.nda.get_value(x)
        nfe = self.nfe.get_value(x)
        neb = self.neb.get_value(x)
        nee = self.nee.get_value(x)

        def activate(x, km, n=2):
            # ex = (km / (x + 1e-20)) ** (n)
            # tf.math.is_inf(ex)
            act = 1 / (1 + (km / (x + 1e-20)) ** (n))
            return act

        def inhibit(x, km, n=2):
            inh = 1 / (1 + (x / (km + 1e-20)) ** (n))
            return inh

        f_A = A_t - b_A - V_A * inhibit(D, k_DA, nda) + mu_ASV * A - D_A * (A_xx + A_yy)
        f_B = B_t - b_B - V_B * activate(A, k_AB, nab) * inhibit(E, k_EB, neb) + mu_ASV * B - D_B * (B_xx + B_yy)
        f_C = C_t - b_C - V_C * inhibit(D, k_DA, nda) + mu_lvA * C
        f_D = D_t - b_D - V_D * activate(B, k_BD, nbd) + mu_lvA * D
        f_E = E_t - b_E - V_E * inhibit(C, k_CE, nce) * inhibit(F, k_FE, nfe) * activate(E, k_EE, nee) + mu_lvA * E
        f_F = F_t - b_F - V_F * activate(B, k_BD, nbd) + mu_lvA * F

        if self.masked:
            f_C *= mask
            f_D *= mask
            f_E *= mask
            f_F *= mask

        return (f_A, f_B, f_C, f_D, f_E, f_F)


class Circuit3954(Loss):
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
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            name="Circuit2_variant5716",
            loss_grad_type=loss_grad_type,
            regularise=regularise,
            residual_ret_num=9,
            residual_ret_names=("res U", "res V", "res A", "res B", "res C", "res D", "res E", "res F", "res aTc"),
            print_precision=print_precision,
            **kwargs,
        )

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
        (
            y,
            U,
            V,
            A,
            B,
            C,
            D,
            E,
            F,
            aTc,
            U_t,
            V_t,
            A_t,
            B_t,
            C_t,
            D_t,
            E_t,
            F_t,
            aTc_t,
            U_xx,
            V_xx,
            U_yy,
            V_yy,
        ) = self.derivatives_multi_nodes(pinn, x)

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

        return (f_U, f_V, f_A, f_B, f_C, f_D, f_E, f_F, f_Atc)


class Koch_Meinhard_steady(Koch_Meinhard):
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
        alpha_u=1.0,
        alpha_v=1.0,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            D_u,
            D_v,
            sigma_u,
            sigma_v,
            mu_u,
            rho_u,
            rho_v,
            kappa_u,
            alpha_u,
            alpha_v,
            loss_grad_type,
            regularise,
            print_precision,
            **kwargs,
        )

    @tf.function
    def residual(self, pinn, x):
        (_, u, u_xx, u_yy, v, v_xx, v_yy) = self.derivatives_steady(pinn, x)

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

        return (f_u, f_v)


class Schnakenberg_steady(Schnakenberg):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        c_0: PDE_Parameter,
        c_1: PDE_Parameter,
        c_2: PDE_Parameter,
        c_3: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            D_u,
            D_v,
            c_0,
            c_1,
            c_2,
            c_3,
            loss_grad_type,
            regularise,
            print_precision,
            **kwargs,
        )

    @tf.function
    def residual(self, pinn, x):
        (_, u, u_xx, u_yy, v, v_xx, v_yy) = self.derivatives_steady(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        c_0 = self.c_0.get_value(x)
        c_1 = self.c_1.get_value(x)
        c_2 = self.c_2.get_value(x)
        c_3 = self.c_3.get_value(x)

        u2v = u * u * v
        f_u = -D_u * (u_xx + u_yy) - c_1 + c_0 * u - c_3 * u2v
        f_v = -D_v * (v_xx + v_yy) - c_2 + c_3 * u2v

        return (f_u, f_v)


class FitzHugh_Nagumo_steady(FitzHugh_Nagumo):
    def __init__(
        self,
        D_u: PDE_Parameter,
        D_v: PDE_Parameter,
        b: PDE_Parameter,
        gamma: PDE_Parameter,
        mu: PDE_Parameter,
        sigma: PDE_Parameter,
        loss_grad_type=Loss_Grad_Type.BOTH,
        regularise=True,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(
            D_u,
            D_v,
            b,
            gamma,
            mu,
            sigma,
            loss_grad_type,
            regularise,
            print_precision,
            **kwargs,
        )

    @tf.function
    def residual(self, pinn, x):
        (_, u, u_xx, u_yy, v, v_xx, v_yy) = self.derivatives_steady(pinn, x)

        D_u = self.D_u.get_value(x)
        D_v = self.D_v.get_value(x)
        b = self.b.get_value(x)
        gamma = self.gamma.get_value(x)
        mu = self.mu.get_value(x)
        sigma = self.sigma.get_value(x)

        f_u = -D_u * (u_xx + u_yy) - mu * u + u * u * u + v - sigma
        f_v = -D_v * (v_xx + v_yy) - b * u + gamma * v

        return (f_u, f_v)
