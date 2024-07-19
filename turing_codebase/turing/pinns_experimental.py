import time
import pickle
import pathlib

# from black import out
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .pinns import NN
from .pinns import Loss
from .pinns import TINN
from .pinns import Loss
from .pinns_multi_nodes import TINN_multi_nodes
from .utils import indices


class TINN_inverse(tf.Module):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_residual: Loss,
        loss: Loss,
        extra_loss=[],
        non_zero_loss=None,
        alpha=0.5,
        loss_penalty_power=2,
        obs_u_pre_reg=1.0,
        obs_v_pre_reg=1.0,
        pde_u_pre_reg=1.0,
        pde_v_pre_reg=1.0,
        pde_para_u_pre_reg=1.0,
        pde_para_v_pre_reg=1.0,
        print_precision=".5f",
    ):
        self.pinn = pinn
        self.pde_residual = pde_residual
        self.non_zero_loss = non_zero_loss
        self.pde_params_len = len(self.pde_residual.trainables())
        self.extra_loss = extra_loss
        self.extra_loss_len = len(extra_loss)
        self.loss = loss
        self.alpha = alpha
        self.loss_penalty_power = tf.Variable(loss_penalty_power, dtype=pinn.dtype, trainable=False)
        self.print_precision = print_precision
        #
        self.lambda_obs_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_obs_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_params_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_params_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)

        self.lambda_obs_pre_u = tf.Variable(obs_u_pre_reg, dtype=pinn.dtype, trainable=False)
        self.lambda_obs_pre_v = tf.Variable(obs_v_pre_reg, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_pre_u = tf.Variable(pde_u_pre_reg, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_pre_v = tf.Variable(pde_v_pre_reg, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_pre_params_u = tf.Variable(pde_para_u_pre_reg, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_pre_params_v = tf.Variable(pde_para_v_pre_reg, dtype=pinn.dtype, trainable=False)

        self.grad_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_params_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_params_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        self.loss_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        self.__reset_losses__()

    def __reset_losses__(self):
        self.loss_total = 0
        self.loss_reg_total = 0
        self.loss_obs_u = 0
        self.loss_obs_v = 0
        self.loss_pde_u = 0
        self.loss_pde_v = 0
        self.loss_extra = np.zeros(self.extra_loss_len)
        self.train_acc = 0
        self.loss_non_zero = 0
        self.loss_pde_params = 0

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, update_lambdas, first_step, last_step, optimizer, train_acc_metric):
        with tf.GradientTape(persistent=True) as tape:
            if x_pde is None:
                outputs, f_u, f_v = self.pde_residual.residual(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_u, f_v = self.pde_residual.residual(self.pinn, x_pde)
            loss_obs_u = self.loss.norm(
                y_obs[:, 0] - outputs[:, 0]
            )  # tf.reduce_mean(tf.square(y_obs[:, 0] - outputs[:, 0]))
            loss_obs_v = self.loss.norm(
                y_obs[:, 1] - outputs[:, 1]
            )  # tf.reduce_mean(tf.square(y_obs[:, 1] - outputs[:, 1]))
            loss_pde_u = self.loss.norm(f_u)  # tf.reduce_mean(tf.square(f_u))
            loss_pde_v = self.loss.norm(f_v)  # tf.reduce_mean(tf.square(f_v))
            trainables = self.pinn.trainable_variables  # + self.pde_loss.trainables()
            if self.extra_loss_len > 0:
                loss_extra_items = [extra_loss.residual(self.pinn, x_obs) for extra_loss in self.extra_loss]
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra = tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_extra = 0.0

            loss_reg = (
                self.lambda_obs_pre_u * self.lambda_obs_u * loss_obs_u
                + self.lambda_obs_pre_v * self.lambda_obs_v * loss_obs_v
                + self.lambda_pde_pre_u * self.lambda_pde_u * loss_pde_u
                + self.lambda_pde_pre_v * self.lambda_pde_v * loss_pde_v
                + loss_extra
            )
            loss_pde_params_reg = (
                self.lambda_pde_pre_params_u * self.lambda_pde_params_u * loss_pde_u
                + self.lambda_pde_pre_params_v * self.lambda_pde_params_v * loss_pde_v
            )
            if self.non_zero_loss is not None:
                non_zero_loss_value = self.non_zero_loss.residual(self.pinn, None)
            else:
                non_zero_loss_value = 0

        if update_lambdas:
            self.__update_lambdas__(
                first_step, last_step, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
            )
        # trainables are just pinns W and any extra from params from extra_loss
        # The pde coeffs are not included
        grads = tape.gradient(loss_reg, trainables)
        # loss_pde_params_reg is speratly regularised and updates the pde params
        grads_params = tape.gradient(loss_pde_params_reg, self.pde_residual.trainables())
        #  None-zero parameter requlirisation
        if self.non_zero_loss is not None:
            grads_non_zero = tape.gradient(non_zero_loss_value, self.non_zero_loss.parameters)

        optimizer.apply_gradients(zip(grads, trainables))
        optimizer.apply_gradients(zip(grads_params, self.pde_residual.trainables()))
        if self.non_zero_loss is not None:
            optimizer.apply_gradients(zip(grads_non_zero, self.non_zero_loss.parameters))

        # grads = tape.gradient(loss_value, trainables)
        # self.optimizer.apply_gradients(zip(grads, trainables))
        train_acc_metric.update_state(y_obs, outputs)
        return (
            loss_reg,
            loss_obs_u,
            loss_obs_v,
            loss_pde_u,
            loss_pde_v,
            non_zero_loss_value,
            loss_extra_items,
            loss_pde_params_reg,
        )

    def __update_lambdas__(
        self, first_step, last_step, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
    ):
        grad_obs_u = tape.gradient(loss_obs_u, trainables)
        grad_obs_v = tape.gradient(loss_obs_v, trainables)
        grad_pde_u = tape.gradient(loss_pde_u, trainables)
        grad_pde_v = tape.gradient(loss_pde_v, trainables)
        grad_pde_params_u = tape.gradient(loss_pde_u, self.pde_residual.trainables())
        grad_pde_params_v = tape.gradient(loss_pde_v, self.pde_residual.trainables())

        temp_1 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_obs_u])
        temp_2 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_obs_v])
        temp_3 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_pde_u])
        temp_4 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_pde_v])
        temp_5 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_pde_params_u])
        temp_6 = tf.reduce_mean([tf.reduce_mean(tf.square(item)) for item in grad_pde_params_v])

        if first_step:
            self.grad_norm_obs_u.assign(temp_1)
            self.grad_norm_obs_v.assign(temp_2)
            self.grad_norm_pde_u.assign(temp_3)
            self.grad_norm_pde_v.assign(temp_4)
            self.grad_norm_pde_params_u.assign(temp_5)
            self.grad_norm_pde_params_v.assign(temp_6)

            self.loss_norm_obs_u.assign(loss_obs_u)
            self.loss_norm_obs_v.assign(loss_obs_v)
            self.loss_norm_pde_u.assign(loss_pde_u)
            self.loss_norm_pde_v.assign(loss_pde_v)

        else:
            self.grad_norm_obs_u.assign(self.grad_norm_obs_u + temp_1)
            self.grad_norm_obs_v.assign(self.grad_norm_obs_v + temp_2)
            self.grad_norm_pde_u.assign(self.grad_norm_pde_u + temp_3)
            self.grad_norm_pde_v.assign(self.grad_norm_pde_v + temp_4)
            self.grad_norm_pde_params_u.assign(self.grad_norm_pde_params_u + temp_5)
            self.grad_norm_pde_params_v.assign(self.grad_norm_pde_params_v + temp_6)

            self.loss_norm_obs_u.assign(self.loss_norm_obs_u + loss_obs_u)
            self.loss_norm_obs_v.assign(self.loss_norm_obs_v + loss_obs_v)
            self.loss_norm_pde_u.assign(self.loss_norm_pde_u + loss_pde_u)
            self.loss_norm_pde_v.assign(self.loss_norm_pde_v + loss_pde_v)

        if last_step:
            # w_1 = self.loss_norm_obs_u**2 / tf.sqrt(self.grad_norm_obs_u)
            # w_2 = self.loss_norm_obs_v**2 / tf.sqrt(self.grad_norm_obs_v)
            # w_3 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_u)
            # w_4 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_v)
            # w_5 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_params_u)
            # w_6 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_params_v)
            w_1 = tf.pow(self.loss_norm_obs_u, self.loss_penalty_power) / tf.sqrt(self.grad_norm_obs_u)
            w_2 = tf.pow(self.loss_norm_obs_v, self.loss_penalty_power) / tf.sqrt(self.grad_norm_obs_v)
            w_3 = tf.pow(self.loss_norm_pde_u, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_u)
            w_4 = tf.pow(self.loss_norm_pde_v, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_v)
            w_5 = tf.pow(self.loss_norm_pde_u, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_params_u)
            w_6 = tf.pow(self.loss_norm_pde_v, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_params_v)

            w_total = w_1 + w_2 + w_3 + w_4 + w_5 + w_6
            self.lambda_obs_u.assign(self.alpha * self.lambda_obs_u + ((1 - self.alpha) * 6.0 * w_1) / w_total)
            self.lambda_obs_v.assign(self.alpha * self.lambda_obs_v + ((1 - self.alpha) * 6.0 * w_2) / w_total)
            self.lambda_pde_u.assign(self.alpha * self.lambda_pde_u + ((1 - self.alpha) * 6.0 * w_3) / w_total)
            self.lambda_pde_v.assign(self.alpha * self.lambda_pde_v + ((1 - self.alpha) * 6.0 * w_4) / w_total)
            self.lambda_pde_params_u.assign(
                self.alpha * self.lambda_pde_params_u + ((1 - self.alpha) * 6.0 * w_5) / w_total
            )
            self.lambda_pde_params_v.assign(
                self.alpha * self.lambda_pde_params_v + ((1 - self.alpha) * 6.0 * w_6) / w_total
            )

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        X_pde=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
    ):

        # Samplling arrays
        samples = self.__create_samples__(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1

        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                else:
                    p_batch_train = X_pde[p_batch_indices]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_non_zero_batch,
                    loss_extra_batch,
                    loss_pde_params_reg_batch,
                ) = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    p_batch_train,
                    regularise,
                    step == 0,
                    step == last_step,
                    optimizer,
                    train_acc_metric,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch
                self.loss_obs_u += loss_obs_u_batch * w_obs
                self.loss_obs_v += loss_obs_v_batch * w_obs
                self.loss_pde_u += loss_pde_u_batch * w_pde
                self.loss_pde_v += loss_pde_v_batch * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_non_zero += loss_non_zero_batch
                self.loss_pde_params += loss_pde_params_reg_batch * w_pde
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = train_acc_metric.result()
            self.loss_non_zero = self.loss_non_zero / (last_step + 1)
            # update the samples
            self.__store_samples__(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self.__print_metrics__()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            self.__reset_losses__()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples

    def __create_samples__(self, epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters):
        # Samplling arrays
        ret = {"training_obs_accuracy": np.zeros(epochs)}
        if sample_losses:
            ret = {
                **ret,
                **{
                    "loss_total": np.zeros(epochs),
                    "loss_regularisd_total": np.zeros(epochs),
                    "loss_pde_params": np.zeros(epochs),
                    "loss_obs_u": np.zeros(epochs),
                    "loss_obs_v": np.zeros(epochs),
                    "loss_pde_u": np.zeros(epochs),
                    "loss_pde_v": np.zeros(epochs),
                },
            }
            if self.extra_loss_len > 0:
                for i, loss in enumerate(self.extra_loss):
                    ret[f"loss_extra_{loss.name}"] = np.zeros(epochs)
        if sample_regularisations:
            ret = {
                **ret,
                **{
                    "lambda_obs_u": np.zeros(epochs),
                    "lambda_obs_v": np.zeros(epochs),
                    "lambda_pde_u": np.zeros(epochs),
                    "lambda_pde_v": np.zeros(epochs),
                    "lambda_pde_params_u": np.zeros(epochs),
                    "lambda_pde_params_v": np.zeros(epochs),
                },
            }
        if sample_gradients:
            ret = {
                **ret,
                **{
                    "grads_obs_u": np.zeros(epochs),
                    "grads_obs_v": np.zeros(epochs),
                    "grads_pde_u": np.zeros(epochs),
                    "grads_pde_v": np.zeros(epochs),
                    "grad_norm_pde_params_u": np.zeros(epochs),
                    "grad_norm_pde_params_v": np.zeros(epochs),
                },
            }
        if self.non_zero_loss is not None:
            ret = {
                **ret,
                **{"loss_non_zero": np.zeros(epochs)},
            }
        if sample_parameters:
            for param in self.pde_residual.trainables():
                ret[f"{param.name.split(':')[0]}"] = np.zeros(epochs)
        return ret

    def __store_samples__(
        self, samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
    ):
        samples["training_obs_accuracy"][epoch] = self.train_acc
        if sample_losses:
            samples["loss_total"][epoch] = self.loss_total
            samples["loss_regularisd_total"][epoch] = self.loss_reg_total
            samples["loss_pde_params"][epoch] = self.loss_pde_params
            samples["loss_obs_u"][epoch] = self.loss_obs_u
            samples["loss_obs_v"][epoch] = self.loss_obs_v
            samples["loss_pde_u"][epoch] = self.loss_pde_u
            samples["loss_pde_v"][epoch] = self.loss_pde_v
            if self.extra_loss_len > 0:
                for i, loss in enumerate(self.extra_loss):
                    samples[f"loss_extra_{loss.name}"][epoch] = self.loss_extra[i]

        if sample_regularisations:
            samples["lambda_obs_u"][epoch] = self.lambda_obs_u.numpy()
            samples["lambda_obs_v"][epoch] = self.lambda_obs_v.numpy()
            samples["lambda_pde_u"][epoch] = self.lambda_pde_u.numpy()
            samples["lambda_pde_v"][epoch] = self.lambda_pde_v.numpy()
            samples["lambda_pde_params_u"][epoch] = self.lambda_pde_params_u.numpy()
            samples["lambda_pde_params_v"][epoch] = self.lambda_pde_params_v.numpy()
        if sample_gradients:
            samples["grads_obs_u"][epoch] = np.sqrt(self.grad_norm_obs_u.numpy())
            samples["grads_obs_v"][epoch] = np.sqrt(self.grad_norm_obs_v.numpy())
            samples["grads_pde_u"][epoch] = np.sqrt(self.grad_norm_pde_u.numpy())
            samples["grads_pde_v"][epoch] = np.sqrt(self.grad_norm_pde_v.numpy())
            samples["grad_norm_pde_params_u"][epoch] = np.sqrt(self.grad_norm_pde_params_u.numpy())
            samples["grad_norm_pde_params_v"][epoch] = np.sqrt(self.grad_norm_pde_params_v.numpy())

        if self.non_zero_loss is not None:
            samples["loss_non_zero"][epoch] = self.loss_non_zero
        if sample_parameters:
            for param in self.pde_residual.trainables():
                samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()

    def __print_metrics__(self):
        print(f"Training observations acc over epoch: {self.train_acc:{self.print_precision}}")
        print(
            f"total loss: {self.loss_total:{self.print_precision}}, "
            f"total regularisd loss (sum of batches): {self.loss_reg_total:{self.print_precision}}"
        )
        print(
            f"obs u loss: {self.loss_obs_u:{self.print_precision}}, "
            f"obs v loss: {self.loss_obs_v:{self.print_precision}}"
        )
        print(
            f"pde u loss: {self.loss_pde_u:{self.print_precision}}, "
            f"pde v loss: {self.loss_pde_v:{self.print_precision}}"
        )
        print(f"pde params loss: {self.loss_pde_params:{self.print_precision}}")
        if self.non_zero_loss is not None:
            print(f"Non-zero loss: {self.loss_non_zero:{self.print_precision}}, ")
        print(
            f"lambda obs u: {self.lambda_obs_u.numpy():{self.print_precision}}, "
            f"lambda obs v: {self.lambda_obs_v.numpy():{self.print_precision}}"
        )
        print(
            f"lambda pde u: {self.lambda_pde_u.numpy():{self.print_precision}}, "
            f"lambda pde v: {self.lambda_pde_v.numpy():{self.print_precision}}"
        )
        print(
            f"lambda pde params u: {self.lambda_pde_params_u.numpy():{self.print_precision}}, "
            f"lambda pde params v: {self.lambda_pde_params_v.numpy():{self.print_precision}}"
        )
        print(self.pde_residual.trainables_str())
        if self.extra_loss_len > 0:
            for i, loss in enumerate(self.extra_loss):
                print(f"extra loss {loss.name}: {self.loss_extra[i]:{self.print_precision}}")

    def save(self, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # tf.saved_model.save(self, str(path))
        #
        # import os
        # if not pathlib.Path(path.joinpath(name)).exists():
        #   os.makedirs(path.joinpath(name))
        with open(f"{str(path)}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def restore(cls, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}.pkl", "rb") as f:
            model = pickle.load(f)
        return model


class TINN_masked(TINN):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_loss: Loss,
        extra_loss=[],
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
        alpha=0.5,
        print_precision=".5f",
    ):
        super().__init__(pinn, pde_loss, extra_loss, optimizer, train_acc_metric, alpha, print_precision)

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, mask, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=True) as tape:
            if x_pde is None:
                outputs, f_u, f_v = self.pde_residual.residual(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_u, f_v = self.pde_residual.residual(self.pinn, x_pde)
            loss_obs_u = tf.reduce_mean(tf.square(y_obs[:, 0] - outputs[:, 0]))
            loss_obs_v = tf.reduce_mean(tf.square(y_obs[:, 1] - outputs[:, 1]))
            # Mask the PDE residuals
            pde_mask = tf.expand_dims(mask, 1)
            loss_pde_u = tf.reduce_mean(tf.square(f_u * pde_mask))
            loss_pde_v = tf.reduce_mean(tf.square(f_v * pde_mask))
            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            if self.extra_loss_len > 0:
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra = tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_extra = 0.0

            loss_value = (
                self.lambda_obs_u * loss_obs_u
                + self.lambda_obs_v * loss_obs_v
                + self.lambda_pde_u * loss_pde_u
                + self.lambda_pde_v * loss_pde_v
                + loss_extra
            )

        if update_lambdas:
            self._update_lambdas_(
                x_pde, first_step, last_step, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
            )

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y_obs, outputs)
        return loss_value, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, loss_extra_items

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        domain_mask,
        X_pde=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        regularise_int=10,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                domain_mask_train = domain_mask[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                    domain_mask_train = domain_mask[o_batch_indices]
                else:
                    p_batch_train = X_pde[p_batch_indices]
                    domain_mask_train = domain_mask[p_batch_indices]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_extra_batch,
                ) = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    p_batch_train,
                    domain_mask_train,
                    regularise and epoch % regularise_int == 0,
                    step == 0,
                    step == last_step,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch
                self.loss_obs_u += loss_obs_u_batch * w_obs
                self.loss_obs_v += loss_obs_v_batch * w_obs
                self.loss_pde_u += loss_pde_u_batch * w_pde
                self.loss_pde_v += loss_pde_v_batch * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = self.train_acc_metric.result()
            # update the samples
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples


class TINN_multi_nodes_masked(TINN_multi_nodes):
    def __init__(
        self,
        pinn: NN,
        pde_loss: Loss,
        extra_loss=[],
        nodes_n=2,
        node_names=None,
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
        alpha=0.5,
        print_precision=".5f",
    ):
        super().__init__(
            pinn, pde_loss, extra_loss, nodes_n, node_names, optimizer, train_acc_metric, alpha, print_precision
        )

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, mask, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=False) as tape:
            if x_pde is None:
                outputs, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_pde)

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs, outputs), axis=0)
            # Mask the PDE residuals
            pde_mask = tf.expand_dims(mask, 0)
            loss_pde = tf.reduce_mean(tf.square(f_pde * pde_mask), axis=1)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            if self.extra_loss_len > 0:
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items) + tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items)

        if update_lambdas:
            self._update_lambdas_(x_pde, first_step, last_step, loss_obs, loss_pde, loss_items, trainables)

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y_obs, outputs)
        return loss_value, loss_obs, loss_pde, loss_extra_items

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        domain_mask,
        X_pde=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        regularise_int=10,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                    domain_mask_train = domain_mask[o_batch_indices]
                else:
                    p_batch_train = X_pde[p_batch_indices]
                    domain_mask_train = domain_mask[p_batch_indices]

                loss_value_batch, loss_obs_batch, loss_pde_batch, loss_extra_batch = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    p_batch_train,
                    domain_mask_train,
                    regularise and epoch % regularise_int == 0,
                    step == 0,
                    step == last_step,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch.numpy()
                self.loss_obs += loss_obs_batch.numpy() * w_obs
                self.loss_pde += loss_pde_batch.numpy() * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    np.sum(loss_obs_batch.numpy() * w_obs)
                    + np.sum(loss_pde_batch.numpy() * w_pde)
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = self.train_acc_metric.result()
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()
            # end for epoch in range(epochs)

        return samples


class TINN_masked2(TINN):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_loss: Loss,
        extra_loss=[],
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
        alpha=0.5,
        print_precision=".5f",
    ):
        super().__init__(pinn, pde_loss, extra_loss, optimizer, train_acc_metric, alpha, print_precision)

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_mask, x_pde, pde_mask, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=True) as tape:
            x_mask_expand = tf.expand_dims(x_mask, 1)
            x_obs_masked = tf.multiply(x_obs, x_mask_expand)
            if x_pde is None:
                outputs, f_u, f_v = self.pde_residual.residual(self.pinn, x_obs_masked)
            else:
                outputs = self.pinn(x_obs_masked)
                x_pde_masked = tf.multiply(x_pde, tf.expand_dims(pde_mask, 1))
                _, f_u, f_v = self.pde_residual.residual(self.pinn, x_pde_masked)
            loss_obs_u = tf.reduce_mean(tf.square(y_obs[:, 0] * x_mask - outputs[:, 0]))
            loss_obs_v = tf.reduce_mean(tf.square(y_obs[:, 1] * x_mask - outputs[:, 1]))
            loss_pde_u = tf.reduce_mean(tf.square(f_u))
            loss_pde_v = tf.reduce_mean(tf.square(f_v))
            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            if self.extra_loss_len > 0:
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra = tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_extra = 0.0

            loss_value = (
                self.lambda_obs_u * loss_obs_u
                + self.lambda_obs_v * loss_obs_v
                + self.lambda_pde_u * loss_pde_u
                + self.lambda_pde_v * loss_pde_v
                + loss_extra
            )

        if update_lambdas:
            self._update_lambdas_(
                x_pde, first_step, last_step, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
            )

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y_obs * x_mask_expand, outputs)
        return loss_value, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, loss_extra_items

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        domain_mask,
        X_pde=None,
        pde_mask=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                domain_mask_train = domain_mask[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                    p_domain_mask = None
                else:
                    p_batch_train = X_pde[p_batch_indices]
                    p_domain_mask = pde_mask[p_batch_indices]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_extra_batch,
                ) = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    domain_mask_train,
                    p_batch_train,
                    p_domain_mask,
                    regularise,
                    step == 0,
                    step == last_step,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch
                self.loss_obs_u += loss_obs_u_batch * w_obs
                self.loss_obs_v += loss_obs_v_batch * w_obs
                self.loss_pde_u += loss_pde_u_batch * w_pde
                self.loss_pde_v += loss_pde_v_batch * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = self.train_acc_metric.result()
            # update the samples
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples


class TINN_multi_nodes_masked2(TINN_multi_nodes):
    def __init__(
        self,
        pinn: NN,
        pde_loss: Loss,
        extra_loss=[],
        nodes_n=2,
        node_names=None,
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
        alpha=0.5,
        print_precision=".5f",
    ):
        super().__init__(
            pinn, pde_loss, extra_loss, nodes_n, node_names, optimizer, train_acc_metric, alpha, print_precision
        )

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_mask, x_pde, pde_mask, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=False) as tape:
            x_mask_expand = tf.expand_dims(x_mask, 1)
            x_obs_masked = tf.multiply(x_obs, x_mask_expand)
            if x_pde is None:
                outputs, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_obs_masked)
            else:
                outputs = self.pinn(x_obs_masked)
                x_pde_masked = tf.multiply(x_pde, tf.expand_dims(pde_mask, 1))
                _, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_pde_masked)

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs * x_mask_expand, outputs), axis=0)
            loss_pde = tf.reduce_mean(tf.square(f_pde), axis=0)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            if self.extra_loss_len > 0:
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items) + tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items)

        if update_lambdas:
            self._update_lambdas_(x_pde, first_step, last_step, loss_obs, loss_pde, loss_items, trainables)

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y_obs * x_mask_expand, outputs)
        return loss_value, loss_obs, loss_pde, loss_extra_items

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        domain_mask,
        X_pde=None,
        pde_mask=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                domain_mask_train = domain_mask[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                    p_domain_mask = None
                else:
                    p_batch_train = X_pde[p_batch_indices]
                    p_domain_mask = pde_mask[p_batch_indices]

                loss_value_batch, loss_obs_batch, loss_pde_batch, loss_extra_batch = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    domain_mask_train,
                    p_batch_train,
                    p_domain_mask,
                    regularise,
                    step == 0,
                    step == last_step,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch.numpy()
                self.loss_obs += loss_obs_batch.numpy() * w_obs
                self.loss_pde += loss_pde_batch.numpy() * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    np.sum(loss_obs_batch.numpy() * w_obs)
                    + np.sum(loss_pde_batch.numpy() * w_pde)
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = self.train_acc_metric.result()
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()
            # end for epoch in range(epochs)

        return samples


class TINN_multi_nodes_masked3(TINN_multi_nodes):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_loss: Loss,
        extra_loss=[],
        nodes_n=2,
        node_names=None,
        optimizer=keras.optimizers.Adam(),
        train_acc_metric=keras.metrics.MeanSquaredError(),
        alpha=0.5,
        loss_penalty_power=1,
        print_precision=".5f",
        lambdas_pre=None,
        log_loss=False,
    ):
        super().__init__(
            pinn,
            pde_loss,
            extra_loss,
            nodes_n,
            node_names,
            optimizer,
            train_acc_metric,
            alpha,
            loss_penalty_power,
            print_precision,
            lambdas_pre,
            log_loss,
        )

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, update_lambdas, first_step, last_step, pde_masks):
        with tf.GradientTape(persistent=False) as tape:

            if x_pde is None:
                outputs, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_pde = self.pde_residual.loss_multi_nodes(self.pinn, x_pde)

            if pde_masks is not None:
                f_pde = f_pde

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs, outputs), axis=0)
            loss_pde = tf.reduce_mean(tf.square(f_pde), axis=1)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            if self.log_loss:
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * tf.math.log(loss_items))
            else:
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items)

            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            if self.extra_loss_len > 0:
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_value += tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []

        if update_lambdas:
            self._update_lambdas_(x_pde, first_step, last_step, loss_obs, loss_pde, loss_items, trainables)

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y_obs, outputs)
        return loss_value, loss_obs, loss_pde, loss_extra_items

    def _update_lambdas_(self, x_pde, first_step, last_step, loss_obs, loss_pde, loss_items, trainables):
        if x_pde is None:
            grads = [tf.gradients(loss_items[i], trainables) for i in range(loss_items.shape[0])]
            reduced_grads = tf.stack(
                [tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_i]) for grad_i in grads]
            )
        else:
            grads_obs = [tf.gradients(loss_obs[i], self.pinn.trainable_variables) for i in range(loss_obs.shape[0])]
            grads_pde = [tf.gradients(loss_pde[i], trainables) for i in range(loss_pde.shape[0])]
            reduced_grads = tf.stack(
                [tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_i]) for grad_i in grads_obs]
                + [tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_i]) for grad_i in grads_pde]
            )

        if first_step:
            for i in range(self.nodes_n * 2):
                self.grad_norms[i].assign(reduced_grads[i])

                self.loss_norms[i].assign(loss_items[i])

        else:
            for i in range(self.nodes_n * 2):
                self.grad_norms[i].assign_add(reduced_grads[i])

                self.loss_norms[i].assign_add(loss_items[i])

        if last_step:
            # Ws = tf.square(self.loss_norms) / tf.sqrt(self.grad_norms)
            Ws = tf.pow(self.loss_norms, self.loss_penalty_power) / tf.sqrt(self.grad_norms)

            w_total = tf.reduce_sum(Ws)
            for i in range(self.nodes_n * 2):
                self.lambdas[i].assign(
                    self.alpha * self.lambdas[i] + (1 - self.alpha) * 2 * self.nodes_n * Ws[i] / w_total
                )

    def train(
        self,
        epochs,
        batch_size,
        X,
        Y,
        X_pde=None,
        pde_masks=None,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        epoch_callback=None,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        x1_size = len(X)
        if X_pde is None:
            #  There is no secondary dataset
            x2_size = x1_size
        else:
            x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                if X_pde is None:
                    p_batch_train = None
                else:
                    p_batch_train = X_pde[p_batch_indices]
                loss_value_batch, loss_obs_batch, loss_pde_batch, loss_extra_batch = self.__train_step__(
                    x_batch_train,
                    y_batch_train,
                    p_batch_train,
                    regularise,
                    step == 0,
                    step == last_step,
                    pde_masks,
                )

                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                w_obs = len(o_batch_indices) / x1_size
                w_pde = w_obs if p_batch_train is None else len(p_batch_indices) / x2_size

                self.loss_reg_total += loss_value_batch.numpy()
                self.loss_obs += loss_obs_batch.numpy() * w_obs
                self.loss_pde += loss_pde_batch.numpy() * w_pde
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    np.sum(loss_obs_batch.numpy() * w_obs)
                    + np.sum(loss_pde_batch.numpy() * w_pde)
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = self.train_acc_metric.result()
            if isinstance(self.train_acc_metric, keras.metrics.MeanSquaredError):
                self.train_acc = np.sqrt(self.train_acc)
            self.loss_total = np.sqrt(self.loss_total)
            self.loss_obs = np.sqrt(self.loss_obs)
            self.loss_pde = np.sqrt(self.loss_pde)
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            if epoch_callback is not None:
                epoch_callback(epoch, samples)
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_()
            if stop_threshold >= float(self.train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return samples
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()
            # end for epoch in range(epochs)

        return samples
