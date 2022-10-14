import time
import copy
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .utils import indices
import pickle


class NN(tf.Module):
    def __init__(self, layers, lb, ub, dtype=tf.float32, **kwargs):
        """A dense Neural Net that is specified by layers argument.

        layers: input, dense layers and outputs dimensions
        lb    : An array of minimums of inputs (lower bounds)
        ub    : An array of maximums of inputs (upper bounds)
        """
        super().__init__(**kwargs)
        self.layers = layers
        self.num_layers = len(self.layers)
        self.lb = lb
        self.ub = ub
        self.dtype = dtype
        self.build()

    def build(self):
        """Create the state of the layers (weights)"""
        weights = []
        biases = []
        for i in range(0, self.num_layers - 1):
            W = self.xavier_init(size=[self.layers[i], self.layers[i + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)

        self.Ws = weights
        self.bs = biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.dtype), dtype=self.dtype
        )

    @tf.function
    def net(self, inputs):
        # Map the inputs to the range [-1, 1]
        H = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = self.Ws[-1]
        b = self.bs[-1]
        outputs = tf.add(tf.matmul(H, W), b)
        return outputs

    def __call__(self, inputs):
        """Defines the computation from inputs to outputs

        Args:
           inputs: A tensor that has a shape [None, D1], where
                   D1 is the input dimensionality, specified in
                   the first element of layes.

        Return:
                A tensor of the dense layer output that has a shape
                [None, Dn], where Dn is the dimensionality of the last
                layer, specificed by the last elements of the layer
                arguemnt.
        """
        X = tf.cast(inputs, self.dtype)
        return self.net(X)

    def gradients(self, inputs, outputs):
        """finds the first and second order griadients of outputs at inputs

        Args:
           inputs: A tensor that has a shape [None, D1], where
                   D1 is the input dimensionality, specified in
                   the first element of layes.
           outputs:  A tensor that has a shape [None, Dn], where
                   Dn is the output dimensionality, specified in
                   the last element of layes.

        Return:   The returns 'partial_1' and 'partial_2' are the first and second
                  order gradients, repsectivly. Each one is a list that its elements
                  corresponds to one of the NN's last layer output. e.g. if the last layer
                  has Dn outputs, each list has Dn tensors as an elements. The dimensionality
                  of the tensors are the same as inputs: [None, D1]

        """
        partials_1 = [tf.gradients(outputs[:, i], inputs)[0] for i in range(outputs.shape[1])]
        partials_2 = [tf.gradients(partials_1[i], inputs)[0] for i in range(outputs.shape[1])]
        return partials_1, partials_2

    def gradients_tape(self, inputs, outputs, tape):
        """finds the first and second order griadients of outputs at inputs

        Args:
           inputs: A tensor that has a shape [None, D1], where
                   D1 is the input dimensionality, specified in
                   the first element of layes.
           outputs:  A tensor that has a shape [None, Dn], where
                   Dn is the output dimensionality, specified in
                   the last element of layes.
           tape:   Gradient Tape object, for eager mode.
                   The outputs must be the list
                   of Tensors.

        Return:   The returns 'partial' gradients. It is a list that its elements
                  corresponds to one of the NN's last layer output. e.g. if the last layer
                  has Dn outputs, the list has Dn tensors as an elements. The dimensionality
                  of the tensors is the same as inputs: [None, D1]

        """
        partials = [tape.gradient(outputs[i], inputs) for i in range(len(outputs))]
        return partials

    def copy(self):
        return copy.deepcopy(self)

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


class PDE_Residual(tf.Module):
    def __init__(self, name, print_precision=".5f", **kwargs):
        """Loss value that is calulated for the output of the pinn

        Args:
            name: The name of the Loss
            print_precision: f string format
        """
        # self.name = name
        super().__init__(name=name, **kwargs)
        self.print_precision = print_precision
        self._trainables_ = ()

    # @tf.function
    def residual(self, pinn, x):
        """A tensorflow function that calculates and returns the loss

        Args:
           pinn:
           x: This is the value(s) that is returned from batch method.
                  Note that the values are converted to Tensors by Tensorflow

           It must return the loss values
        """
        pass

    @tf.function
    def residual_multi_nodes(self, pinn, x):
        """A tensorflow function that override the loss method

        Returns a concatenated tensor of derivatives, which
        TINN_multi_node expects.
        """
        res = self.residual(pinn, x)
        outputs = res[0]
        #  return outputs, tf.concat([tf.expand_dims(f_u, axis=1) for f_u in res[1:]], axis=1)
        #  return outputs, tf.concat([f_u for f_u in res[1:]], axis=0)
        #  return outputs, tf.concat([tf.expand_dims(f_u, axis=1) for f_u in res[1:]], axis=1)
        return outputs, res[1:]

    def trainables(self):
        """Retruns a tuple of Tensorflow variables for training

        If the loss class has some tarinable variables, it can
        return them as a tuple. These variables will be updated
        by optimiser, if they are already part of the computation
        graph.

        """
        return self._trainables_

    def trainables_str(self):
        s = ""
        CR = "\n"
        C = ""
        t_vars = self.trainables()
        if len(t_vars) > 0:
            s += "".join(
                [
                    f"{v.name.split(':')[0]}: {self.__get_val__(v):{self.print_precision}} {CR if (i+1)%4 == 0 else C}"
                    for i, v in enumerate(t_vars)
                ]
            )
        return s

    def __get_val__(self, item):
        val = item.numpy()
        if type(val) is float:
            return val
        else:
            return val[0]

    def parameter_names(self):
        return [f"{v.name.split(':')[0]}" for v in self.trainables()]

    # def __getstate__(self):

    #    return

    def save(self, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        tf.saved_model.save(self, str(path))
        #
        # import os
        # if not pathlib.Path(path.joinpath(name)).exists():
        #   os.makedirs(path.joinpath(name))
        with open(f"{str(path)}.pkl", "wb") as f:
            pickle.dump(self, f)

    # def __getstate__(self):
    #    return

    @classmethod
    def restore(cls, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}.pkl", "rb") as f:
            model = pickle.load(f)
        return model


class Loss:
    def __init__(self, *childs):
        self.childs = childs

    def norm(self, x, axis=None):
        if self.childs is None:
            return 0
        else:
            ret = self.childs[0].norm(x)
            for c in self.childs[1:]:
                ret += self.childs[0].norm(x)
        return ret

    def __add__(self, left):
        return Loss(self, left)


class TINN(tf.Module):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_residual: PDE_Residual,
        loss: Loss,
        extra_loss=[],
        alpha=0.5,
        loss_penalty_power=2,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pinn = pinn
        self.pde_residual = pde_residual
        self.extra_loss = extra_loss
        self.extra_loss_len = len(extra_loss)
        self.loss = loss
        self.alpha = tf.Variable(alpha, dtype=pinn.dtype, trainable=False)
        self.loss_penalty_power = tf.Variable(loss_penalty_power, dtype=pinn.dtype, trainable=False)
        self.print_precision = print_precision
        #
        self.lambda_obs_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_obs_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        self.lambda_pde_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)

        self.grad_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.grad_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        self.loss_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        self.loss_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        self._reset_losses_()

    def _reset_losses_(self):
        self.loss_total = 0
        self.loss_reg_total = 0
        self.loss_obs_u = 0
        self.loss_obs_v = 0
        self.loss_pde_u = 0
        self.loss_pde_v = 0
        self.loss_extra = np.zeros(self.extra_loss_len)
        self.train_acc = 0

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, update_lambdas, first_step, last_step, optimizer, train_acc_metric):
        with tf.GradientTape(persistent=True) as tape:
            if x_pde is None:
                outputs, f_u, f_v = self.pde_residual.residual(self.pinn, x_obs)
            else:
                outputs = self.pinn.net(x_obs)
                _, f_u, f_v = self.pde_residual.residual(self.pinn, x_pde)
            loss_obs_u = self.loss.norm(
                y_obs[:, 0] - outputs[:, 0]
            )  # tf.reduce_mean(tf.square(y_obs[:,0]-outputs[:,0]))
            loss_obs_v = self.loss.norm(
                y_obs[:, 1] - outputs[:, 1]
            )  # tf.reduce_mean(tf.square(y_obs[:,1]-outputs[:,1]))
            loss_pde_u = self.loss.norm(f_u)  # tf.reduce_mean(tf.square(f_u))
            loss_pde_v = self.loss.norm(f_v)  # tf.reduce_mean(tf.square(f_v))
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
        optimizer.apply_gradients(zip(grads, trainables))
        train_acc_metric.update_state(y_obs, outputs)
        return loss_value, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, loss_extra_items

    def _update_lambdas_(
        self, x_pde, first_step, last_step, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
    ):
        if x_pde is None:
            grad_obs_u = tape.gradient(loss_obs_u, trainables)
            grad_obs_v = tape.gradient(loss_obs_v, trainables)
        else:
            grad_obs_u = tape.gradient(loss_obs_u, self.pinn.trainable_variables)
            grad_obs_v = tape.gradient(loss_obs_v, self.pinn.trainable_variables)
        grad_pde_u = tape.gradient(loss_pde_u, trainables)
        grad_pde_v = tape.gradient(loss_pde_v, trainables)

        temp_1 = tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_obs_u])
        temp_2 = tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_obs_v])
        temp_3 = tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_pde_u])
        temp_4 = tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_pde_v])

        if first_step:
            self.grad_norm_obs_u.assign(temp_1)
            self.grad_norm_obs_v.assign(temp_2)
            self.grad_norm_pde_u.assign(temp_3)
            self.grad_norm_pde_v.assign(temp_4)

            self.loss_norm_obs_u.assign(loss_obs_u)
            self.loss_norm_obs_v.assign(loss_obs_v)
            self.loss_norm_pde_u.assign(loss_pde_u)
            self.loss_norm_pde_v.assign(loss_pde_v)

        else:
            self.grad_norm_obs_u.assign_add(temp_1)
            self.grad_norm_obs_v.assign_add(temp_2)
            self.grad_norm_pde_u.assign_add(temp_3)
            self.grad_norm_pde_v.assign_add(temp_4)

            self.loss_norm_obs_u.assign_add(loss_obs_u)
            self.loss_norm_obs_v.assign_add(loss_obs_v)
            self.loss_norm_pde_u.assign_add(loss_pde_u)
            self.loss_norm_pde_v.assign_add(loss_pde_v)

        if last_step:
            # w_1 = self.loss_norm_obs_u**2 / tf.sqrt(self.grad_norm_obs_u)
            # w_2 = self.loss_norm_obs_v**2 / tf.sqrt(self.grad_norm_obs_v)
            # w_3 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_u)
            # w_4 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_v)
            w_1 = tf.pow(self.loss_norm_obs_u, self.loss_penalty_power) / tf.sqrt(self.grad_norm_obs_u)
            w_2 = tf.pow(self.loss_norm_obs_v, self.loss_penalty_power) / tf.sqrt(self.grad_norm_obs_v)
            w_3 = tf.pow(self.loss_norm_pde_u, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_u)
            w_4 = tf.pow(self.loss_norm_pde_v, self.loss_penalty_power) / tf.sqrt(self.grad_norm_pde_v)

            w_total = w_1 + w_2 + w_3 + w_4
            self.lambda_obs_u.assign(self.alpha * self.lambda_obs_u + ((1 - self.alpha) * 4.0 * w_1) / w_total)
            self.lambda_obs_v.assign(self.alpha * self.lambda_obs_v + ((1 - self.alpha) * 4.0 * w_2) / w_total)
            self.lambda_pde_u.assign(self.alpha * self.lambda_pde_u + ((1 - self.alpha) * 4.0 * w_3) / w_total)
            self.lambda_pde_v.assign(self.alpha * self.lambda_pde_v + ((1 - self.alpha) * 4.0 * w_4) / w_total)

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
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    + total_loss_extra_batch * w_obs
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = train_acc_metric.result()
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
            train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples

    def _create_samples_(self, epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters):
        # Samplling arrays
        ret = {"training_obs_accuracy": np.zeros(epochs)}
        if sample_losses:
            ret = {
                **ret,
                **{
                    "loss_total": np.zeros(epochs),
                    "loss_regularisd_total": np.zeros(epochs),
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
                },
            }
        if sample_parameters:
            for param in self.pde_residual.trainables():
                ret[f"{param.name.split(':')[0]}"] = np.zeros(epochs)
        return ret

    def _store_samples_(
        self, samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
    ):
        samples["training_obs_accuracy"][epoch] = self.train_acc
        if sample_losses:
            samples["loss_total"][epoch] = self.loss_total
            samples["loss_regularisd_total"][epoch] = self.loss_reg_total
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
        if sample_gradients:
            samples["grads_obs_u"][epoch] = np.sqrt(self.grad_norm_obs_u.numpy())
            samples["grads_obs_v"][epoch] = np.sqrt(self.grad_norm_obs_v.numpy())
            samples["grads_pde_u"][epoch] = np.sqrt(self.grad_norm_pde_u.numpy())
            samples["grads_pde_v"][epoch] = np.sqrt(self.grad_norm_pde_v.numpy())

        if sample_parameters:
            for param in self.pde_residual.trainables():
                samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()

    def _print_metrics_(self):
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
        print(
            f"lambda obs u: {self.lambda_obs_u.numpy():{self.print_precision}}, "
            f"lambda obs v: {self.lambda_obs_v.numpy():{self.print_precision}}"
        )
        print(
            f"lambda pde u: {self.lambda_pde_u.numpy():{self.print_precision}}, "
            f"lambda pde v: {self.lambda_pde_v.numpy():{self.print_precision}}"
        )
        print(self.pde_residual.trainables_str())
        if self.extra_loss_len > 0:
            for i, loss in enumerate(self.extra_loss):
                print(f"extra loss {loss.name}: {self.loss_extra[i]:{self.print_precision}}")


class TINN_inverse:
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        pde_residual: PDE_Residual,
        loss: Loss,
        extra_loss=[],
        non_zero_loss=None,
        alpha=0.5,
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
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
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
            w_1 = self.loss_norm_obs_u**2 / tf.sqrt(self.grad_norm_obs_u)
            w_2 = self.loss_norm_obs_v**2 / tf.sqrt(self.grad_norm_obs_v)
            w_3 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_u)
            w_4 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_v)
            w_5 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_params_u)
            w_6 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_params_v)

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
