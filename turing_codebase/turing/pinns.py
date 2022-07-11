import time
import copy
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .utils import indices


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
    def __net__(self, inputs):
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
        return self.__net__(X)

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


class Loss:
    def __init__(self, name, print_precision=".5f"):
        """Loss value that is calulated for the output of the pinn

        Args:
            name: The name of the Loss
            print_precision: f string format
        """
        self.name = name
        self.print_precision = print_precision
        self._trainables_ = ()

    # @tf.function
    def loss(self, pinn, x):
        """A tensorflow function that calculates and returns the loss

        Args:
           pinn:
           x: This is the value(s) that is returned from batch method.
                  Note that the values are converted to Tensors by Tensorflow

           It must return the loss values
        """
        pass

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


class TINN:
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
        self.pinn = pinn
        self.pde_loss = pde_loss
        self.extra_loss = extra_loss
        self.extra_loss_len = len(extra_loss)
        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.alpha = alpha
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

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=True) as tape:
            if x_pde is None:
                outputs, f_u, f_v = self.pde_loss.loss(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_u, f_v = self.pde_loss.loss(self.pinn, x_pde)
            loss_obs_u = tf.reduce_mean(tf.square(y_obs[:, 0] - outputs[:, 0]))
            loss_obs_v = tf.reduce_mean(tf.square(y_obs[:, 1] - outputs[:, 1]))
            loss_pde_u = tf.reduce_mean(tf.square(f_u))
            loss_pde_v = tf.reduce_mean(tf.square(f_v))
            trainables = self.pinn.trainable_variables + self.pde_loss.trainables()
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
                self.grad_norm_obs_u.assign(self.grad_norm_obs_u + temp_1)
                self.grad_norm_obs_v.assign(self.grad_norm_obs_v + temp_2)
                self.grad_norm_pde_u.assign(self.grad_norm_pde_u + temp_3)
                self.grad_norm_pde_v.assign(self.grad_norm_pde_v + temp_4)

                self.loss_norm_obs_u.assign(self.loss_norm_obs_u + loss_obs_u)
                self.loss_norm_obs_v.assign(self.loss_norm_obs_v + loss_obs_v)
                self.loss_norm_pde_u.assign(self.loss_norm_pde_u + loss_pde_u)
                self.loss_norm_pde_v.assign(self.loss_norm_pde_v + loss_pde_v)

            if last_step:
                w_1 = self.loss_norm_obs_u**2 / tf.sqrt(self.grad_norm_obs_u)
                w_2 = self.loss_norm_obs_v**2 / tf.sqrt(self.grad_norm_obs_v)
                w_3 = self.loss_norm_pde_u**2 / tf.sqrt(self.grad_norm_pde_u)
                w_4 = self.loss_norm_pde_v**2 / tf.sqrt(self.grad_norm_pde_v)

                w_total = w_1 + w_2 + w_3 + w_4
                self.lambda_obs_u.assign(self.alpha * self.lambda_obs_u + ((1 - self.alpha) * 4.0 * w_1) / w_total)
                self.lambda_obs_v.assign(self.alpha * self.lambda_obs_v + ((1 - self.alpha) * 4.0 * w_2) / w_total)
                self.lambda_pde_u.assign(self.alpha * self.lambda_pde_u + ((1 - self.alpha) * 4.0 * w_3) / w_total)
                self.lambda_pde_v.assign(self.alpha * self.lambda_pde_v + ((1 - self.alpha) * 4.0 * w_4) / w_total)

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
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        if sample_losses:
            arr_loss_total = np.zeros(epochs)
            arr_loss_regularisd_total = np.zeros(epochs)
            arr_obs_acc = np.zeros(epochs)
            arr_loss_obs_u = np.zeros(epochs)
            arr_loss_obs_v = np.zeros(epochs)
            arr_loss_pde_u = np.zeros(epochs)
            arr_loss_pde_v = np.zeros(epochs)
            if self.extra_loss_len > 0:
                arr_loss_extra = np.zeros((epochs, self.extra_loss_len))
        if sample_regularisations:
            arr_lambda_obs_u = np.zeros(epochs)
            arr_lambda_obs_v = np.zeros(epochs)
            arr_lambda_pde_u = np.zeros(epochs)
            arr_lambda_pde_v = np.zeros(epochs)
        if sample_gradients:
            arr_grads_obs_u = np.zeros(epochs)
            arr_grads_obs_v = np.zeros(epochs)
            arr_grads_pde_u = np.zeros(epochs)
            arr_grads_pde_v = np.zeros(epochs)

        def fill_return():
            ret = {"training_obs_accuracy": arr_obs_acc}
            if sample_losses:
                ret = {
                    **ret,
                    **{
                        "loss_total": arr_loss_total,
                        "loss_regularisd_total": arr_loss_regularisd_total,
                        "loss_obs_u": arr_loss_obs_u,
                        "loss_obs_v": arr_loss_obs_v,
                        "loss_pde_u": arr_loss_pde_u,
                        "loss_pde_v": arr_loss_pde_v,
                    },
                }
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        ret[f"loss_extra_{loss.name}"] = arr_loss_extra[:, i]
            if sample_regularisations:
                ret = {
                    **ret,
                    **{
                        "lambda_obs_u": arr_lambda_obs_u,
                        "lambda_obs_v": arr_lambda_obs_v,
                        "lambda_pde_u": arr_lambda_pde_u,
                        "lambda_pde_v": arr_lambda_pde_v,
                    },
                }
            if sample_gradients:
                ret = {
                    **ret,
                    **{
                        "grads_obs_u": arr_grads_obs_u,
                        "grads_obs_v": arr_grads_obs_v,
                        "grads_pde_u": arr_grads_pde_u,
                        "grads_pde_v": arr_grads_pde_v,
                    },
                }
            return ret

        #
        x1_size = len(X)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            loss_total, loss_reg_total, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v = 0, 0, 0, 0, 0, 0
            loss_extra = np.zeros(self.extra_loss_len)
            # Iterate over the batches of the dataset.
            for step, o_batch_indices in enumerate(indices(batch_size, shuffle, x1_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_extra_batch,
                ) = self.__train_step__(x_batch_train, y_batch_train, None, regularise, step == 0, step == last_step)

                if step > last_step:
                    last_step = step

                loss_reg_total += loss_value_batch
                loss_obs_u += loss_obs_u_batch
                loss_obs_v += loss_obs_v_batch
                loss_pde_u += loss_pde_u_batch
                loss_pde_v += loss_pde_v_batch
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                loss_extra += total_loss_extra_batch
                loss_total += (
                    loss_obs_u_batch + loss_obs_v_batch + loss_pde_u_batch + loss_pde_v_batch + total_loss_extra_batch
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            train_acc = self.train_acc_metric.result()
            arr_obs_acc[epoch] = train_acc
            if sample_losses:
                arr_loss_total[epoch] = loss_total
                arr_loss_regularisd_total[epoch] = loss_reg_total
                arr_loss_obs_u[epoch] = loss_obs_u
                arr_loss_obs_v[epoch] = loss_obs_v
                arr_loss_pde_u[epoch] = loss_pde_u
                arr_loss_pde_v[epoch] = loss_pde_v
                if self.extra_loss_len > 0:
                    arr_loss_extra[epoch, :] = loss_extra

            if sample_regularisations:
                arr_lambda_obs_u[epoch] = self.lambda_obs_u.numpy()
                arr_lambda_obs_v[epoch] = self.lambda_obs_v.numpy()
                arr_lambda_pde_u[epoch] = self.lambda_pde_u.numpy()
                arr_lambda_pde_v[epoch] = self.lambda_pde_v.numpy()
            if sample_gradients:
                arr_grads_obs_u[epoch] = np.sqrt(self.grad_norm_obs_u.numpy())
                arr_grads_obs_v[epoch] = np.sqrt(self.grad_norm_obs_v.numpy())
                arr_grads_pde_u[epoch] = np.sqrt(self.grad_norm_pde_u.numpy())
                arr_grads_pde_v[epoch] = np.sqrt(self.grad_norm_pde_v.numpy())
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                print(f"Training observations acc over epoch: {train_acc:{self.print_precision}}")
                print(
                    f"total loss: {loss_total:{self.print_precision}}, "
                    f"total regularisd loss: {loss_reg_total:{self.print_precision}}"
                )
                print(
                    f"obs u loss: {loss_obs_u:{self.print_precision}}, "
                    f"obs v loss: {loss_obs_v:{self.print_precision}}"
                )
                print(
                    f"pde u loss: {loss_pde_u:{self.print_precision}}, "
                    f"pde v loss: {loss_pde_v:{self.print_precision}}"
                )
                print(
                    f"lambda obs u: {self.lambda_obs_u.numpy():{self.print_precision}}, "
                    f"lambda obs v: {self.lambda_obs_v.numpy():{self.print_precision}}"
                )
                print(
                    f"lambda pde u: {self.lambda_pde_u.numpy():{self.print_precision}}, "
                    f"lambda pde v: {self.lambda_pde_v.numpy():{self.print_precision}}"
                )
                print(self.pde_loss.trainables_str())
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        print(f"extra loss {loss.name}: {loss_extra[i]:{self.print_precision}}")

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

            if stop_threshold >= float(train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return fill_return()
            # end for epoch in range(epochs)

        return fill_return()

    def train2(
        self,
        epochs,
        batch_size,
        X,
        Y,
        X_pde,
        print_interval=10,
        stop_threshold=0,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        if sample_losses:
            arr_loss_total = np.zeros(epochs)
            arr_loss_regularisd_total = np.zeros(epochs)
            arr_obs_acc = np.zeros(epochs)
            arr_loss_obs_u = np.zeros(epochs)
            arr_loss_obs_v = np.zeros(epochs)
            arr_loss_pde_u = np.zeros(epochs)
            arr_loss_pde_v = np.zeros(epochs)
            if self.extra_loss_len > 0:
                arr_loss_extra = np.zeros((epochs, self.extra_loss_len))
        if sample_regularisations:
            arr_lambda_obs_u = np.zeros(epochs)
            arr_lambda_obs_v = np.zeros(epochs)
            arr_lambda_pde_u = np.zeros(epochs)
            arr_lambda_pde_v = np.zeros(epochs)
        if sample_gradients:
            arr_grads_obs_u = np.zeros(epochs)
            arr_grads_obs_v = np.zeros(epochs)
            arr_grads_pde_u = np.zeros(epochs)
            arr_grads_pde_v = np.zeros(epochs)

        def fill_return():
            ret = {"training_obs_accuracy": arr_obs_acc}
            if sample_losses:
                ret = {
                    **ret,
                    **{
                        "loss_total": arr_loss_total,
                        "loss_regularisd_total": arr_loss_regularisd_total,
                        "loss_obs_u": arr_loss_obs_u,
                        "loss_obs_v": arr_loss_obs_v,
                        "loss_pde_u": arr_loss_pde_u,
                        "loss_pde_v": arr_loss_pde_v,
                    },
                }
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        ret[f"loss_extra_{loss.name}"] = arr_loss_extra[:, i]
            if sample_regularisations:
                ret = {
                    **ret,
                    **{
                        "lambda_obs_u": arr_lambda_obs_u,
                        "lambda_obs_v": arr_lambda_obs_v,
                        "lambda_pde_u": arr_lambda_pde_u,
                        "lambda_pde_v": arr_lambda_pde_v,
                    },
                }
            if sample_gradients:
                ret = {
                    **ret,
                    **{
                        "grads_obs_u": arr_grads_obs_u,
                        "grads_obs_v": arr_grads_obs_v,
                        "grads_pde_u": arr_grads_pde_u,
                        "grads_pde_v": arr_grads_pde_v,
                    },
                }
            return ret

        #
        x1_size = len(X)
        x2_size = len(X_pde)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            loss_total, loss_reg_total, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v = 0, 0, 0, 0, 0, 0
            loss_extra = np.zeros(self.extra_loss_len)
            # Iterate over the batches of the dataset.
            for step, (o_batch_indices, p_batch_indices) in enumerate(indices(batch_size, shuffle, x1_size, x2_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]
                x_batch_pde = X_pde[p_batch_indices]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_extra_batch,
                ) = self.__train_step__(
                    x_batch_train, y_batch_train, x_batch_pde, regularise, step == 0, step == last_step
                )

                if step > last_step:
                    last_step = step

                loss_reg_total += loss_value_batch
                loss_obs_u += loss_obs_u_batch
                loss_obs_v += loss_obs_v_batch
                loss_pde_u += loss_pde_u_batch
                loss_pde_v += loss_pde_v_batch
                total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                loss_extra += total_loss_extra_batch
                loss_total += (
                    loss_obs_u_batch + loss_obs_v_batch + loss_pde_u_batch + loss_pde_v_batch + total_loss_extra_batch
                )
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            train_acc = self.train_acc_metric.result()
            arr_obs_acc[epoch] = train_acc
            if sample_losses:
                arr_loss_total[epoch] = loss_total
                arr_loss_regularisd_total[epoch] = loss_reg_total
                arr_loss_obs_u[epoch] = loss_obs_u
                arr_loss_obs_v[epoch] = loss_obs_v
                arr_loss_pde_u[epoch] = loss_pde_u
                arr_loss_pde_v[epoch] = loss_pde_v
                if self.extra_loss_len > 0:
                    arr_loss_extra[epoch, :] = loss_extra

            if sample_regularisations:
                arr_lambda_obs_u[epoch] = self.lambda_obs_u.numpy()
                arr_lambda_obs_v[epoch] = self.lambda_obs_v.numpy()
                arr_lambda_pde_u[epoch] = self.lambda_pde_u.numpy()
                arr_lambda_pde_v[epoch] = self.lambda_pde_v.numpy()
            if sample_gradients:
                arr_grads_obs_u[epoch] = np.sqrt(self.grad_norm_obs_u.numpy())
                arr_grads_obs_v[epoch] = np.sqrt(self.grad_norm_obs_v.numpy())
                arr_grads_pde_u[epoch] = np.sqrt(self.grad_norm_pde_u.numpy())
                arr_grads_pde_v[epoch] = np.sqrt(self.grad_norm_pde_v.numpy())
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                print(f"Training observations acc over epoch: {train_acc:{self.print_precision}}")
                print(
                    f"total loss: {loss_total:{self.print_precision}}, "
                    f"total regularisd loss: {loss_reg_total:{self.print_precision}}"
                )
                print(
                    f"obs u loss: {loss_obs_u:{self.print_precision}}, "
                    f"obs v loss: {loss_obs_v:{self.print_precision}}"
                )
                print(
                    f"pde u loss: {loss_pde_u:{self.print_precision}}, "
                    f"pde v loss: {loss_pde_v:{self.print_precision}}"
                )
                print(
                    f"lambda obs u: {self.lambda_obs_u.numpy():{self.print_precision}}, "
                    f"lambda obs v: {self.lambda_obs_v.numpy():{self.print_precision}}"
                )
                print(
                    f"lambda pde u: {self.lambda_pde_u.numpy():{self.print_precision}}, "
                    f"lambda pde v: {self.lambda_pde_v.numpy():{self.print_precision}}"
                )
                print(self.pde_loss.trainables_str())
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        print(f"extra loss {loss.name}: {loss_extra[i]:{self.print_precision}}")

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            if epoch % print_interval == 0:
                print(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

            if stop_threshold >= float(train_acc):
                print("############################################")
                print("#               Early stop                 #")
                print("############################################")
                return fill_return()
            # end for epoch in range(epochs)

        return fill_return()
