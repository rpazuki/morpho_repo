from email.errors import FirstHeaderLineIsContinuationDefect
import time
import copy
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .utils import indices
from .tf_utils import TINN_Dataset
from .tf_utils import Loss_Grad_Type
import pickle

# from functools import reduce
# import operator


def default_printer(s):
    print(s)


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
        self.__version__ = 0.1
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


class NN2(NN):
    def __init__(self, layers, lb, ub, dtype=tf.float32, **kwargs):
        super().__init__(layers, lb, ub, dtype, **kwargs)

    def build(self):
        """Create the state of the layers (weights)"""
        weights_2 = []
        biases_2 = []
        for _ in range(self.layers[-1]):
            weights = []
            biases = []
            for i in range(0, self.num_layers - 2):
                W = self.xavier_init(size=[self.layers[i], self.layers[i + 1]])
                b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=self.dtype), dtype=self.dtype)
                weights.append(W)
                biases.append(b)
            W = self.xavier_init(size=[self.layers[i + 1], 1])
            b = tf.Variable(tf.zeros([1, 1], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)
            weights_2.append(weights)
            biases_2.append(biases)

        self.Ws = weights_2
        self.bs = biases_2

    @tf.function
    def net(self, inputs):
        def get_output(H, output):
            H2 = H
            for W, b in zip(self.Ws[output][:-1], self.bs[output][:-1]):
                H2 = tf.tanh(tf.add(tf.matmul(H2, W), b))

            W = self.Ws[output][-1]
            b = self.bs[output][-1]
            return tf.add(tf.matmul(H2, W), b)

        # Map the inputs to the range [-1, 1]
        H = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        return tf.squeeze(tf.stack([get_output(H, output) for output in range(self.layers[-1])], axis=1), [2])


class Res_NN(NN):
    def __init__(self, layers, lb, ub, dtype=tf.float32, **kwargs):
        super().__init__(layers, lb, ub, dtype, **kwargs)

    @tf.function
    def net(self, inputs):
        # Map the inputs to the range [-1, 1]
        H = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        H_sum = None
        for i, (W, b) in enumerate(zip(self.Ws[:-1], self.bs[:-1])):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            if i % 2 == 0 and i != 0:
                if H_sum is None:
                    H_sum = H
                else:
                    H += H_sum
                    H_sum = H

        W = self.Ws[-1]
        b = self.bs[-1]
        outputs = tf.add(tf.matmul(H, W), b)
        return outputs


class Loss(tf.Module):
    def __init__(
        self,
        name,
        loss_grad_type: Loss_Grad_Type = Loss_Grad_Type.BOTH,
        regularise=True,
        residual_ret_num=2,
        residual_ret_names="",
        print_precision=".5f",
        **kwargs,
    ):
        """Loss value that is calulated for the output of the pinn

        Args:
            name: The name of the Loss
            print_precision: f string format
        """
        # self.name = name
        super().__init__(name=name, **kwargs)
        self.print_precision = print_precision
        self._trainables_ = ()
        self.residual_ret_num = residual_ret_num
        if residual_ret_names == "":
            self.residual_ret_names = tuple(["" for _ in range(residual_ret_num)])
        else:
            self.residual_ret_names = residual_ret_names
        self.loss_grad_type = loss_grad_type
        self.regularise = regularise
        self.__version__ = 0.1

    def add_trainable(self, param, param_name):
        setattr(self, param_name, param.build())
        self._trainables_ += param.trainable

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

    def derivatives(self, pinn, x):
        y = pinn.net(x)
        p1, p2 = pinn.gradients(x, y)

        u = y[:, 0]
        v = y[:, 1]

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
        return y, u, u_t, u_xx, u_yy, v, v_t, v_xx, v_yy

    def derivatives_multi_nodes(self, pinn, x):
        y = pinn.net(x)
        p1, p2 = pinn.gradients(x, y)

        # A = outputs[:, 0]
        # B = outputs[:, 1]
        # C = outputs[:, 2]
        # D = outputs[:, 3]
        # E = outputs[:, 4]
        # F = outputs[:, 5]
        vs = [y[:, i] for i in range(y.shape[1])]

        # A_x = tf.cast(p1[0][:, 0], pinn.dtype)
        # A_y = tf.cast(p1[0][:, 1], pinn.dtype)
        # A_t = tf.cast(p1[0][:, 2], pinn.dtype)

        # A_xx = tf.cast(tf.gradients(A_x, x)[0][:, 0], pinn.dtype)
        # A_yy = tf.cast(tf.gradients(A_y, x)[0][:, 1], pinn.dtype)
        # A_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        # A_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        # B_x = tf.cast(p1[1][:, 0], pinn.dtype)
        # B_y = tf.cast(p1[1][:, 1], pinn.dtype)
        # B_t = tf.cast(p1[1][:, 2], pinn.dtype)

        # B_xx = tf.cast(tf.gradients(B_x, x)[0][:, 0], pinn.dtype)
        # B_yy = tf.cast(tf.gradients(B_y, x)[0][:, 1], pinn.dtype)
        # B_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        # B_yy = tf.cast(p2[1][:, 1], pinn.dtype)

        # C_t = tf.cast(p1[2][:, 2], pinn.dtype)
        # D_t = tf.cast(p1[3][:, 2], pinn.dtype)
        # E_t = tf.cast(p1[4][:, 2], pinn.dtype)
        # F_t = tf.cast(p1[5][:, 2], pinn.dtype)

        v_ts = [tf.cast(p1[i][:, 2], pinn.dtype) for i in range(y.shape[1])]
        v_xxs = [tf.cast(p2[i][:, 0], pinn.dtype) for i in range(2)]
        v_yys = [tf.cast(p2[i][:, 1], pinn.dtype) for i in range(2)]

        return [y, *vs, *v_ts, *v_xxs, *v_yys]

    def derivatives_steady(self, pinn, x):
        y = pinn.net(x)
        _, p2 = pinn.gradients(x, y)

        u = y[:, 0]
        v = y[:, 1]

        u_xx = tf.cast(p2[0][:, 0], pinn.dtype)
        u_yy = tf.cast(p2[0][:, 1], pinn.dtype)

        v_xx = tf.cast(p2[1][:, 0], pinn.dtype)
        v_yy = tf.cast(p2[1][:, 1], pinn.dtype)
        return y, u, u_xx, u_yy, v, v_xx, v_yy

    def derivatives_steady_multi_nodes(self, pinn, x):
        y = pinn.net(x)
        _, p2 = pinn.gradients(x, y)

        vs = [y[:, i] for i in range(y.shape[1])]

        v_xxs = [tf.cast(p2[i][:, 0], pinn.dtype) for i in range(2)]
        v_yys = [tf.cast(p2[i][:, 1], pinn.dtype) for i in range(2)]

        return [y, *vs, *v_xxs, *v_yys]

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

    def set_parameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k].set_value(v)

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


class Norm:
    def __init__(self, *childs):
        self.childs = childs
        self.__version__ = 0.1

    def reduce_norm(self, tupled_x, axis=None):
        if len(self.childs) == 0:
            return 0
        else:

            ret = self.childs[0].reduce_norm(tupled_x, axis)
            for c in self.childs[1:]:
                ret += c.reduce_norm(tupled_x, axis)
        return ret

    def __add__(self, left):
        return Norm(self, left)


class TINN(tf.Module):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        losses,
        norm: Norm,
        no_input_losses=[],
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        alpha=0.5,
        loss_penalty_power=2,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(losses) > 0
        self.pinn = pinn
        self.losses = losses
        self.no_input_losses = no_input_losses
        self.norm = norm
        self.optimizer = optimizer
        self.alpha = tf.Variable(alpha, dtype=pinn.dtype, trainable=False)
        self.loss_penalty_power = tf.Variable(loss_penalty_power, dtype=pinn.dtype, trainable=False)
        self.print_precision = print_precision
        self.weight_values = None
        #
        self.num_loss = len(losses)
        self.num_no_input_loss = len(no_input_losses)
        self.total_num_loss = self.num_loss + self.num_no_input_loss

        self.regularisable_loss_indices = [i for i, loss in enumerate(self.losses) if loss.regularise]
        self.regularisable_no_input_loss_indices = [i for i, loss in enumerate(self.no_input_losses) if loss.regularise]
        self.unregularisable_loss_indices = [i for i, loss in enumerate(self.losses) if loss.regularise is False]
        self.unregularisable_no_input_loss_indices = [
            i for i, loss in enumerate(self.no_input_losses) if loss.regularise is False
        ]

        self.num_regularisers = int(
            np.sum([self.losses[i].residual_ret_num for i in self.regularisable_loss_indices])
            + np.sum([self.losses[i].residual_ret_num for i in self.regularisable_no_input_loss_indices])
        )

        self.lambdas = [tf.Variable(1.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]

        self.grad_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]

        self.loss_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]

        self.__version__ = 0.1

        self._reset_losses_()

    def _reset_losses_(self):
        self.loss_total = 0
        self.loss_reg_total = 0
        self.train_acc = 0
        self.loss_values = np.array([np.zeros(loss.residual_ret_num) for loss in self.losses])
        self.loss_no_input_values = np.array([np.zeros(loss.residual_ret_num) for loss in self.no_input_losses])

    @tf.function
    def __train_step__(self, elements, lambdas_state, dummy_train=False):
        """_summary_

        Args:
            elements (_type_): _description_
            lambdas_state ([int]): [0] == No Lambda
                                   [1] == Update_Lambda start
                                   [2] == Update_Lambda in Middle
                                   [3] == Update_Lambda end
            train_acc_metric (_type_): _description_
            dummy_train (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        with tf.GradientTape(persistent=True) as tape:

            loss_items = [loss.residual(self.pinn, elements[index]) for index, loss in enumerate(self.losses)]
            loss_items_norm = [self.norm.reduce_norm(item) for item in loss_items]
            trainables = self.pinn.trainable_variables
            regularisable_trainables = self.pinn.trainable_variables
            for loss in self.losses:
                trainables += loss.trainables()
                if loss.regularise:
                    regularisable_trainables += loss.trainables()
            if len(self.no_input_losses) > 0:
                no_input_loss_items = [loss.residual(self.pinn, None) for loss in self.no_input_losses]
                no_input_loss_items_norm = [self.norm.reduce_norm(item) for item in no_input_loss_items]
                for loss in self.no_input_losses:
                    trainables += loss.trainables()
                    if loss.regularise:
                        regularisable_trainables += loss.trainables()
            else:
                no_input_loss_items = []
                no_input_loss_items_norm = 0.0

            regularisable_norms = [loss_items_norm[i] for i in self.regularisable_loss_indices] + [
                no_input_loss_items_norm[i] for i in self.regularisable_no_input_loss_indices
            ]
            if len(regularisable_norms) > 0:
                regularisable_norms = tf.concat(regularisable_norms, axis=0)
                regularised_loss_value = tf.reduce_sum(tf.stack(self.lambdas) * regularisable_norms)
            else:
                regularisable_norms = 0.0
                regularised_loss_value = 0.0

            unregularisable_norms = [loss_items_norm[i] for i in self.unregularisable_loss_indices] + [
                no_input_loss_items_norm[i] for i in self.unregularisable_no_input_loss_indices
            ]
            if len(unregularisable_norms) > 0:
                unregularisable_norms = tf.concat(unregularisable_norms, axis=0)
                unregularised_loss_value = tf.reduce_sum(unregularisable_norms)
            else:
                unregularisable_norms = 0.0
                unregularised_loss_value = 0.0

            loss_value = regularised_loss_value + unregularised_loss_value

        if lambdas_state[0] > 0:
            self._update_lambdas_(lambdas_state, regularisable_norms, regularisable_trainables)

        grads = tape.gradient(loss_value, trainables)
        if dummy_train:
            self.dummy_update_grads(trainables)
        else:
            self.optimizer.apply_gradients(zip(grads, trainables))

        return loss_value, loss_items_norm, no_input_loss_items_norm

    def dummy_update_grads(self, trainables):
        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in trainables]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in trainables]
        # Apply gradients which don't do nothing
        self.optimizer.apply_gradients(zip(zero_grads, trainables))
        # Reload variables
        [x.assign(y) for x, y in zip(trainables, saved_vars)]

    def _update_lambdas_(self, lambdas_state, regularisable_norms, trainables):
        grads = [tf.gradients(regularisable_norms[i], trainables) for i in range(regularisable_norms.shape[0])]
        reduced_grads = tf.stack(
            [tf.reduce_sum([tf.reduce_sum(tf.square(item)) for item in grad_i]) for grad_i in grads]
        )

        if lambdas_state[0] == 1:
            for i in range(self.num_regularisers):
                self.grad_norms[i].assign(reduced_grads[i])

                self.loss_norms[i].assign(regularisable_norms[i])

        else:
            for i in range(self.num_regularisers):
                self.grad_norms[i].assign_add(reduced_grads[i])

                self.loss_norms[i].assign_add(regularisable_norms[i])

        if lambdas_state[0] == 3:
            Ws = tf.pow(self.loss_norms, self.loss_penalty_power) / tf.sqrt(self.grad_norms)

            w_total = tf.reduce_sum(Ws)
            for i in range(self.num_regularisers):
                self.lambdas[i].assign(
                    self.alpha * self.lambdas[i] + (1 - self.alpha) * self.num_regularisers * Ws[i] / w_total
                )

    def train(
        self,
        epochs,
        batch_size,
        dataset: TINN_Dataset,
        print_interval=10,
        stop_threshold=0,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        regularise=True,
        regularise_interval=1,
        train_acc_metric=keras.metrics.MeanSquaredError(),
        relative_mean_denominators=None,
        printer=default_printer,
        epoch_callback=None,
    ):

        # Samplling arrays
        samples = self._create_samples_(epochs, sample_losses, sample_regularisations, sample_parameters)
        #
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        if print_interval > 0:
            start_time = time.time()
        x_sizes = np.max(dataset.sizes)
        #
        if relative_mean_denominators is not None:
            assert np.sum([loss.shape[0] for loss in self.loss_values]) == len(relative_mean_denominators)
        #
        dataset2 = dataset.cache().batch(batch_size)
        for epoch in range(epochs):
            if print_interval > 0 and epoch % print_interval == 0:
                printer(f"\nStart of epoch {epoch:d}")

            step = 0
            # Iterate over the batches of the dataset.
            for element in dataset2:
                if regularise and epoch % regularise_interval == 0:
                    if step == 0:
                        lambdas_state = [1]
                    elif step == last_step:
                        lambdas_state = [3]
                    else:
                        lambdas_state = [2]
                else:
                    lambdas_state = [0]

                (
                    loss_reg_total_batch,
                    loss_values_batch,
                    loss_no_input_values_batch,
                ) = self.__train_step__(element, lambdas_state)
                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                ws = np.array([len(item) for item in element]) / x_sizes  # [:, np.newaxis]

                self.loss_reg_total += loss_reg_total_batch
                if self.num_loss > 0:
                    loss_values_batch = np.array([item.numpy() for item in loss_values_batch])
                    index = 0
                    for i, norms in enumerate(loss_values_batch):
                        if relative_mean_denominators is not None:
                            norms /= relative_mean_denominators[index : index + len(norms)]
                            index += len(norms)
                        self.loss_values[i] += ws[i] * norms
                        self.loss_total += np.sum(np.sum(norms))

                if self.num_no_input_loss > 0:
                    loss_no_input_values_batch = np.array([item.numpy() for item in loss_no_input_values_batch])
                    self.loss_no_input_values += loss_no_input_values_batch
                    self.loss_total += np.sum(np.sum([np.sum(item) for item in loss_no_input_values_batch]))

                step += 1
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = train_acc_metric.result()
            # update the samples
            self._store_samples_(samples, epoch, sample_losses, sample_regularisations, sample_parameters)
            if epoch_callback is not None:
                epoch_callback(epoch, samples, self)
            # Display metrics at the end of each epoch.
            if print_interval > 0 and epoch % print_interval == 0:
                self._print_metrics_(sample_regularisations, sample_parameters, printer)
            if stop_threshold >= float(self.loss_total):
                printer("############################################")
                printer("#               Early stop                 #")
                printer("############################################")
                self._cut_samples_(samples, epoch, sample_losses, sample_regularisations, sample_parameters)
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            self._reset_losses_()
            if print_interval > 0 and epoch % print_interval == 0:
                printer(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples

    def _create_samples_(self, epochs, sample_losses, sample_regularisations, sample_parameters):
        # Samplling arrays
        ret = {"training_obs_accuracy": np.zeros(epochs)}
        if sample_losses:
            ret = {
                **ret,
                **{
                    "loss_total": np.zeros(epochs),
                    "loss_regularisd_total": np.zeros(epochs),
                },
            }
            for i, loss in enumerate(self.losses):
                ret[f"{loss.name}_values"] = np.zeros((loss.residual_ret_num, epochs))
            for i, loss in enumerate(self.no_input_losses):
                ret[f"{loss.name}_values"] = np.zeros((loss.residual_ret_num, epochs))

        if sample_regularisations:
            ret = {
                **ret,
                **{
                    "lambdas": np.zeros((len(self.lambdas), epochs)),
                    "grads": np.zeros((len(self.grad_norms), epochs)),
                },
            }
        if sample_parameters:
            for loss in self.losses:
                for param in loss.trainables():
                    ret[f"{param.name.split(':')[0]}"] = np.zeros(epochs)
            for loss in self.no_input_losses:
                for param in loss.trainables():
                    ret[f"{param.name.split(':')[0]}"] = np.zeros(epochs)
        return ret

    def _store_samples_(self, samples, epoch, sample_losses, sample_regularisations, sample_parameters):
        samples["training_obs_accuracy"][epoch] = self.train_acc
        if sample_losses:
            samples["loss_total"][epoch] = self.loss_total
            samples["loss_regularisd_total"][epoch] = self.loss_reg_total
            for i, loss in enumerate(self.losses):
                samples[f"{loss.name}_values"][:, epoch] = self.loss_values[i]
            for i, loss in enumerate(self.no_input_losses):
                samples[f"{loss.name}_values"][:, epoch] = self.loss_no_input_values[i]

        if sample_regularisations:
            lambdas = np.array([item.numpy() for item in self.lambdas])
            samples["lambdas"][:, epoch] = lambdas
            grads = np.array([item.numpy() for item in self.grad_norms])
            samples["grads"][:, epoch] = grads

        if sample_parameters:
            for loss in self.losses:
                for param in loss.trainables():
                    samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()
            for loss in self.no_input_losses:
                for param in loss.trainables():
                    samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()

    def _cut_samples_(self, samples, epoch, sample_losses, sample_regularisations, sample_parameters):
        samples["training_obs_accuracy"] = samples["training_obs_accuracy"][:epoch]
        if sample_losses:
            samples["loss_total"] = samples["loss_total"][:epoch]
            samples["loss_regularisd_total"] = samples["loss_regularisd_total"][:epoch]
            for i, loss in enumerate(self.losses):
                samples[f"{loss.name}_values"] = samples[f"{loss.name}_values"][:, :epoch]
            for i, loss in enumerate(self.no_input_losses):
                samples[f"{loss.name}_values"] = samples[f"{loss.name}_values"][:, :epoch]

        if sample_regularisations:
            grads = np.array([item.numpy() for item in self.grad_norms])
            samples["grads"] = samples["grads"][:, :epoch]

        if sample_parameters:
            for loss in self.losses:
                for param in loss.trainables():
                    samples[f"{param.name.split(':')[0]}"] = samples[f"{param.name.split(':')[0]}"][:epoch]
            for loss in self.no_input_losses:
                for param in loss.trainables():
                    samples[f"{param.name.split(':')[0]}"] = samples[f"{param.name.split(':')[0]}"][:epoch]

    def _print_metrics_(self, sample_regularisations, sample_parameters, printer=default_printer):
        # printer(f"Training observations acc over epoch: {self.train_acc:{self.print_precision}}")
        printer(
            f"total loss: {self.loss_total:{self.print_precision}}, "
            f"total regularised loss: {self.loss_reg_total:{self.print_precision}}"
        )
        # printer("")
        start_index = 0
        for i, loss in enumerate(self.losses):
            values = self.loss_values[i]
            printer(f"{loss.name} -> \n" + self.CRLR_str(values, loss.residual_ret_names, start_index=start_index))
            start_index += len(values)
        # printer("")
        for i, loss in enumerate(self.no_input_losses):
            printer(
                f"{loss.name} -> \n"
                + self.CRLR_str(self.loss_no_input_values[i], loss.residual_ret_names, start_index=start_index)
            )
            start_index += len(self.loss_vloss_no_input_valuesalues[i])
        # printer("")
        if sample_regularisations:
            lambdas = [item.numpy() for item in self.lambdas]
            printer(self.CRLR_str(lambdas, title="lambdas"))

        if sample_parameters:
            for loss in self.losses:
                s = loss.trainables_str()
                if s != "":
                    printer(s)
            for loss in self.no_input_losses:
                s = loss.trainables_str()
                if s != "":
                    printer(s)

    def CRLR_str(self, t_vars, labels=None, title="", start_index=0):
        s = ""
        CR = "\n"
        C = ""
        if labels is None:
            labels = ["" for _ in t_vars]
        if len(t_vars) > 0:
            s += "".join(
                [
                    f"({i+1+start_index}) {title} {labels[i]}: {v:{self.print_precision}} {CR if (i+1)%3 == 0 else C}"
                    for i, v in enumerate(t_vars)
                ]
            )
        return s

    def save(self, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)

        # save optimizer state
        weight_values = self.optimizer.get_weights()  # tf.keras.backend.batch_get_value(symbolic_weights)
        with open(f"{str(path)}_optimizer.pkl", "wb") as f:
            pickle.dump(weight_values, f)
        # remove optimiser
        opt = self.optimizer
        # self.optimizer = None
        delattr(self, "optimizer")
        with open(f"{str(path)}.pkl", "wb") as f:
            pickle.dump(self, f)
        # Restore optimizer
        # self.optimizer = opt
        setattr(self, "optimizer", opt)

        # save optimizer config
        conf = self.optimizer.get_config()
        with open(f"{str(path)}_optimizer_config.pkl", "wb") as f:
            pickle.dump(conf, f)

    @classmethod
    def restore(cls, path_dir, name, ds):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{str(path)}_optimizer.pkl", "rb") as f:
            weight_values = pickle.load(f)
        with open(f"{str(path)}_optimizer_config.pkl", "rb") as f:
            conf = pickle.load(f)

        model.optimizer = keras.optimizers.Adam(learning_rate=5e-4).from_config(conf)
        obs = next(iter(ds.batch(2).take(1)))
        model.__train_step__(obs, [0], dummy_train=True)

        # Set the weights of the optimizer
        model.optimizer.set_weights(weight_values)
        return model

    @classmethod
    def restore_no_optimizer(cls, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}.pkl", "rb") as f:
            model = pickle.load(f)
        # with open(f"{str(path)}_optimizer.pkl", "rb") as f:
        #    weight_values = pickle.load(f)
        with open(f"{str(path)}_optimizer_config.pkl", "rb") as f:
            conf = pickle.load(f)

        model.optimizer = keras.optimizers.Adam(learning_rate=5e-4).from_config(conf)
        # obs = next(iter(ds.batch(2).take(1)))
        # model.__train_step__(obs, [0], dummy_train=True)

        # Set the weights of the optimizer
        # model.optimizer.set_weights(weight_values)
        return model


class TINN_Inverse(TINN):
    def __init__(
        self,
        pinn: NN,
        losses,
        norm: Norm,
        no_input_losses=[],
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        alpha=0.5,
        loss_penalty_power=2,
        print_precision=".5f",
        **kwargs,
    ):

        super().__init__(
            pinn, losses, norm, no_input_losses, optimizer, alpha, loss_penalty_power, print_precision, **kwargs
        )
        # Make sure if any loss has trainable paremeters, their
        # Loss_Grad_Type is set to PAREMETER of BOTH
        for loss in losses:
            if loss.loss_grad_type == Loss_Grad_Type.PINN and len(loss.trainables()) > 0:
                raise ValueError(
                    f"The loss {loss.name} has some trainable parameters, while"
                    f" its loss_grad_type is 'PINN' (It should be 'PARAMETER' or 'BOTH')"
                )

    @tf.function
    def __train_step__(self, elements, lambdas_state, dummy_train=False):
        """_summary_

        Args:
            elements (_type_): _description_
            lambdas_state ([int]): [0] == No Lambda
                                   [1] == Update_Lambda start
                                   [2] == Update_Lambda in Middle
                                   [3] == Update_Lambda end
            train_acc_metric (_type_): _description_
            dummy_train (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        flg_parameters_grad = False
        with tf.GradientTape(persistent=True) as tape:

            loss_items = [loss.residual(self.pinn, elements[index]) for index, loss in enumerate(self.losses)]
            loss_items_norm = [self.norm.reduce_norm(item) for item in loss_items]

            # trainables = self.pinn.trainable_variables
            pinn_trainables = self.pinn.trainable_variables
            param_trainables = ()
            regularisable_trainables = self.pinn.trainable_variables
            for loss in self.losses:
                # trainables += loss.trainables()
                param_trainables += loss.trainables()
                if loss.regularise:
                    regularisable_trainables += loss.trainables()
            if len(self.no_input_losses) > 0:
                no_input_loss_items = [loss.residual(self.pinn, None) for loss in self.no_input_losses]
                no_input_loss_items_norm = [self.norm.reduce_norm(item) for item in no_input_loss_items]
                for loss in self.no_input_losses:
                    # trainables += loss.trainables()
                    param_trainables += loss.trainables()
                    if loss.regularise:
                        regularisable_trainables += loss.trainables()
            else:
                no_input_loss_items = []
                no_input_loss_items_norm = 0.0
            #######################
            pinn_regularisable_norms = [
                loss_items_norm[i]
                for i in self.regularisable_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PINN
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ] + [
                no_input_loss_items_norm[i]
                for i in self.regularisable_no_input_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PINN
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ]
            if len(pinn_regularisable_norms) > 0:
                pinn_regularisable_norms_c = tf.concat(pinn_regularisable_norms, axis=0)
                pinn_regularised_loss_value = tf.reduce_sum(tf.stack(self.lambdas) * pinn_regularisable_norms_c)
            else:
                pinn_regularisable_norms = 0.0
                pinn_regularised_loss_value = tf.constant(0.0, dtype=self.pinn.dtype)
            #
            param_regularisable_norms = [
                loss_items_norm[i]
                for i in self.regularisable_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PARAMETER
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ] + [
                no_input_loss_items_norm[i]
                for i in self.regularisable_no_input_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PARAMETER
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ]
            if len(param_regularisable_norms) > 0:
                param_regularisable_norms_c = tf.concat(param_regularisable_norms, axis=0)
                param_regularised_loss_value = tf.reduce_sum(tf.stack(self.lambdas) * param_regularisable_norms_c)
                flg_parameters_grad = True
            else:
                param_regularisable_norms = 0.0
                param_regularised_loss_value = tf.constant(0.0, dtype=self.pinn.dtype)
            ###########################
            pinn_unregularisable_norms = [
                loss_items_norm[i]
                for i in self.unregularisable_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PINN
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ] + [
                no_input_loss_items_norm[i]
                for i in self.unregularisable_no_input_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PINN
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ]
            if len(pinn_unregularisable_norms) > 0:
                pinn_unregularisable_norms = tf.concat(pinn_unregularisable_norms, axis=0)
                pinn_unregularised_loss_value = tf.reduce_sum(pinn_unregularisable_norms)
            else:
                pinn_unregularisable_norms = 0.0
                pinn_unregularised_loss_value = tf.constant(0.0, dtype=self.pinn.dtype)
            #
            param_unregularisable_norms = [
                loss_items_norm[i]
                for i in self.unregularisable_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PARAMETER
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ] + [
                no_input_loss_items_norm[i]
                for i in self.unregularisable_no_input_loss_indices
                if self.losses[i].loss_grad_type == Loss_Grad_Type.PARAMETER
                or self.losses[i].loss_grad_type == Loss_Grad_Type.BOTH
            ]
            if len(param_unregularisable_norms) > 0:
                param_unregularisable_norms = tf.concat(param_unregularisable_norms, axis=0)
                param_unregularised_loss_value = tf.reduce_sum(param_unregularisable_norms)
                flg_parameters_grad = True
            else:
                param_unregularisable_norms = 0.0
                param_unregularised_loss_value = tf.constant(0.0, dtype=self.pinn.dtype)
            ##############################
            pinn_loss_value = pinn_regularised_loss_value + pinn_unregularised_loss_value
            param_loss_value = param_regularised_loss_value + param_unregularised_loss_value
            loss_value = pinn_loss_value + param_loss_value

        if lambdas_state[0] > 0:
            if param_regularisable_norms == 0.0:
                regularisable_norms = tf.concat(pinn_regularisable_norms, axis=0)
            else:
                regularisable_norms = tf.concat(pinn_regularisable_norms + param_regularisable_norms, axis=0)
            self._update_lambdas_(lambdas_state, regularisable_norms, regularisable_trainables)

        pinn_grads = tape.gradient(pinn_loss_value, pinn_trainables)
        if flg_parameters_grad:
            param_grads = tape.gradient(param_loss_value, param_trainables)
        if dummy_train:
            # self.dummy_update_grads(trainables)
            raise NotImplementedError
        else:
            if pinn_regularisable_norms > 0 or pinn_unregularisable_norms > 0:
                self.optimizer.apply_gradients(zip(pinn_grads, pinn_trainables))
            if flg_parameters_grad:
                self.optimizer.apply_gradients(zip(param_grads, param_trainables))

        return loss_value, loss_items_norm, no_input_loss_items_norm


class TINNBackup(tf.Module):
    """Turing-Informed Neural Net"""

    def __init__(
        self,
        pinn: NN,
        losses,
        norm: Norm,
        no_input_losses=[],
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        alpha=0.5,
        loss_penalty_power=2,
        print_precision=".5f",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(losses) == 0
        self.pinn = pinn
        self.losses = losses
        self.no_input_losses = no_input_losses
        # self.extra_loss = extra_loss
        # self.extra_loss_len = len(extra_loss)
        self.norm = norm
        self.optimizer = optimizer
        self.alpha = tf.Variable(alpha, dtype=pinn.dtype, trainable=False)
        self.loss_penalty_power = tf.Variable(loss_penalty_power, dtype=pinn.dtype, trainable=False)
        self.print_precision = print_precision
        self.weight_values = None
        #
        # self.lambda_obs_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        # self.lambda_obs_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        # self.lambda_pde_u = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)
        # self.lambda_pde_v = tf.Variable(1.0, dtype=pinn.dtype, trainable=False)

        # self.grad_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.grad_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.grad_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.grad_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        # self.loss_norm_obs_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.loss_norm_obs_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.loss_norm_pde_u = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)
        # self.loss_norm_pde_v = tf.Variable(0.0, dtype=pinn.dtype, trainable=False)

        self.num_loss = len(losses) + len(no_input_losses)
        self.regularisable_loss_indices = [i for i, loss in enumerate(self.losses) if loss.regularise]
        self.regularisable_no_input_loss_indices = [i for i, loss in enumerate(self.no_input_losses) if loss.regularise]
        self.unregularisable_loss_indices = [i for i, loss in enumerate(self.losses) if loss.regularise is False]
        self.unregularisable_no_input_loss_indices = [
            i for i, loss in enumerate(self.no_input_losses) if loss.regularise is False
        ]

        self.num_regularisers = int(
            np.sum([self.losses[i].residual_ret_num for i in self.regularisable_loss_indices])
            + np.sum([self.losses[i].residual_ret_num for i in self.regularisable_no_input_loss_indices])
        )

        self.lambdas = tf.stack(
            [tf.Variable(1.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]
        )

        self.grad_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]

        self.loss_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(self.num_regularisers)]

        self.__version__ = 0.1

        self._reset_losses_()

    def _reset_losses_(self):
        self.loss_total = 0
        self.loss_reg_total = 0
        # self.loss_obs_u = 0
        # self.loss_obs_v = 0
        # self.loss_pde_u = 0
        # self.loss_pde_v = 0
        # self.loss_extra = np.zeros(self.extra_loss_len)
        self.train_acc = 0
        # self.loss_non_zero = 0
        self.loss_values = np.zeros(self.num_regularisers)

    @tf.function
    def __train_step__(self, elements, lambdas_state, dummy_train=False):
        """_summary_

        Args:
            elements (_type_): _description_
            lambdas_state ([int]): [0] == No Lambda
                                   [1] == Update_Lambda start
                                   [2] == Update_Lambda in Middle
                                   [3] == Update_Lambda end
            train_acc_metric (_type_): _description_
            dummy_train (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        with tf.GradientTape(persistent=True) as tape:

            loss_items = [loss.residual(self.pinn, elements[index]) for index, loss in enumerate(self.losses)]
            loss_items_norm = [self.norm.reduce_norm(item) for item in loss_items]
            trainables = self.pinn.trainable_variables
            for loss in self.losses:
                trainables += loss.trainables()
            if len(self.no_input_losses) > 0:
                no_input_loss_items = [loss.residual(self.pinn, None) for loss in self.no_input_losses]
                no_input_loss_items_norm = [self.norm.reduce_norm(item) for item in no_input_loss_items]
                for loss in self.no_input_loss_items_norm:
                    trainables += loss.trainables()
            else:
                no_input_loss_items = []

            regularisable_norms = tf.concat(
                [loss_items_norm[i] for i in self.regularisable_loss_indices]
                + [no_input_loss_items_norm[i] for i in self.regularisable_no_input_loss_indices],
                axis=0,
            )
            unregularisable_norms = tf.concat(
                [loss_items_norm[i] for i in self.unregularisable_loss_indices]
                + [no_input_loss_items_norm[i] for i in self.unregularisable_no_input_loss_indices],
                axis=0,
            )
            regularised_loss_value = tf.reduce_sum(self.lambdas * regularisable_norms)
            unregularised_loss_value = tf.reduce_sum(unregularisable_norms)
            loss_value = regularised_loss_value + unregularised_loss_value

        if lambdas_state[0] > 0:
            self._update_lambdas_(elements, lambdas_state, tape, regularisable_norms, trainables)

        grads = tape.gradient(loss_value, trainables)
        if dummy_train:
            self.dummy_update_grads(trainables)
        else:
            self.optimizer.apply_gradients(zip(grads, trainables))

        return loss_value, loss_items_norm, no_input_loss_items_norm

    @tf.function
    def __train_step2__(self, elements, lambdas_state, train_acc_metric, dummy_train=False):
        """_summary_

        Args:
            elements (_type_): _description_
            lambdas_state ([int]): [0] == No Lambda
                                   [1] == Update_Lambda start
                                   [2] == Update_Lambda in Middle
                                   [3] == Update_Lambda end
            train_acc_metric (_type_): _description_
            dummy_train (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        elements_len = len(elements)
        if elements_len == 2:
            x_obs, y_obs = elements
            x_pde = None
            extra_elements = None
        elif elements_len == 3:
            x_obs, y_obs, x_pde = elements
            extra_elements = None
        else:
            x_obs = elements[0]
            y_obs = elements[1]
            x_pde = elements[2]
            extra_elements = elements[3:]
        with tf.GradientTape(persistent=True) as tape:
            if x_pde is None:
                outputs, f_u, f_v = self.pde_residual.residual(self.pinn, x_obs)
            else:
                outputs = self.pinn.net(x_obs)
                _, f_u, f_v = self.pde_residual.residual(self.pinn, x_pde)
            loss_obs_u = self.norm.norm(
                y_obs[:, 0] - outputs[:, 0]
            )  # tf.reduce_mean(tf.square(y_obs[:,0]-outputs[:,0]))
            loss_obs_v = self.norm.norm(
                y_obs[:, 1] - outputs[:, 1]
            )  # tf.reduce_mean(tf.square(y_obs[:,1]-outputs[:,1]))
            loss_pde_u = self.norm.norm(f_u)  # tf.reduce_mean(tf.square(f_u))
            loss_pde_v = self.norm.norm(f_v)  # tf.reduce_mean(tf.square(f_v))

            if self.extra_loss_len > 0:
                # [tape.watch(extra_elements[index]) for index, _ in enumerate(self.extra_loss)]
                loss_extra_items = [
                    extra_loss.residual(self.pinn, extra_elements[index])
                    for index, extra_loss in enumerate(self.extra_loss)
                ]
                loss_extra_items = [self.norm.norm(item) for item in loss_extra_items]
                loss_extra = tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_extra = 0.0

            trainables = self.pinn.trainable_variables + self.pde_residual.trainables()
            for extra_loss in self.extra_loss:
                trainables += extra_loss.trainables()

            loss_value = (
                loss_extra
                + self.lambda_obs_u * loss_obs_u
                + self.lambda_obs_v * loss_obs_v
                + self.lambda_pde_u * loss_pde_u
                + self.lambda_pde_v * loss_pde_v
            )

            if self.no_input_losses is not None:
                non_zero_loss_value = self.no_input_losses.residual(self.pinn, None)
            else:
                non_zero_loss_value = 0.0

        if lambdas_state[0] > 0:
            self._update_lambdas_(
                x_pde, lambdas_state, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables
            )

        grads = tape.gradient(loss_value, trainables)
        #  None-zero parameter requlirisation
        if self.no_input_losses is not None:
            grads_non_zero = tape.gradient(non_zero_loss_value, self.no_input_losses.parameters)
        #
        if dummy_train:
            self.dummy_update_grads(trainables)
        else:
            self.optimizer.apply_gradients(zip(grads, trainables))
            if self.no_input_losses is not None:
                self.optimizer.apply_gradients(zip(grads_non_zero, self.no_input_losses.parameters))
            train_acc_metric.update_state(y_obs, outputs)

        return loss_value, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, non_zero_loss_value, loss_extra_items

    def dummy_update_grads(self, trainables):
        # dummy zero gradients
        zero_grads = [tf.zeros_like(w) for w in trainables]
        # save current state of variables
        saved_vars = [tf.identity(w) for w in trainables]
        if self.no_input_losses is not None:
            # dummy zero gradients
            zero_grads_params = [tf.zeros_like(w) for w in self.no_input_losses.parameters]
            # save current state of variables
            saved_vars_params = [tf.identity(w) for w in self.no_input_losses.parameters]
        # Apply gradients which don't do nothing
        self.optimizer.apply_gradients(zip(zero_grads, trainables))
        if self.no_input_losses is not None:
            self.optimizer.apply_gradients(zip(zero_grads_params, self.no_input_losses.parameters))
        # Reload variables
        [x.assign(y) for x, y in zip(trainables, saved_vars)]
        if self.no_input_losses is not None:
            [x.assign(y) for x, y in zip(self.no_input_losses.parameters, saved_vars_params)]

    def _update_lambdas_(self, x_pde, lambdas_state, tape, loss_obs_u, loss_obs_v, loss_pde_u, loss_pde_v, trainables):
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

        if lambdas_state[0] == 1:
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

        if lambdas_state[0] == 3:
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
        dataset: TINN_Dataset,
        print_interval=10,
        stop_threshold=0,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        regularise_interval=1,
        train_acc_metric=keras.metrics.MeanSquaredError(),
        printer=default_printer,
        epoch_callback=None,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()
        # indeces_list = list(indices(batch_size, shuffle, x1_size, x2_size))
        # x_size = dataset.x_size
        # x_pde_size = dataset.x_pde_size
        x_sizes = np.array(dataset.sizes)
        #
        dataset2 = dataset.cache().batch(batch_size)
        for epoch in range(epochs):
            if epoch % print_interval == 0:
                printer(f"\nStart of epoch {epoch:d}")

            step = 0
            # Iterate over the batches of the dataset.
            for element in dataset2:
                # if dataset.has_x_pde:
                #    x_batch_train, y_batch_train, p_batch_train = element
                # else:
                #    x_batch_train, y_batch_train = element
                #    p_batch_train = None
                if regularise and epoch % regularise_interval == 0:
                    if step == 0:
                        lambdas_state = [1]
                    elif step == last_step:
                        lambdas_state = [3]
                    else:
                        lambdas_state = [2]
                else:
                    lambdas_state = [0]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_non_zero_batch,
                    loss_extra_batch,
                ) = self.__train_step__(element, lambdas_state)
                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                # current_batch_size = x_batch_train.shape[0]
                # current_pde_batch_size = p_batch_train.shape[0] if dataset.has_x_pde else current_batch_size
                # w_obs = current_batch_size / x_size
                # w_pde = current_pde_batch_size / x_pde_size  # if p_batch_train is None else batch_size / x2_size
                ws = np.array([len(item) for item in element]) / x_sizes
                w_obs = ws[0]
                w_pde = w_obs if len(ws) < 3 else ws[2]

                self.loss_reg_total += loss_value_batch
                self.loss_obs_u += loss_obs_u_batch * w_obs
                self.loss_obs_v += loss_obs_v_batch * w_obs
                self.loss_pde_u += loss_pde_u_batch * w_pde
                self.loss_pde_v += loss_pde_v_batch * w_pde
                # total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                total_loss_extra_batch = np.array([item.numpy() for item in loss_extra_batch])
                self.loss_non_zero += loss_non_zero_batch
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    # + np.sum(total_loss_extra_batch) * w_obs
                )
                step += 1
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = train_acc_metric.result()
            self.loss_non_zero = self.loss_non_zero / (last_step + 1)
            # update the samples
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            if epoch_callback is not None:
                epoch_callback(epoch, samples, self)
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_(printer)
            if stop_threshold >= float(self.train_acc):
                printer("############################################")
                printer("#               Early stop                 #")
                printer("############################################")
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                printer(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples

    def train2(
        self,
        epochs,
        batch_size,
        dataset: TINN_Dataset,
        print_interval=10,
        stop_threshold=0,
        sample_losses=True,
        sample_parameters=True,
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
        regularise_interval=1,
        train_acc_metric=keras.metrics.MeanSquaredError(),
        printer=default_printer,
        epoch_callback=None,
    ):

        # Samplling arrays
        samples = self._create_samples_(
            epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters
        )
        #
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        start_time = time.time()
        # indeces_list = list(indices(batch_size, shuffle, x1_size, x2_size))
        # x_size = dataset.x_size
        # x_pde_size = dataset.x_pde_size
        x_sizes = np.array(dataset.sizes)
        #
        dataset2 = dataset.cache().batch(batch_size)
        for epoch in range(epochs):
            if epoch % print_interval == 0:
                printer(f"\nStart of epoch {epoch:d}")

            step = 0
            # Iterate over the batches of the dataset.
            for element in dataset2:
                # if dataset.has_x_pde:
                #    x_batch_train, y_batch_train, p_batch_train = element
                # else:
                #    x_batch_train, y_batch_train = element
                #    p_batch_train = None
                if regularise and epoch % regularise_interval == 0:
                    if step == 0:
                        lambdas_state = [1]
                    elif step == last_step:
                        lambdas_state = [3]
                    else:
                        lambdas_state = [2]
                else:
                    lambdas_state = [0]

                (
                    loss_value_batch,
                    loss_obs_u_batch,
                    loss_obs_v_batch,
                    loss_pde_u_batch,
                    loss_pde_v_batch,
                    loss_non_zero_batch,
                    loss_extra_batch,
                ) = self.__train_step__(
                    element,
                    lambdas_state,
                    train_acc_metric,
                )
                if step > last_step:
                    last_step = step
                # add the batch loss: Note that the weights are calculated based on the batch size
                #                     especifically, the last batch can have a different size
                # current_batch_size = x_batch_train.shape[0]
                # current_pde_batch_size = p_batch_train.shape[0] if dataset.has_x_pde else current_batch_size
                # w_obs = current_batch_size / x_size
                # w_pde = current_pde_batch_size / x_pde_size  # if p_batch_train is None else batch_size / x2_size
                ws = np.array([len(item) for item in element]) / x_sizes
                w_obs = ws[0]
                w_pde = w_obs if len(ws) < 3 else ws[2]

                self.loss_reg_total += loss_value_batch
                self.loss_obs_u += loss_obs_u_batch * w_obs
                self.loss_obs_v += loss_obs_v_batch * w_obs
                self.loss_pde_u += loss_pde_u_batch * w_pde
                self.loss_pde_v += loss_pde_v_batch * w_pde
                # total_loss_extra_batch = np.sum([item.numpy() for item in loss_extra_batch])
                total_loss_extra_batch = np.array([item.numpy() for item in loss_extra_batch])
                self.loss_non_zero += loss_non_zero_batch
                self.loss_extra += total_loss_extra_batch * w_obs
                self.loss_total += (
                    loss_obs_u_batch * w_obs
                    + loss_obs_v_batch * w_obs
                    + loss_pde_u_batch * w_pde
                    + loss_pde_v_batch * w_pde
                    # + np.sum(total_loss_extra_batch) * w_obs
                )
                step += 1
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            self.train_acc = train_acc_metric.result()
            self.loss_non_zero = self.loss_non_zero / (last_step + 1)
            # update the samples
            self._store_samples_(
                samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
            )
            if epoch_callback is not None:
                epoch_callback(epoch, samples, self)
            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                self._print_metrics_(printer)
            if stop_threshold >= float(self.train_acc):
                printer("############################################")
                printer("#               Early stop                 #")
                printer("############################################")
                return samples
            # end for epoch in range(epochs)
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            self._reset_losses_()
            if epoch % print_interval == 0:
                printer(f"Time taken: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        return samples

    def train3(
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
                    loss_non_zero_batch,
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
                self.loss_non_zero += loss_non_zero_batch
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
            self.loss_non_zero = self.loss_non_zero / (last_step + 1)
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
        if self.no_input_losses is not None:
            ret = {
                **ret,
                **{"loss_non_zero": np.zeros(epochs)},
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
        if self.no_input_losses is not None:
            samples["loss_non_zero"][epoch] = self.loss_non_zero
        if sample_parameters:
            for param in self.pde_residual.trainables():
                samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()

    def _print_metrics_(self, printer=default_printer):
        printer(f"Training observations acc over epoch: {self.train_acc:{self.print_precision}}")
        printer(
            f"total loss: {self.loss_total:{self.print_precision}}, "
            f"total regularisd loss (sum of batches): {self.loss_reg_total:{self.print_precision}}"
        )
        printer(
            f"obs u loss: {self.loss_obs_u:{self.print_precision}}, "
            f"obs v loss: {self.loss_obs_v:{self.print_precision}}"
        )
        printer(
            f"pde u loss: {self.loss_pde_u:{self.print_precision}}, "
            f"pde v loss: {self.loss_pde_v:{self.print_precision}}"
        )
        if self.no_input_losses is not None:
            printer(f"Non-zero loss: {self.loss_non_zero:{self.print_precision}}, ")
        printer(
            f"lambda obs u: {self.lambda_obs_u.numpy():{self.print_precision}}, "
            f"lambda obs v: {self.lambda_obs_v.numpy():{self.print_precision}}"
        )
        printer(
            f"lambda pde u: {self.lambda_pde_u.numpy():{self.print_precision}}, "
            f"lambda pde v: {self.lambda_pde_v.numpy():{self.print_precision}}"
        )
        printer(self.pde_residual.trainables_str())
        if self.extra_loss_len > 0:
            for i, loss in enumerate(self.extra_loss):
                printer(f"extra loss {loss.name}: {self.loss_extra[i]:{self.print_precision}}")

    def save(self, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)

        # save optimizer state
        weight_values = self.optimizer.get_weights()  # tf.keras.backend.batch_get_value(symbolic_weights)
        with open(f"{str(path)}_optimizer.pkl", "wb") as f:
            pickle.dump(weight_values, f)
        # remove optimiser
        opt = self.optimizer
        # self.optimizer = None
        delattr(self, "optimizer")
        with open(f"{str(path)}.pkl", "wb") as f:
            pickle.dump(self, f)
        # Restore optimizer
        # self.optimizer = opt
        setattr(self, "optimizer", opt)

        # save optimizer config
        conf = self.optimizer.get_config()
        with open(f"{str(path)}_optimizer_config.pkl", "wb") as f:
            pickle.dump(conf, f)

    @classmethod
    def restore(cls, path_dir, name, ds):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{str(path)}_optimizer.pkl", "rb") as f:
            weight_values = pickle.load(f)
        with open(f"{str(path)}_optimizer_config.pkl", "rb") as f:
            conf = pickle.load(f)

        model.optimizer = keras.optimizers.Adam(learning_rate=5e-4).from_config(conf)
        obs = next(iter(ds.batch(2).take(1)))
        model.__train_step__(obs, [0], None, dummy_train=True)

        # Set the weights of the optimizer
        model.optimizer.set_weights(weight_values)
        return model


class NN_Compound(NN):
    def __init__(self, pre_pinn, lb, ub, pre_lb, pre_ub, layers, skips=None, dtype=tf.float32, **kwargs):
        assert pre_pinn.layers[-1] == layers[0]
        if skips is not None:
            assert len(skips) == len(layers)
        self.pre_Ws = [w.numpy() for w in pre_pinn.Ws]
        self.pre_bs = [b.numpy() for b in pre_pinn.bs]
        self.pre_lb = pre_lb
        self.pre_ub = pre_ub
        self.skips = skips
        self.swtich_index = -1
        super().__init__(layers, lb, ub, dtype, **kwargs)
        self.lb = tf.expand_dims(tf.Variable(self.lb, dtype=dtype), axis=0)
        self.ub = tf.expand_dims(tf.Variable(self.ub, dtype=dtype), axis=0)
        self.__version__ = 0.1

    def build(self):
        """Create the state of the layers (weights)"""
        weights = []
        biases = []
        for pre_W, pre_b in zip(self.pre_Ws, self.pre_bs):
            W = tf.Variable(pre_W, trainable=False, dtype=self.dtype)
            b = tf.Variable(pre_b, trainable=False, dtype=self.dtype)
            weights.append(W)
            biases.append(b)
            self.swtich_index += 1

        for i in range(0, self.num_layers - 1):
            W = self.xavier_init(size=[self.layers[i], self.layers[i + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)

        self.Ws = weights
        self.bs = biases

    @tf.function
    def net(self, inputs):
        # inputs_2 = self.pre_pinn(inputs)
        # Map the inputs to the range [-1, 1]
        H = 2.0 * (inputs - self.pre_lb) / (self.pre_ub - self.pre_lb) - 1.0
        for i, (W, b) in enumerate(zip(self.Ws[:-1], self.bs[:-1])):
            if i == self.swtich_index:
                H = tf.add(tf.matmul(H, W), b)
                H_skip = H
                H = 2.0 * (H - self.lb) / (self.ub - self.lb) - 1.0
            else:
                H = tf.tanh(tf.add(tf.matmul(H, W), b))

            if i > self.swtich_index and self.skips[i - self.swtich_index] == 1:
                H += H_skip
                H_skip = H
                print(f" to layer{i- self.swtich_index}")

            if i > self.swtich_index and self.skips[i - self.swtich_index] == 2:
                print(f" from layer{i- self.swtich_index}")
                H_skip = H

        W = self.Ws[-1]
        b = self.bs[-1]
        outputs = tf.add(tf.matmul(H, W), b)
        if self.skips[-1] == 1:
            outputs += H_skip
            print(f" to layer{len(self.skips)-1}")
        return outputs
