import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .pinns import NN
from .pinns import Loss
from .pinns import TINN
from .pinns_multi_nodes import TINN_multi_nodes
from .utils import indices


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
                outputs, f_u, f_v = self.pde_loss.loss(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_u, f_v = self.pde_loss.loss(self.pinn, x_pde)
            loss_obs_u = tf.reduce_mean(tf.square(y_obs[:, 0] - outputs[:, 0]))
            loss_obs_v = tf.reduce_mean(tf.square(y_obs[:, 1] - outputs[:, 1]))
            # Mask the PDE residuals
            pde_mask = tf.expand_dims(mask, 1)
            loss_pde_u = tf.reduce_mean(tf.square(f_u * pde_mask))
            loss_pde_v = tf.reduce_mean(tf.square(f_v * pde_mask))
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
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(epochs, sample_losses, sample_regularisations, sample_gradients)
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
            self._store_samples_(samples, epoch, sample_losses, sample_regularisations, sample_gradients)
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
                outputs, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_pde)

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs, outputs), axis=0)
            # Mask the PDE residuals
            pde_mask = tf.expand_dims(mask, 0)
            loss_pde = tf.reduce_mean(tf.square(f_pde * pde_mask), axis=1)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_loss.trainables()
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
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(epochs, sample_losses, sample_regularisations, sample_gradients)
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

                if np.any(np.isnan(loss_value_batch.numpy())):
                    print(f"Nan values in loss_value_batch. Epoch: {epoch}, step:{step}")
                    return (
                        samples,
                        loss_value_batch,
                        loss_obs_batch,
                        loss_pde_batch,
                        x_batch_train,
                        y_batch_train,
                        p_batch_train,
                        domain_mask_train,
                    )

                if np.any(np.isnan(loss_obs_batch.numpy())):
                    print(f"Nan values in loss_obs_batch. Epoch: {epoch}, step:{step}")
                    return (
                        samples,
                        loss_value_batch,
                        loss_obs_batch,
                        loss_pde_batch,
                        x_batch_train,
                        y_batch_train,
                        p_batch_train,
                        domain_mask_train,
                    )

                if np.any(np.isnan(loss_pde_batch.numpy())):
                    print(f"Nan values in loss_pde_batch. Epoch: {epoch}, step:{step}")
                    return (
                        samples,
                        loss_value_batch,
                        loss_obs_batch,
                        loss_pde_batch,
                        x_batch_train,
                        y_batch_train,
                        p_batch_train,
                        domain_mask_train,
                    )

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
            self._store_samples_(samples, epoch, sample_losses, sample_regularisations, sample_gradients)
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
                outputs, f_u, f_v = self.pde_loss.loss(self.pinn, x_obs_masked)
            else:
                outputs = self.pinn(x_obs_masked)
                x_pde_masked = tf.multiply(x_pde, tf.expand_dims(pde_mask, 1))
                _, f_u, f_v = self.pde_loss.loss(self.pinn, x_pde_masked)
            loss_obs_u = tf.reduce_mean(tf.square(y_obs[:, 0] * x_mask - outputs[:, 0]))
            loss_obs_v = tf.reduce_mean(tf.square(y_obs[:, 1] * x_mask - outputs[:, 1]))
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
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(epochs, sample_losses, sample_regularisations, sample_gradients)
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
            self._store_samples_(samples, epoch, sample_losses, sample_regularisations, sample_gradients)
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
                outputs, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_obs_masked)
            else:
                outputs = self.pinn(x_obs_masked)
                x_pde_masked = tf.multiply(x_pde, tf.expand_dims(pde_mask, 1))
                _, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_pde_masked)

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs * x_mask_expand, outputs), axis=0)
            loss_pde = tf.reduce_mean(tf.square(f_pde), axis=0)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_loss.trainables()
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
        sample_regularisations=True,
        sample_gradients=False,
        regularise=True,
    ):

        # Samplling arrays
        samples = self._create_samples_(epochs, sample_losses, sample_regularisations, sample_gradients)
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
            self._store_samples_(samples, epoch, sample_losses, sample_regularisations, sample_gradients)
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
