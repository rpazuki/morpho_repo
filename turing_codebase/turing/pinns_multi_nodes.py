import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .utils import indices
from .pinns import NN, Loss


class TINN_multi_nodes:
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
        print_precision=".5f",
    ):
        self.pinn = pinn
        self.extra_loss = extra_loss
        self.extra_loss_len = len(extra_loss)
        self.pde_loss = pde_loss
        self.nodes_n = nodes_n
        if node_names is None:
            self.node_names = [f"node_{i+1}" for i in range(nodes_n)]
        else:
            self.node_names = node_names

        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.alpha = tf.Variable(alpha, dtype=pinn.dtype, trainable=False)
        self.print_precision = print_precision
        #
        self.lambdas = [tf.Variable(1.0, dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

        self.grad_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

        self.loss_norms = [tf.Variable(0.0, dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

        self._reset_losses_()

    def _reset_losses_(self):
        self.loss_total = 0
        self.loss_reg_total = 0
        self.loss_obs = np.zeros(self.nodes_n)
        self.loss_pde = np.zeros(self.nodes_n)
        self.loss_extra = np.zeros(self.extra_loss_len)
        self.train_acc = 0

    @tf.function
    def __train_step__(self, x_obs, y_obs, x_pde, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=False) as tape:

            if x_pde is None:
                outputs, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_obs)
            else:
                outputs = self.pinn(x_obs)
                _, f_pde = self.pde_loss.loss_multi_nodes(self.pinn, x_pde)

            loss_obs = tf.reduce_mean(tf.math.squared_difference(y_obs, outputs), axis=0)
            loss_pde = tf.reduce_mean(tf.square(f_pde), axis=1)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_loss.trainables()
            if self.extra_loss_len > 0:
                loss_extra_items = [extra_loss.loss(self.pinn, x_obs) for extra_loss in self.extra_loss]
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
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
                self.grad_norms[i].assign(self.grad_norms[i] + reduced_grads[i])

                self.loss_norms[i].assign(self.loss_norms[i] + loss_items[i])

        if last_step:
            Ws = tf.square(self.loss_norms) / tf.sqrt(self.grad_norms)

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
                if X_pde is None:
                    p_batch_train = None
                else:
                    p_batch_train = X_pde[p_batch_indices]
                loss_value_batch, loss_obs_batch, loss_pde_batch, loss_extra_batch = self.__train_step__(
                    x_batch_train, y_batch_train, p_batch_train, regularise, step == 0, step == last_step
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

    def _create_samples_(self, epochs, sample_losses, sample_regularisations, sample_gradients, sample_parameters):
        # Samplling arrays
        ret = {"training_obs_accuracy": np.zeros(epochs)}
        if sample_losses:
            ret = {
                **ret,
                **{
                    "loss_total": np.zeros(epochs),
                    "loss_regularisd_total": np.zeros(epochs),
                    "loss_obs": np.zeros((epochs, self.nodes_n)),
                    "loss_pde": np.zeros((epochs, self.nodes_n)),
                },
            }
            if self.extra_loss_len > 0:
                for i, loss in enumerate(self.extra_loss):
                    ret[f"loss_extra_{loss.name}"] = np.zeros(epochs)
        if sample_regularisations:
            ret = {
                **ret,
                **{
                    "lambda_obs": np.zeros((epochs, self.nodes_n)),
                    "lambda_pde": np.zeros((epochs, self.nodes_n)),
                },
            }
        if sample_gradients:
            ret = {
                **ret,
                **{
                    "grads_obs": np.zeros((epochs, self.nodes_n)),
                    "grads_pde": np.zeros((epochs, self.nodes_n)),
                },
            }
        if sample_parameters:
            for param in self.pde_loss.trainables():
                ret[f"{param.name.split(':')[0]}"] = np.zeros(epochs)
        return ret

    def _store_samples_(
        self, samples, epoch, sample_losses, sample_regularisations, sample_gradients, sample_parameters
    ):
        samples["training_obs_accuracy"][epoch] = self.train_acc
        if sample_losses:
            samples["loss_total"][epoch] = self.loss_total
            samples["loss_regularisd_total"][epoch] = self.loss_reg_total
            samples["loss_obs"][epoch, :] = self.loss_obs
            samples["loss_pde"][epoch, :] = self.loss_pde
            if self.extra_loss_len > 0:
                for i, loss in enumerate(self.extra_loss):
                    samples[f"loss_extra_{loss.name}"][epoch] = self.loss_extra[i]

        if sample_regularisations:
            samples["lambda_obs"][epoch, :] = np.array([item.numpy() for item in self.lambdas[: self.nodes_n]])
            samples["lambda_pde"][epoch, :] = np.array([item.numpy() for item in self.lambdas[self.nodes_n :]])
        if sample_gradients:
            samples["grads_obs"][epoch, :] = [np.sqrt(item.numpy()) for item in self.grad_norms[: self.nodes_n]]
            samples["grads_pde"][epoch, :] = [np.sqrt(item.numpy()) for item in self.grad_norms[self.nodes_n :]]

        if sample_parameters:
            for param in self.pde_loss.trainables():
                samples[f"{param.name.split(':')[0]}"][epoch] = param.numpy()

    def _print_metrics_(self):
        print(f"Training observations acc over epoch: {self.train_acc:{self.print_precision}}")
        print(
            f"total loss: {self.loss_total:{self.print_precision}}, "
            f"total regularisd loss (sum of batches): {self.loss_reg_total:{self.print_precision}}"
        )
        for i, name in enumerate(self.node_names):
            print(
                f"obs {name} loss: {self.loss_obs[i]:{self.print_precision}}, "
                f"pde {name} loss: {self.loss_pde[i]:{self.print_precision}}"
            )
        last_lambda_obs = np.array([item.numpy() for item in self.lambdas[: self.nodes_n]])
        last_lambda_pde = np.array([item.numpy() for item in self.lambdas[self.nodes_n :]])
        for i, name in enumerate(self.node_names):
            print(
                f"lambda obs {name}: {last_lambda_obs[i]:{self.print_precision}}, "
                f"lambda pde {name}: {last_lambda_pde[i]:{self.print_precision}}"
            )

        print(self.pde_loss.trainables_str())
        if self.extra_loss_len > 0:
            for i, loss in enumerate(self.extra_loss):
                print(f"extra loss {loss.name}: {self.loss_extra[i]:{self.print_precision}}")
