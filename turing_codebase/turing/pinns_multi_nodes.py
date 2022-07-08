import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from .utils import indices
from .pinns import NN, Loss


class TINN_multi_nodes():
    """Turing-Informed Neural Net"""

    def __init__(self,
                 pinn: NN,
                 pde_loss: Loss,
                 extra_loss=[],
                 nodes_n=2,
                 node_names=None,
                 optimizer=keras.optimizers.Adam(),
                 train_acc_metric=keras.metrics.MeanSquaredError(),
                 alpha=0.5,
                 print_precision=".5f"):
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
        self.alpha = alpha
        self.print_precision = print_precision
        #
        self.lambdas = [tf.Variable(
            1., dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

        self.grad_norms = [tf.Variable(
            0., dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

        self.loss_norms = [tf.Variable(
            0., dtype=pinn.dtype, trainable=False) for i in range(nodes_n * 2)]

    @tf.function
    def __train_step__(self, x, y, update_lambdas, first_step, last_step):
        with tf.GradientTape(persistent=False) as tape:
            outputs, f_pde = self.pde_loss.loss(self.pinn, x)

            loss_obs = tf.reduce_mean(
                tf.math.squared_difference(y, outputs), axis=0)
            loss_pde = tf.reduce_mean(tf.square(f_pde), axis=0)
            loss_items = tf.concat([loss_obs, loss_pde], axis=0)

            trainables = self.pinn.trainable_variables + self.pde_loss.trainables()
            if self.extra_loss_len > 0:
                for extra_loss in self.extra_loss:
                    trainables += extra_loss.trainables()
                loss_extra_items = [extra_loss.loss(
                    self.pinn, x) for extra_loss in self.extra_loss]
                loss_value = tf.reduce_sum(
                    tf.stack(self.lambdas) * loss_items) + tf.reduce_sum(loss_extra_items)
            else:
                loss_extra_items = []
                loss_value = tf.reduce_sum(tf.stack(self.lambdas) * loss_items)

        if update_lambdas:
            grads = [tf.gradients(loss_items[i], trainables)
                     for i in range(loss_items.shape[0])]

            reduced_grads = tf.stack([
                tf.reduce_sum([tf.reduce_sum(tf.square(item))
                              for item in grad_i])
                for grad_i in grads])

            if first_step:
                for i in range(self.nodes_n * 2):
                    self.grad_norms[i].assign(reduced_grads[i])

                    self.loss_norms[i].assign(loss_items[i])

            else:
                for i in range(self.nodes_n * 2):
                    self.grad_norms[i].assign(
                        self.grad_norms[i] + reduced_grads[i])

                    self.loss_norms[i].assign(
                        self.loss_norms[i] + loss_items[i])

            if last_step:
                Ws = tf.square(self.loss_norms) / tf.sqrt(self.grad_norms)

                w_total = tf.reduce_sum(Ws)
                for i in range(self.nodes_n * 2):
                    self.lambdas[i].assign(
                        self.alpha * self.lambdas[i] + (1 - self.alpha) * 4.0 * Ws[i] / w_total)

        grads = tape.gradient(loss_value, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))
        self.train_acc_metric.update_state(y, outputs)
        return loss_value, loss_obs, loss_pde, loss_extra_items

    def train(self,
              epochs,
              batch_size,
              X,
              Y,
              print_interval=10,
              stop_threshold=0,
              shuffle=True,
              sample_losses=True,
              sample_regularisations=True,
              sample_gradients=False):

        # Samplling arrays
        if sample_losses:
            arr_loss_total = np.zeros(epochs)
            arr_loss_regularisd_total = np.zeros(epochs)
            arr_obs_acc = np.zeros(epochs)
            arr_loss_obs = np.zeros((epochs, self.nodes_n))
            arr_loss_pde = np.zeros((epochs, self.nodes_n))
            if self.extra_loss_len > 0:
                arr_loss_extra = np.zeros((epochs, self.extra_loss_len))
        if sample_regularisations:
            arr_lambda_obs = np.zeros((epochs, self.nodes_n))
            arr_lambda_pde = np.zeros((epochs, self.nodes_n))
        if sample_gradients:
            arr_grads_obs = np.zeros((epochs, self.nodes_n))
            arr_grads_pde = np.zeros((epochs, self.nodes_n))

        #
        def fill_return():
            ret = {'training_obs_accuracy': arr_obs_acc}
            if sample_losses:
                ret = {**ret,
                       **{'loss_total': arr_loss_total,
                          'loss_regularisd_total': arr_loss_regularisd_total}
                       }
                for i, name in enumerate(self.node_names):
                    ret[f"loss_obs_{name}"] = arr_loss_obs[:, i]
                    ret[f"loss_pde_{name}"] = arr_loss_pde[:, i]
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        ret[f"loss_extra_{loss.name}"] = arr_loss_extra[:, i]

            if sample_regularisations:
                for i, name in enumerate(self.node_names):
                    ret[f"lambda_obs_{name}"] = arr_lambda_obs[:, i]
                    ret[f"lambda_pde_{name}"] = arr_lambda_pde[:, i]
            if sample_gradients:
                for i, name in enumerate(self.node_names):
                    ret[f"grads_obs_{name}"] = arr_grads_obs[:, i]
                    ret[f"grads_pde_{name}"] = arr_grads_pde[:, i]
            return ret
        #
        X_size = len(X)
        # For first epoch, we store the number of steps to compelete a full epoch
        last_step = -1
        last_lambda_obs = np.ones(self.nodes_n)
        last_lambda_pde = np.ones(self.nodes_n)
        start_time = time.time()

        for epoch in range(epochs):
            if epoch % print_interval == 0:
                print(f"\nStart of epoch {epoch:d}")

            loss_total, loss_reg_total = 0, 0
            loss_obs = np.zeros(self.nodes_n)
            loss_pde = np.zeros(self.nodes_n)
            loss_extra = np.zeros(self.extra_loss_len)
            # Iterate over the batches of the dataset.
            for step, o_batch_indices in enumerate(indices(batch_size, shuffle, X_size)):
                x_batch_train, y_batch_train = X[o_batch_indices], Y[o_batch_indices]

                loss_value_batch, loss_obs_batch, loss_pde_batch, loss_extra_batch = \
                    self.__train_step__(x_batch_train,
                                        y_batch_train,
                                        True,
                                        step == 0,
                                        step == last_step)

                if step > last_step:
                    last_step = step

                loss_reg_total += loss_value_batch.numpy()
                loss_obs += loss_obs_batch.numpy()
                loss_pde += loss_pde_batch.numpy()
                total_loss_extra_batch = np.sum([item.numpy()
                                                 for item in loss_extra_batch])
                loss_extra += total_loss_extra_batch
                loss_total += np.sum(loss_obs_batch.numpy()) + np.sum(loss_pde_batch.numpy()) +\
                    total_loss_extra_batch
            # end of for step, o_batch_indices in enumerate(indice(batch_size, shuffle, X_size))
            train_acc = self.train_acc_metric.result()
            arr_obs_acc[epoch] = train_acc
            if sample_losses:
                arr_loss_total[epoch] = loss_total
                arr_loss_regularisd_total[epoch] = loss_reg_total
                arr_loss_obs[epoch, :] = loss_obs
                arr_loss_pde[epoch, :] = loss_pde

            last_lambda_obs = np.array([item.numpy()
                                       for item in self.lambdas[:self.nodes_n]])
            last_lambda_pde = np.array([item.numpy()
                                       for item in self.lambdas[self.nodes_n:]])

            if sample_regularisations:
                arr_lambda_obs[epoch, :] = last_lambda_obs
                arr_lambda_pde[epoch, :] = last_lambda_pde
            if sample_gradients:
                arr_grads_obs[epoch, :] = [
                    np.sqrt(item.numpy()) for item in self.grad_norms[:self.nodes_n]]
                arr_grads_pde[epoch, :] = [
                    np.sqrt(item.numpy()) for item in self.grad_norms[self.nodes_n:]]

            # Display metrics at the end of each epoch.
            if epoch % print_interval == 0:
                print(
                    f"Training observations acc over epoch: {train_acc:{self.print_precision}}")
                print(f"total loss: {loss_total:{self.print_precision}}, "
                      f"total regularisd loss: {loss_reg_total:{self.print_precision}}")
                for i, name in enumerate(self.node_names):
                    print(f"obs {name} loss: {loss_obs[i]:{self.print_precision}}, "
                          f"pde {name} loss: {loss_pde[i]:{self.print_precision}}")
                for i, name in enumerate(self.node_names):
                    print(f"lambda obs {name}: {last_lambda_obs[i]:{self.print_precision}}, "
                          f"lambda pde {name}: {last_lambda_pde[i]:{self.print_precision}}")

                print(self.pde_loss.trainables_str())
                if self.extra_loss_len > 0:
                    for i, loss in enumerate(self.extra_loss):
                        print(
                            f"extra loss {loss.name}: {loss_extra[i]:{self.print_precision}}")

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
