import time
import gc
import copy
import tensorflow as tf
from tensorflow import keras
import numpy as np


class NN(tf.Module):

    def __init__(self, layers, lb, ub, **kwargs):
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
        self.build()

    def build(self):
        """Create the state of the layers (weights)"""
        weights = []
        biases = []
        for l in range(0, self.num_layers - 1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        self.Ws = weights
        self.bs = biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim],
                                                         stddev=xavier_stddev,
                                                         dtype=tf.float32),
                           dtype=tf.float32)


    @tf.function
    def __net__(self, inputs):
        # Map the inputs to the range [-1, 1]
        H = 2.0*(inputs - self.lb)/(self.ub - self.lb) - 1.0
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = self.Ws[-1]
        b = self.bs[-1]
        outputs = tf.add(tf.matmul(H, W), b)
        return outputs

    def __call__(self, inputs, grads=True):
        """Defines the computation from inputs to outputs

        Args:
           inputs: A tensor that has a shape [None, D1], where
                   D1 is the input dimensionality, specified in
                   the first element of layes.
           grads:  Default 'True'. Returns the first and second
                   order gradients of the output with respect to
                   the input when the grads argument is 'True'.

        Return:
                A tensor of the dense layer output that has a shape
                [None, Dn], where Dn is the dimensionality of the last
                layer, specificed by the last elements of the layer
                arguemnt.
                When 'grads=True', the list of first and second order gradients
                of the output with respect to the input.

        The returns 'partial_1' and 'partial_2' are the first and second
        order gradients, repsectivly. Each one is a list that its elements
        corresponds to on of the NN's last layer output. e.g. if the last layer
        has Dn outputs, each list has Dn tensors as an elements. The dimensionality
        of the tensors are the same as inputs: [None, D1]

        """
        X = tf.cast(inputs, tf.float32)
        outputs = self.__net__(X)
        if grads:
            partials_1 = [tf.gradients(outputs[:, i], X)[0] for i in range(outputs.shape[1])]
            partials_2 = [tf.gradients(partials_1[i], X)[0] for i in range(outputs.shape[1])]
            return outputs, partials_1, partials_2
        else:
            return outputs

    def copy(self):
        return copy.deepcopy(self)


class Loss():
    def __init__(self, pinn, name, data_size=0, init_loss_weight=1.0):
        """Loss value that is calulated for the output of the pinn

        Args:
            pinn: NN object that will be trained.
            name: The name of the Loss
            data_size: The length of its internal dataset.
            init_loss_weight: Initial weigth of the loss in comparision to others.


            When the Loss class has some internal data (e.g. inputs),
            it must provid the length of its data_size, since that value
            will be used to create randomly shuffled indices on btach training
            time.


        """
        self.pinn = pinn
        self.data_size = data_size
        self.name = name
        self.loss_weight = tf.Variable(init_loss_weight, trainable=False, name=f"{name}__loss_weight")


    def batch(self, indices):
        """Returns a batch that will be proccessed in loss method

        Args:
           indices: Randomly shuffled indices for the current batch.

           Each loss class is responsible for its data. However, the
           batch indices are provided by TINN class. So, this method
           slices the data that it will use in loss calculation (loss method)
           Note that whatever returns fromthis method will be the batch argument
           of the loss method, which is casted as Tensor by tensorflow.

           Example:
             return (self.input_1[indices], self.input_2[indices])

        """
        pass

    #  @tf.function
    def loss(self, batch):
        """A tensorflow function that calculates and returns the loss

        Args:
           batch: This is the value(s) that is returned from batch method.
                  Note that the values are converted to Tensors by Tensorflow

           It must return the loss values

           Example:
              input_1, input_2 = batch
              ouput_1 = self.pinn(input_1)
              ouput_2 = self.pinn(input_2)
              L = tf.reduce_sum(tf.square(output_1 - output_1), name=self.name)
              self.L = L
              return L

        """
        pass

    #def loss_weight(self, iteration):
    #    """Return weigth of the loss in comparision to others
    #
    #    Args:
    #        iteration: the iteration index ( for hyper-parametrs that update
    #                   during the training)
    #    """
    #    return self.loss_weight

    def trainable_vars(self):
        """Retruns a list of Tensorflow variables for training

           If the loss class has some tarinable variables, it can
           return them as a list. These variables will be updated
           by optimiser, if they are already part of the computation
           graph.

        """
        return []

    def trainable_vars_str(self):
        s = ""
        t_vars = self.trainable_vars()
        if len(t_vars) > 0:
            s +=  ", ".join([ f"{v.name}:{self.__get_val__(v):.8f}" for v in t_vars])
        return s

    def __get_val__(self, item):
        val = item.numpy()
        if type(val) is float:
            return val
        else:
            return val[0]

class TINN():
    """Turing-Informed Neural Net"""
    def __init__(self,
                 pinn: NN,
                 losses: Loss,
                 #optimizer = keras.optimizers.SGD(learning_rate=1e-6,
                 #                                 momentum=0.0,
                 #                                 nesterov=False)
                 optimizer=keras.optimizers.SGD(learning_rate=1e-6,
                                                momentum=0.0,
                                                nesterov=False),
                 fixed_pinn=False,
                 fixed_loss_params=False):
        self.pinn = pinn
        self.losses = losses
        # self.optimizer = keras.optimizers.Adam(epsilon=1.0)
        self.optimizer = optimizer
        self.fixed_pinn = fixed_pinn
        self.fixed_loss_params = fixed_loss_params
        self.__trainable_vars_tuple__ = None
        self.warmed_up = False

    def trainable_vars(self):
        if self.__trainable_vars_tuple__ is None:
            self.__trainable_vars_tuple__ = ()
            if self.fixed_pinn == False:
                self.__trainable_vars_tuple__ += self.pinn.trainable_variables
            if self.fixed_loss_params == False:
                for l in self.losses:
                    self.__trainable_vars_tuple__ += tuple(l.trainable_vars())

        return self.__trainable_vars_tuple__

    def __indices__(self, batch_size: int, *ns):
        """Generator of indices for specified sizes"""
        n1 = ns[0]
        ns_remain = ns[1:] if len(ns) > 1 else []
        # First indices
        batch_steps = n1//batch_size
        batch_steps = batch_steps + (n1-1)//(batch_steps*batch_size)
        # remaining indices
        indices_batch_size = [n_i//batch_steps for n_i in ns_remain]
        indices_batch_size = [size + (batch_size//size)*(batch_size%size)
                                for size in indices_batch_size]

        # indices
        indices = [np.array(list(range(n_i))) for n_i in ns]
        for arr in indices:
            np.random.shuffle(arr)

        for batch in range(batch_steps):
            # Observation start-end
            n1_start = batch*batch_size
            n1_end = (batch+1)*batch_size
            n1_end = n1_end - (n1_end//n1)*(n1_end%n1)
            # remaining indices
            starts = [batch*size for size in indices_batch_size]
            ends = [(batch+1)*size for size in indices_batch_size]
            # Correction for remining indices
            if batch == batch_steps-1:
                ends = [ns[i+1] if end != ns[i+1] else end  for i, end in enumerate(ns_remain)]
            # step's indices
            yield [indices[0][n1_start:n1_end]] + \
                  [indices[i+1][star:end] for i, (star, end) in enumerate(zip(starts, ends))]



    @tf.function
    def train_step(self, batches_list, iteration):

        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:
            losses = []
            batch_loss = None
            for l, batch in zip(self.losses, batches_list):
                L = l.loss(batch)*l.loss_weight#*l.loss_weight(iteration)
                losses += [L]
                if batch_loss is None:
                    batch_loss = L
                else:
                    batch_loss = batch_loss + L

            #batch_loss = tf.reduce_sum(losses, name="Total_batch_loss")

        grads = tape.gradient(batch_loss,  self.trainable_vars())
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars()))

        return batch_loss, losses

    @tf.function
    def parameter_gradiants(self, full_batches):
        grads_parts = [tf.gradients(l.loss(batch),  self.trainable_vars())
                           for l, batch in zip(self.losses, full_batches)]
        grads_parts = [tf.math.reduce_mean([0.0 if v is None else tf.math.reduce_mean(tf.math.abs(v))
                            for v in g_part
                       ])
                            for g_part in grads_parts
                      ]
        return grads_parts

    @tf.function
    def train_step_gradiants(self, batches_list, iteration):

        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            losses = []
            batch_loss = None
            for l, batch in zip(self.losses, batches_list):
                L = l.loss(batch)*l.loss_weight#*l.loss_weight(iteration)
                losses += [L]
                if batch_loss is None:
                    batch_loss = L
                else:
                    batch_loss = batch_loss + L

            #batch_loss = tf.reduce_sum(losses, name="Total_batch_loss")
        grads_parts = [tape.gradient(L,  self.trainable_vars()) for L in losses]
        grads_parts = [[0.0 if v is None else v for v in g_part] for g_part in grads_parts]
        #
        grads = tape.gradient(batch_loss,  self.trainable_vars())
        self.optimizer.apply_gradients(zip(grads, self.trainable_vars()))
        #


        #zero_c = tf.constant([0.0], dtype=tf.float32)

        #grads_parts = [tape.gradient(L,  self.trainable_vars()) for L in losses]
        #loss_size = len(grads_parts)
        #vars_size = len(grads_parts[0])
        # sums gradiants along losses, each with same shape as self.trainable_vars()
        #grads = [tf.reduce_sum( [grads_parts[row][col] if grads_parts[row][col] is not None else zero_c
        #            for row in range(loss_size)], axis=0)
        #               for col in range(vars_size)
        #        ]
        #self.optimizer.apply_gradients(zip(grads, self.trainable_vars()))
        #grads_parts = [[0.0 if v is None else v for v in g_part] for g_part in grads_parts]

        return batch_loss, losses, grads_parts

    def _train_batch_experimental_(self,
                                   epoch,
                                   batch_size,
                                   datasets_sizes,
                                   init_warmup,
                                   observation_index,
                                   pde_index,
                                   gradient_ratio,
                                   loss_update_ratio):

        total_loss = 0
        loss_vals = np.zeros(len(self.losses))


        for indices_list in self.__indices__(batch_size, *datasets_sizes):
            batches_list = [ l.batch(indices) for l, indices in zip(self.losses,indices_list)]
            batch_total_loss, batch_loss_vals = self.train_step(batches_list,
                                                                tf.convert_to_tensor(epoch))
            total_loss += batch_total_loss
            loss_vals += [l.numpy() for l in batch_loss_vals]

        obs_L = loss_vals[observation_index]
        pde_L = loss_vals[pde_index]

        @tf.function
        def update_weights():
            full_batches = [ l.batch(list(range(n))) for l,n in zip(self.losses, datasets_sizes) ]
            grads_parts = self.parameter_gradiants(full_batches)
            obs = self.losses[observation_index]
            #print("it: ", epoch)
            #if f:
            #    print(obs.loss_weight, grads_parts[observation_index].numpy(), grads_parts[pde_index].numpy())
            #else:
            #    print(obs.loss_weight.numpy(), grads_parts[observation_index].numpy(), grads_parts[pde_index].numpy())
            #print(obs_L, pde_L, obs_L/pde_L)
            obs.loss_weight.assign(gradient_ratio*grads_parts[pde_index]/grads_parts[observation_index])

        # The first update
        if epoch >= init_warmup and self.warmed_up == False:
            self.warmed_up = True
            update_weights()

        # Next updates
        if self.warmed_up and obs_L/pde_L <= loss_update_ratio:
            update_weights()



        return (total_loss, loss_vals)

    def _train_batch_(self,
                      epoch,
                      batch_size,
                      datasets_sizes,
                      find_gradiants=False,
                      regularised = False,
                      regularisation_freq = 10):

        total_loss = 0
        loss_vals = np.zeros(len(self.losses))
        self.grads_vals = None

        def grad_accumulation(batch_grads_vals):
            #  | nabla_{theta} |
            if self.grads_vals is None:
                self.grads_vals = np.abs(np.array(batch_grads_vals))
            else:
                self.grads_vals += np.abs(np.array(batch_grads_vals))



        def update_loss_weigths():
            full_batches = [ l.batch(list(range(n))) for l,n in zip(self.losses, datasets_sizes) ]
            # E[| nabla_{theta} |]
            #grads_reg = np.array([np.mean([np.mean(v.numpy()) for v in L]) for L in self.grads_vals])
            #for i, l in enumerate(self.losses[1:]):
            #    # Moving average withe exponential decay
            #    # (0.5^epoch)*l.loss_weight
            #    l.loss_weight = (1-.5)*(grads_reg[0]/grads_reg[i+1])*self.losses[0].loss_weight + 0.5*l.loss_weight

            grads_reg = np.array(self.parameter_gradiants(full_batches))
            #print(grads_reg)
            for i, l in enumerate(self.losses[1:]):
                # Moving average withe exponential decay
                # (0.5^epoch)*l.loss_weight
                #l.loss_weight = (1-.5)*(grads_reg[0]/grads_reg[i+1])*self.losses[0].loss_weight + 0.5*l.loss_weight
                l.loss_weight = (grads_reg[0]/grads_reg[i+1])*self.losses[0].loss_weight

            #print([l.loss_weight for l in self.losses])

        for indices_list in self.__indices__(batch_size, *datasets_sizes):
            batches_list = [ l.batch(indices) for l, indices in zip(self.losses,indices_list)]
            if find_gradiants:
                batch_total_loss, batch_loss_vals, batch_grads_vals = \
                                   self.train_step_gradiants(batches_list,
                                                             tf.convert_to_tensor(epoch))
                grad_accumulation(batch_grads_vals)

            #elif regularised and epoch%regularisation_freq == 0:
            #    batch_total_loss, batch_loss_vals, batch_grads_vals = \
            #                       self.train_step_gradiants(batches_list,
            #                                                 tf.convert_to_tensor(epoch))
            #    grad_accumulation(batch_grads_vals)
            else:
                batch_total_loss, batch_loss_vals = self.train_step(batches_list,
                                                                tf.convert_to_tensor(epoch))
            total_loss += batch_total_loss
            loss_vals += [l.numpy() for l in batch_loss_vals]
        # regularised
        if regularised and epoch%regularisation_freq == 0:
            update_loss_weigths()

        if find_gradiants:
            return (total_loss, loss_vals, self.grads_vals)
        else:
            return (total_loss, loss_vals, None)

    def train_experimental(self,
                           epochs,
                           batch_size,
                           print_iter=10,
                           sample_stats = False,
                           sample_params = False,
                           init_warmup = 100,
                           observation_index = 0,
                           pde_index = 1,
                           observation_error_threshold = 1e-4,
                           gradient_ratio = 10,
                           loss_update_ratio=1.2):

        obs_n = self.losses[observation_index].inputs_obs.shape[0]

        datasets_sizes = [ item.data_size for item in self.losses]

        if sample_stats:
            samples_losses = np.zeros((epochs,len(self.losses)))
        else:
            samples_total_loss = None
            samples_losses = None
        if sample_params:
            samples_params = np.zeros((epochs,
                                      len([item.numpy()[0]
                                           for l in self.losses
                                           for item in l.trainable_vars()])))
        else:
            samples_params = None



        start_time = time.time()
        for epoch in range(epochs):
            total_loss, loss_vals = self._train_batch_experimental_(epoch,
                                                                    batch_size,
                                                                    datasets_sizes,
                                                                    init_warmup,
                                                                    observation_index,
                                                                    pde_index,
                                                                    gradient_ratio,
                                                                    loss_update_ratio)



            if print_iter > 0 and (epoch == 0 or (epoch+1) % print_iter == 0):
                elapsed = time.time() - start_time
                print(f"Epoch: {epoch+1}, loss:{total_loss:.2f}\n" + \
                      f"\n".join([ f"{l.name}:{val:.8f} {l.trainable_vars_str()}"
                                  for l, val in zip(self.losses, loss_vals)] ) +\
                      f"\nTime:{elapsed:.2f}\n")
                start_time = time.time()

            if sample_stats:
                samples_losses[epoch, :] = loss_vals
            if sample_params:
                samples_params[epoch, :] = [item.numpy()[0]
                                            for l in self.losses
                                            for item in l.trainable_vars()]
            obs_loss = loss_vals[observation_index]

            if obs_loss/(obs_n*loss_update_ratio) <= observation_error_threshold:
                elapsed = time.time() - start_time
                print("###############################################")
                print("#           Early Stop                        #")
                print("###############################################")
                print(f"Epoch: {epoch+1}, loss:{total_loss:.2f}\n" + \
                      f"observation loss:{obs_loss:.2f}, per data point:{obs_loss/(obs_n*loss_update_ratio):.8f}\n" + \
                      f"\n".join([ f"{l.name}:{val:.8f} {l.trainable_vars_str()}"
                                  for l, val in zip(self.losses, loss_vals)] ) +\
                      f"\nTime:{elapsed:.2f}\n")
                return (samples_losses[:epoch,:] if sample_stats else None,
                        samples_params[:epoch,:] if sample_params else None)



            gc.collect()
            tf.keras.backend.clear_session()

        return (samples_losses, samples_params)

    def train(self,
              epochs,
              batch_size,
              print_iter=10,
              stop_threshold = 0,
              regularised = False,
              regularisation_freq = 10,
              sample_stats = False,
              sample_params = False,
              sample_gradiants = False):

        datasets_sizes = [ item.data_size for item in self.losses]
        if sample_stats:
            samples_losses = np.zeros((epochs,len(self.losses)))
        else:
            samples_total_loss = None
            samples_losses = None
        if sample_params:
            samples_params = np.zeros((epochs,
                                      len([item.numpy()[0]
                                           for l in self.losses
                                           for item in l.trainable_vars()])))
        else:
            samples_params = None

        samples_grads = None

        start_time = time.time()
        for epoch in range(epochs):
            total_loss, loss_vals, grads_vals = self._train_batch_(epoch,
                                                                   batch_size,
                                                                   datasets_sizes,
                                                                   sample_gradiants,
                                                                   regularised,
                                                                   regularisation_freq)



            if print_iter > 0 and (epoch == 0 or (epoch+1) % print_iter == 0):
                elapsed = time.time() - start_time
                print(f"Epoch: {epoch+1}, loss:{total_loss:.2f}\n" + \
                      f"\n".join([ f"{l.name}:{val:.8f} {l.trainable_vars_str()}"
                                  for l, val in zip(self.losses, loss_vals)] ) +\
                      f"\nTime:{elapsed:.2f}\n")
                start_time = time.time()
            if sample_stats:
                samples_losses[epoch, :] = loss_vals
            if sample_params:
                samples_params[epoch, :] = [item.numpy()[0]
                                            for l in self.losses
                                            for item in l.trainable_vars()]

            if sample_gradiants and samples_grads is None:
                g = np.array([[np.sum(v.numpy()) for v in L] for L in grads_vals])
                #           [epochs, # losses, # trainable vars]
                #                              # trainable vars = 2 # layers + # loss params
                samples_grads = np.zeros((epochs, g.shape[0], g.shape[1]))
                samples_grads[epoch]  = g
            elif sample_gradiants:
                samples_grads[epoch]  = np.array([[np.sum(v.numpy()) for v in L] for L in grads_vals])

            if total_loss <= stop_threshold :
                elapsed = time.time() - start_time
                print("###############################################")
                print("#           Early Stop                        #")
                print("###############################################")
                print(f"Epoch: {epoch+1}, loss:{total_loss:.2f}\n" + \
                      f"\n".join([ f"{l.name}:{val:.8f} {l.trainable_vars_str()}\n"
                                  for l, val in zip(self.losses, loss_vals)] ) +\
                      f"\nTime:{elapsed:.2f}\n")
                return (samples_losses[:epoch,:] if sample_stats else None,
                        samples_params[:epoch,:] if sample_params else None,
                        samples_grads[:epoch] if sample_gradiants else None)



            gc.collect()
            #tf.keras.backend.clear_session()

        return (samples_losses, samples_params, samples_grads)