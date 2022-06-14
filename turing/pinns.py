import time
import tensorflow as tf
from tensorflow import keras
import numpy as np

class NN(tf.keras.layers.Layer):
    
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
        
    def build(self, input_shape):         
        """Create the state of the layers (weights)"""
        weights = []
        biases = []        
        for l in range(0,self.num_layers-1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]], dtype=tf.float64), dtype=tf.float64)
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
                                                         dtype=tf.float64), 
                           dtype=tf.float64)
    
    @tf.function
    def normalise_input(self, inputs):
        """Map the inputs to the range [-1, 1]"""
        return 2.0*(inputs - self.lb)/(self.ub - self.lb) - 1.0
    
    @tf.function
    def __net__(self, inputs):
        H = self.normalise_input(inputs)
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
            W = self.Ws[-1]
            b = self.bs[-1]
            outputs = tf.add(tf.matmul(H, W), b)
        return outputs
    
    def call(self, inputs, grads=True):
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
        X = tf.cast(inputs, tf.float64)
        outputs = self.__net__(X)
        if grads:                        
            partials_1 = [tf.gradients(outputs[:,i], X)[0] for i in range(outputs.shape[1])]
            partials_2 = [tf.gradients(partials_1[i], X)[0] for i in range(outputs.shape[1])]
            return outputs, partials_1, partials_2            
        else:            
            return outputs   
    
    
class Loss():
    def __init__(self, pinn, name, data_size = 0, init_loss_weight = 1.0):
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
        self.init_loss_weight = init_loss_weight
    
    
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
    
    @tf.function
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
              return L
        
        """
        pass
    
    def loss_weight(self, iteration):
        """Return weigth of the loss in comparision to others
        
        Args: 
            iteration: the iteration index ( for hyper-parametrs that update
                       during the training)
        """
        return self.init_loss_weight
    
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
    """Turing-Informed Neoral Net"""
    def __init__(self,
                 pinn,
                 losses: Loss):
        self.pinn = pinn
        self.losses = losses        
        self.optimizer_Adam = keras.optimizers.Adam()
        
        
    def __indices__(self, batch_size, *ns):
        """Generator of indices for specified sizes"""
        n1 = ns[0]
        ns_remain = ns[1:] if len(ns) > 1 else []                
        # First indices        
        batch_steps = n1//batch_size
        batch_steps = batch_steps + (n1-1)//(batch_steps*batch_size)
        # remaining indices        
        indices_batch_size = [n_i//batch_steps for n_i in ns_remain]         
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
                ends = [end if end < ns[i] else ns[i]  for i, end in enumerate(ns_remain)]
            # step's indices            
            yield [indices[0][n1_start:n1_end]] + \
                  [indices[i+1][star:end] for i, (star, end) in enumerate(zip(starts, ends))]
                   
    def train(self, epochs, batch_size, print_iter=10):
        
        datasets_sizes = [ item.data_size for item in self.losses] 
        samples_total_loss = np.zeros(epochs)
        samples_losses = np.zeros((epochs,len(self.losses)))
        
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0
            loss_vals = np.zeros(len(self.losses))

            for indices_list in self.__indices__(batch_size, *datasets_sizes):
                batches_list = [ l.batch(indices) for l, indices in zip(self.losses,indices_list)]                 
                batch_total_loss, batch_loss_vals = self.train_step(batches_list, epoch)
                total_loss += batch_total_loss
                loss_vals += [l.numpy() for l in batch_loss_vals]

                
            if epoch % print_iter == 0:
                elapsed = time.time() - start_time                                                                
                print(f"Epoch: {epoch}, loss:{total_loss:.2f}\n" + \
                      f"\n".join([ f"{l.name}:{val:.8f} {l.trainable_vars_str()}" 
                                  for l, val in zip(self.losses, loss_vals)] ) +\
                      f"\nTime:{elapsed:.2f}\n")
                start_time = time.time()
                
            samples_total_loss[epoch] = total_loss
            samples_losses[epoch, : ] = loss_vals
            
        return (samples_total_loss, samples_losses)
    
    @tf.function
    def train_step(self, batches_list, iteration):        
            
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:            
            losses = []
            for l, batch in zip(self.losses, batches_list):
                L = l.loss(batch)*l.loss_weight(iteration)
                losses += [L]
                
            batch_loss = tf.reduce_sum(losses, name="Total_batch_loss")
            
        #grads = tape.gradient(batch_loss,  self.trainable_vars)
        grads = tape.gradient(batch_loss,  tape.watched_variables())
        #self.optimizer_Adam.apply_gradients(zip(grads, self.trainable_vars))
        self.optimizer_Adam.apply_gradients(zip(grads, tape.watched_variables()))
        
        return batch_loss, losses

