import tensorflow as tf
from tensorflow import keras
import numpy as np
from . import Loss

class Observations(Loss):
    def __init__(self, pinn, inputs_obs, outputs_obs):        
        self.inputs_obs = inputs_obs
        self.outputs_obs = outputs_obs
        super().__init__(pinn, "Loss_observations", inputs_obs.shape[0])
        
    def batch(self, indices):
        return (self.inputs_obs[indices], self.outputs_obs[indices])
    
    @tf.function
    def loss(self, batch):
        inputs, outputs = batch
        obs_pred = self.pinn(inputs, grads = False)
        L = tf.reduce_sum(tf.square(obs_pred - outputs), name = self.name)
        return L
    
class Periodic_boundary(Loss):
    def __init__(self, pinn, inputs_LB_boundary, inputs_TR_boundary):        
        self.inputs_LB_boundary = inputs_LB_boundary
        self.inputs_TR_boundary = inputs_TR_boundary
        super().__init__(pinn, "Loss_Periodic_Boundary", inputs_LB_boundary.shape[0])
    
    def batch(self, indices):
        return self.inputs_LB_boundary[indices], self.inputs_TR_boundary[indices]
    
    @tf.function
    def loss(self, batch):
        inputs_LB, inputs_TR = batch
        boundary_LB_pred = self.pinn(inputs_LB, grads = False)
        boundary_TR_pred = self.pinn(inputs_TR, grads = False)
        L = tf.reduce_sum(tf.square(boundary_LB_pred - boundary_TR_pred), 
                          name = self.name)
        return L
    
class Truing_PDE(Loss):
    def __init__(self, pinn, inputs_pde, name="Loss_Turing_PDE"):        
        self.inputs_pde = inputs_pde
        super().__init__(pinn, name ,inputs_pde.shape[0])
        
    def batch(self, indices):
        return self.inputs_pde[indices]
       
    @tf.function
    def loss(self, batch):
        inputs = batch
        pde_outputs, partials_1, partials_2 = self.pinn(inputs, grads = True)
        
        pde_res = self.pde(pde_outputs, partials_1, partials_2)
        L = tf.reduce_sum(tf.square(pde_res), name = self.name)
        return L
    
    def pde(self, outputs, partials_1, partials_2):
        pass
    
class ASDM(Truing_PDE):
    def __init__(self, pinn, inputs_pde):
        super().__init__(pinn, inputs_pde, name="Loss_ASDM")
        self.sigma_a = tf.Variable([0.0], dtype=tf.float64,
                                   name="sigma_a",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.sigma_s = tf.Variable([1.00], dtype=tf.float64, 
                                   name="sigma_s",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.mu_a = tf.Variable([1.00], dtype=tf.float64, 
                                name="mu_a",
                                constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.rho_a = tf.Variable([1.00], dtype=tf.float64, 
                                 name="rho_a",
                                 constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.rho_s = tf.Variable([1.00], dtype=tf.float64, 
                                 name="rho_s",
                                 constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.kappa_a = tf.Variable([1.00], dtype=tf.float64,
                                   name="kappa_a",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        
    def trainable_vars(self):
        return [self.sigma_a,
                self.sigma_s,
                self.mu_a,
                self.rho_a,
                self.rho_s,
                self.kappa_a]
        
    def pde(self, outputs, partials_1, partials_2):
        a = outputs[:, 0]
        s = outputs[:, 1]
        
        a_x = partials_1[0][:, 0]
        a_y = partials_1[0][:, 1]
        a_t = partials_1[0][:, 2]
        
        a_xx = partials_2[0][:, 0]
        a_yy = partials_2[0][:, 1]
        
        
        s_x = partials_1[1][:, 0]
        s_y = partials_1[1][:, 1]
        s_t = partials_1[1][:, 2]
        
        s_xx = partials_2[1][:, 0]
        s_yy = partials_2[1][:, 1]
        
        sigma_a = self.sigma_a
        sigma_s = self.sigma_s
        mu_a = self.mu_a
        rho_a = self.rho_a
        rho_s = self.rho_s
        kappa_a = self.kappa_a
        
        one_1 = tf.constant(1.0, dtype=tf.float64)
        f = a*a*s/(one_1 + kappa_a*a*a)
        f_a = a_t - (a_xx + a_yy) - rho_a*f + mu_a*a - sigma_a
        f_s = s_t - (s_xx + s_yy) + rho_s*f - sigma_s
        
        return tf.concat([tf.expand_dims(f_a,axis=1), 
                          tf.expand_dims(f_s,axis=1),], axis = 1)
