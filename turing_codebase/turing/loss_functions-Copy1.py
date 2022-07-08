import tensorflow as tf
from tensorflow import keras
import numpy as np
from . import Loss

class Observations(Loss):
    def __init__(self, pinn, inputs_obs, outputs_obs, init_loss_weight = 1.0):        
        self.inputs_obs = inputs_obs
        self.outputs_obs = outputs_obs
        super().__init__(pinn, "Loss_observations", inputs_obs.shape[0], init_loss_weight)
        
    def batch(self, indices):
        return (tf.convert_to_tensor(self.inputs_obs[indices]), 
                tf.cast(tf.convert_to_tensor(self.outputs_obs[indices]), tf.float32))
    
    @tf.function
    def loss(self, batch):
        inputs, outputs = batch        
        obs_pred = self.pinn(inputs, grads = False)
        L = tf.reduce_sum(tf.square(obs_pred - outputs), name = self.name)
        return L
    
class Periodic_boundary(Loss):
    def __init__(self, pinn, inputs_LB_boundary, inputs_RT_boundary, init_loss_weight = 1.0):        
        self.inputs_LB_boundary = inputs_LB_boundary
        self.inputs_RT_boundary = inputs_RT_boundary
        super().__init__(pinn, "Loss_Periodic_Boundary", inputs_LB_boundary.shape[0], init_loss_weight)
    
    def batch(self, indices):
        return (tf.convert_to_tensor(self.inputs_LB_boundary[indices]), 
                tf.convert_to_tensor(self.inputs_RT_boundary[indices]))
    
    #@tf.function
    def loss(self, batch):
        inputs_LB, inputs_RT = batch
        boundary_LB_pred = self.pinn(inputs_LB, grads = False)
        boundary_RT_pred = self.pinn(inputs_RT, grads = False)
        L = tf.reduce_sum(tf.square(boundary_LB_pred - boundary_RT_pred), 
                          name = self.name)        
        return L
    
class Turing_PDE(Loss):
    def __init__(self, pinn, inputs_pde, name="Loss_Turing_PDE", init_loss_weight = 1.0):        
        self.inputs_pde = inputs_pde
        super().__init__(pinn, name ,inputs_pde.shape[0], init_loss_weight)
        
    def batch(self, indices):
        return tf.convert_to_tensor(self.inputs_pde[indices])
       
    @tf.function
    def loss(self, batch):
        inputs = batch
        pde_outputs, partials_1, partials_2 = self.pinn(inputs, grads = True)
        
        pde_res = self.loss(pde_outputs, partials_1, partials_2)
        L = tf.reduce_sum(tf.square(pde_res), name = self.name)
        return L
    
    def loss(self, outputs, partials_1, partials_2):
        pass
    
class ASDM(Turing_PDE):
    def __init__(self, 
                 pinn, 
                 inputs_pde, 
                 init_loss_weight = 1.0,
                 sigma_a = 1.0,
                 sigma_s = 1.0,
                 mu_a = 1.0,
                 rho_a = 1.0,
                 rho_s = 1.0,
                 kappa_a = 1.0
                ):
        super().__init__(pinn, inputs_pde, name="Loss_ASDM", init_loss_weight= init_loss_weight)
        
        self.sigma_a = tf.Variable([sigma_a], dtype=tf.float32,
                                   name="sigma_a",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.sigma_s = tf.Variable([sigma_s], dtype=tf.float32, 
                                   name="sigma_s",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.mu_a = tf.Variable([mu_a], dtype=tf.float32, 
                                name="mu_a",
                                constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.rho_a = tf.Variable([rho_a], dtype=tf.float32, 
                                 name="rho_a",
                                 constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.rho_s = tf.Variable([rho_s], dtype=tf.float32, 
                                 name="rho_s",
                                 constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.kappa_a = tf.Variable([kappa_a], dtype=tf.float32,
                                   name="kappa_a",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        
    def trainable_vars(self):
        return [self.sigma_a,
                self.sigma_s,                
                self.rho_a,
                self.rho_s,
                self.mu_a,
                self.kappa_a]
        
    def loss(self, outputs, partials_1, partials_2):
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
        
        one_1 = tf.constant(1.0, dtype=tf.float32)
        f = a*a*s/(one_1 + kappa_a*a*a)
        f_a = a_t - (a_xx + a_yy) - rho_a*f + mu_a*a - sigma_a
        f_s = s_t - (s_xx + s_yy) + rho_s*f - sigma_s
        
        return tf.concat([tf.expand_dims(f_a, axis=1), 
                          tf.expand_dims(f_s, axis=1)], axis = 1)
    
class schnakenberg(Turing_PDE):
    def __init__(self, 
                 pinn, 
                 inputs_pde, 
                 init_loss_weight = 1.0,
                 D_u = 10.0,
                 D_v = 10.0,
                 c_0 = 10.0,
                 c_1 = 10.0,
                 c_2 = 10.0                 
                ):
        super().__init__(pinn, inputs_pde, name="Loss_Schnakenberg", init_loss_weight= init_loss_weight)
        
        self.D_u = tf.Variable([D_u], dtype=tf.float32,
                                   name="D_u",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.D_v = tf.Variable([D_v], dtype=tf.float32,
                                   name="D_v",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.c_0 = tf.Variable([c_0], dtype=tf.float32,
                                   name="c_0",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.c_1 = tf.Variable([c_1], dtype=tf.float32, 
                                   name="c_1",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.c_2 = tf.Variable([c_2], dtype=tf.float32, 
                                name="c_2",
                                constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        
        
        
    def trainable_vars(self):
        return [self.D_u,
                self.D_v,
                self.c_0,
                self.c_1,                
                self.c_2]
        
    def loss(self, outputs, partials_1, partials_2):
        u = outputs[:, 0]
        v = outputs[:, 1]
        
        u_x = partials_1[0][:, 0]
        u_y = partials_1[0][:, 1]
        u_t = partials_1[0][:, 2]
        
        u_xx = partials_2[0][:, 0]
        u_yy = partials_2[0][:, 1]
        
        
        v_x = partials_1[1][:, 0]
        v_y = partials_1[1][:, 1]
        v_t = partials_1[1][:, 2]
        
        v_xx = partials_2[1][:, 0]
        v_yy = partials_2[1][:, 1]
        
        D_u = self.D_u
        D_v = self.D_v
        c_0 = self.c_0
        c_1 = self.c_1
        c_2 = self.c_2
               
        
        u2v = u*u*v
        f_u = u_t - D_u*(u_xx + u_yy) - c_1 + c_0*u -  u2v
        f_v = v_t - D_v*(v_xx + v_yy) - c_2 + u2v
        
        return tf.concat([tf.expand_dims(f_u, axis=1), 
                          tf.expand_dims(f_v, axis=1)], axis = 1)    
    
class schnakenberg2(Turing_PDE):
    def __init__(self, 
                 pinn, 
                 inputs_pde, 
                 init_loss_weight = 1.0,
                 #D_u = 10.0,
                 #D_v = 10.0,
                 c_0 = 10.0#,
                 #c_1 = 10.0,
                 #c_2 = 10.0                 
                ):
        super().__init__(pinn, inputs_pde, name="Loss_Schnakenberg", init_loss_weight= init_loss_weight)
        
        #self.D_u = tf.Variable([D_u], dtype=tf.float32,
        #                           name="D_u",
        #                           constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        #self.D_v = tf.Variable([D_v], dtype=tf.float32,
        #                           name="D_v",
        #                           constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        self.c_0 = tf.Variable([c_0], dtype=tf.float32,
                                   name="c_0",
                                   constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
#         self.c_1 = tf.Variable([c_1], dtype=tf.float32, 
#                                    name="c_1",
#                                    constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
#         self.c_2 = tf.Variable([c_2], dtype=tf.float32, 
#                                 name="c_2",
#                                 constraint= lambda z: tf.clip_by_value(z, 0, 1e10))
        
        
        
    def trainable_vars(self):
        return [#self.D_u,
                #self.D_v,
                self.c_0,
                #self.c_1,                
                #self.c_2
               ]
        
    def loss(self, outputs, partials_1, partials_2):
        u = outputs[:, 0]
        v = outputs[:, 1]
        
        u_x = partials_1[0][:, 0]
        u_y = partials_1[0][:, 1]
        u_t = partials_1[0][:, 2]
        
        u_xx = partials_2[0][:, 0]
        u_yy = partials_2[0][:, 1]
        
        
        v_x = partials_1[1][:, 0]
        v_y = partials_1[1][:, 1]
        v_t = partials_1[1][:, 2]
        
        v_xx = partials_2[1][:, 0]
        v_yy = partials_2[1][:, 1]
        
        #D_u = self.D_u
        #D_v = self.D_v
        c_0 = self.c_0
        #c_1 = self.c_1
        #c_2 = self.c_2
               
        
        u2v = u*u*v
        f_u = u_t - 1.0*(u_xx + u_yy) - 0.1 + c_0*u -  u2v
        f_v = v_t - 40.0*(v_xx + v_yy) - 0.9 + u2v
        
        return tf.concat([tf.expand_dims(f_u, axis=1), 
                          tf.expand_dims(f_v, axis=1)], axis = 1)