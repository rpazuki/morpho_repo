from collections import namedtuple
import torch
import torch.nn as nn

train_settings = namedtuple('train_settings',
                             ('model', 'optimizer', 'epochs'))


    
# for i in range(epochs):
#     optimizer.zero_grad()

#     #################################
#     # data loss
#     # data_f_hat = model(data_X)
#     # loss_data = torch.mean((data_f - data_f_hat)**2)

#     #################################
#     # boundary loss
#     # boundary_f_hat = model(boundary_X)
#     # loss_boundary = torch.mean((boundary_f - boundary_f_hat)**2)

#     #################################
#     # initial conditoion loss
#     # init_cond_f_hat = model(init_cond_X)
#     # loss_init_cond = torch.mean((init_cond_f - init_cond_f_hat)**2)

#     #################################
#     # physics derivatives
#     # physics_f = model(physics_X)
#     # dx  = torch.autograd.grad(physics_f, physics_X, torch.ones_like(physics_f), create_graph=True)[0]# computes dy/dx
#     # dx2 = torch.autograd.grad(dx, physics_X, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
#     ################################
#     # physics loss
#     # Construct the physics loss here
#     # diff_pde
#     # loss_physics = torch.mean(diff_pde**2)    

#     total_loss = (
#             lambda_data*loss_data
#         +lambda_boundary*loss_boundary
#         +lambda_init_cond*loss_init_cond
#         +lambda_physics*loss_physics
#                     )

#     total_loss.backward()
#     optimizer.step() 

def train_template(settings, callback):
    model = settings.model
    optimizer = settings.optimizer
    epochs = settings.epochs

    loss_data = 0.0
    lambda_data = 1.0
    loss_boundary = 0.0
    lambda_boundary = 1.0
    loss_init_cond = 0.0
    lambda_init_cond = 1.0
    loss_physics = 0.0
    lambda_physics = 1.0
    
    for i in range(epochs):
        optimizer.zero_grad()                
        
        loss_data, loss_boundary, loss_init_cond, loss_physics = callback(i, epochs, settings, model)

        total_loss = (
             lambda_data*loss_data
            +lambda_boundary*loss_boundary
            +lambda_init_cond*loss_init_cond
            +lambda_physics*loss_physics
                     )

        total_loss.backward()
        optimizer.step() 

class Net_base(nn.Module):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self):
        pass
        
class Net_dense(Net_base):   
    def __init__(self, layers, activation = nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = activation
        self.build()        
        
    def build(self):
        """Create the state of the layers (weights)"""                
        self.input_layer = nn.Sequential(*[
                        nn.Linear(self.layers[0], self.layers[1]),
                        self.activation()])
        
        self.hidden_layers = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.layers[i], self.layers[i+1]),
                            self.activation()]) for i in range(1, self.num_layers - 2)])
        
        self.output_layer = nn.Linear(self.layers[-2], self.layers[-1])
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class Net_dense_normalised(Net_dense):
    def __init__(self, layers, lb, ub, activation = nn.Tanh, **kwargs):
        self.lb = lb
        self.ub = ub
        super().__init__(layers, activation, **kwargs)

    def forward(self, x):
        # Map the inputs to the range [-1, 1]
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
        

    