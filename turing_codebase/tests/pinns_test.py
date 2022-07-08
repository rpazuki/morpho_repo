import os
import tensorflow as tf
import numpy as np
from turing import NN
from turing import TINN
from turing.loss_functions import ASDM
from turing.utils import create_dataset

data_path = os.path.abspath("turing.npy")
with open(data_path, 'rb') as f:
    data = np.load(f)

data_path = os.path.abspath("turing_t.npy")
with open(data_path, 'rb') as f:
    t_star = np.load(f)

T = t_star.shape[0]
L = 50
x_size = data.shape[1]
y_size = data.shape[2]
N = x_size * y_size

model_params = {'training_data_size': T * N,  # T*32,
                'pde_data_size': (T * N) // 32,
                'boundary_data_size': ((x_size + y_size) * T) // 8}

dataset = create_dataset(data, t_star, N, T, L, **model_params)
lb = dataset['lb']
ub = dataset['ub']
obs_X = dataset['obs_input']
obs_Y = dataset['obs_output']

def test_pinn():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_a=0.005, D_s=0.2)
    model = TINN(pinn,
                 pde_loss,
                 extra_loss=[])

    results = model.train(epochs=2,
                          batch_size=512,
                          X=obs_X,
                          Y=obs_Y,
                          print_interval=1,
                          stop_threshold=1e-5,
                          shuffle=True,
                          sample_losses=True,
                          sample_regularisations=True,
                          sample_gradients=True)
    assert results["training_obs_accuracy"].shape[0] == 2
