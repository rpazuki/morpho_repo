import os
import copy
import tensorflow as tf
import numpy as np
from turing import NN
from turing import TINN, TINN_multi_nodes
from turing.loss_functions import ASDM
from turing.loss_functions import Schnakenberg
from turing.loss_functions import Non_zero_params
from turing.utils import create_dataset

data_path = os.path.abspath("turing.npy")
with open(data_path, "rb") as f:
    data = np.load(f)

data_path = os.path.abspath("turing_t.npy")
with open(data_path, "rb") as f:
    t_star = np.load(f)

T = t_star.shape[0]
L = 50
x_size = data.shape[1]
y_size = data.shape[2]
N = x_size * y_size

model_params = {
    "training_data_size": T * N,  # T*32,
    "pde_data_size": (T * N) // 32,
    "boundary_data_size": ((x_size + y_size) * T) // 8,
}

dataset = create_dataset(data, t_star, N, T, L, **model_params)
lb = dataset["lb"]
ub = dataset["ub"]
obs_X = dataset["obs_input"]
obs_Y = dataset["obs_output"]
pde_X = dataset["pde"]


def test_pinn_samples():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_u=0.005, D_v=0.2)
    non_zero_loss_1 = Non_zero_params(f"{pde_loss.name}_1", [pde_loss.D_u, pde_loss.D_v])
    non_zero_loss_2 = Non_zero_params(f"{pde_loss.name}_2", [pde_loss.D_u, pde_loss.D_v])
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[non_zero_loss_1, non_zero_loss_2], node_names=["u", "v"])

    results = model.train(
        epochs=3,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 3
    assert results["loss_total"].shape[0] == 3
    assert results["loss_regularisd_total"].shape[0] == 3
    assert results["loss_obs"].shape[0] == 3
    assert results["loss_obs"].shape[1] == 2
    assert results["loss_pde"].shape[0] == 3
    assert results["loss_pde"].shape[1] == 2
    assert results["lambda_obs"].shape[0] == 3
    assert results["lambda_obs"].shape[1] == 2
    assert results["lambda_pde"].shape[0] == 3
    assert results["lambda_pde"].shape[1] == 2
    assert results["grads_obs"].shape[0] == 3
    assert results["grads_obs"].shape[1] == 2
    assert results["grads_pde"].shape[0] == 3
    assert results["grads_pde"].shape[1] == 2
    assert results["loss_extra_non_zero_Loss_ASDM_1"].shape[0] == 3
    assert results["loss_extra_non_zero_Loss_ASDM_2"].shape[0] == 3


def test_pinn_ASDM():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_u=0.005, D_v=0.2)
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[])

    results = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 2


def test_pinn_Schnakenberg():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = Schnakenberg(dtype=tf.float64, D_u=1.0, D_v=40)
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[])

    results = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 2


def test_pinn_Non_zero_params():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = Schnakenberg(dtype=tf.float64, D_u=1.0, D_v=40)
    non_zero_loss = Non_zero_params(pde_loss.name, [pde_loss.D_u, pde_loss.D_v])
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[non_zero_loss])

    results = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 2


def test_pinn_extra_loss():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = Schnakenberg(dtype=tf.float64, D_u=1.0, D_v=40)
    non_zero_loss = Non_zero_params(pde_loss.name, [pde_loss.D_u, pde_loss.D_v])
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[non_zero_loss, non_zero_loss])

    results = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 2


def test_pinn_observations_and_pde():
    layers = [3, 64, 64, 64, 64, 2]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_u=0.005, D_v=0.2)
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[])

    results = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        X_pde=pde_X,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=True,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )
    assert results["training_obs_accuracy"].shape[0] == 2


def test_multi_and_2_are_the_same():
    layers = [3, 64, 64, 64, 64, 2]
    model_params = {
        "training_data_size": T * N,  # T*32,
        "pde_data_size": (T * N) // 32,
        "boundary_data_size": ((x_size + y_size) * T) // 8,
        "shuffle": False,
    }
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)
    tf.random.set_seed(42)
    dataset = create_dataset(data, t_star, N, T, L, **model_params)
    lb = dataset["lb"]
    ub = dataset["ub"]
    obs_X = dataset["obs_input"]
    obs_Y = dataset["obs_output"]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_u=0.005, D_v=0.2)
    model = TINN(pinn, pde_loss, extra_loss=[])

    results1 = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=False,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )

    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)
    tf.random.set_seed(42)
    dataset = create_dataset(data, t_star, N, T, L, **model_params)
    lb = dataset["lb"]
    ub = dataset["ub"]
    obs_X = dataset["obs_input"]
    obs_Y = dataset["obs_output"]
    pinn = NN(layers, lb, ub, dtype=tf.float64)
    pde_loss = ASDM(dtype=tf.float64, D_u=0.005, D_v=0.2)
    model = TINN_multi_nodes(pinn, pde_loss, extra_loss=[])

    results2 = model.train(
        epochs=2,
        batch_size=512,
        X=obs_X,
        Y=obs_Y,
        print_interval=1,
        stop_threshold=1e-5,
        shuffle=False,
        sample_losses=True,
        sample_regularisations=True,
        sample_gradients=True,
    )

    assert np.isclose(results1["loss_total"][0], results2["loss_total"][0])
    assert np.isclose(results1["loss_total"][1], results2["loss_total"][1])
