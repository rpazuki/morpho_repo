import os
import pathlib
from collections import namedtuple
import numpy as np
import tensorflow as tf

Simulation = namedtuple(
    "Simulation",
    [
        "name",
        "n",
        "L",
        "Ds",
        "dt",
        "t_start",
        "t_end",
        "t_steps",
        "parameters",
        "steady_state_func",
        "perturbation_size",
        "kinetic_func",
        "tol",
        "sample_parameters",
        "sample_parameters_num",
        "sample_parameters_std",
    ],
)


class TINN_Dataset(tf.data.Dataset):
    # def
    def __new__(cls, X, Y, X_PDE=None, shuffle=True):

        ds = tf.data.Dataset.from_tensor_slices((X, Y) if X_PDE is None else (X, Y, X_PDE))
        if shuffle:
            ds = ds.shuffle(len(X), reshuffle_each_iteration=True)

        setattr(ds, "x_size", len(X))
        setattr(ds, "x_pde_size", ds.x_size if X_PDE is None else len(X_PDE))
        # setattr(ds, "X", X)
        return ds


class TINN_Single_Sim_Dataset(TINN_Dataset):
    def __new__(cls, path, name, thining_start=0, thining_step=0, pde_ratio=0, shuffle=True):
        data_path = pathlib.PurePath(path)
        with open(data_path.joinpath(f"{name}.npy"), "rb") as f:
            data = np.load(f)
        with open(data_path.joinpath("simulation.txt"), "r") as f:
            simulation = eval(f.read())
        t_star = np.linspace(simulation.t_start, simulation.t_end, simulation.t_steps)
        # Thining the dataset
        t_star = t_star[thining_start:]
        data = data[..., thining_start:]
        if thining_step > 0:
            t_star = t_star[::thining_step]
            data = data[..., ::thining_step]

        T = t_star.shape[0]

        L = simulation.L[0]
        x_size = simulation.n[0]  #
        y_size = simulation.n[1]  #
        N = x_size * y_size

        model_params = {"training_data_size": T * N}
        if pde_ratio > 0:
            model_params = {**model_params, **{"pde_data_size": (T * N) / pde_ratio}}

        dataset = create_dataset(data, t_star, N, T, L, **model_params)
        obs_X = dataset["obs_input"]
        obs_Y = dataset["obs_output"]
        if pde_ratio > 0:
            pde_X = dataset["pde"]
        else:
            pde_X = None
        ds = super().__new__(cls, obs_X, obs_Y, pde_X, shuffle)
        setattr(ds, "lb", dataset["lb"])
        setattr(ds, "ub", dataset["ub"])
        setattr(ds, "simulation", simulation)
        setattr(ds, "ts", t_star)
        return ds


def lower_upper_bounds(inputs_of_inputs):
    """Find the lower and upper bounds of inputs

    inputs_of_inputs: a list of tensors that their axis one have the same number
                      of columns
    """

    inputs_dim = np.asarray(inputs_of_inputs[0]).shape[1]
    lb = np.array([np.inf] * inputs_dim)
    ub = np.array([-np.inf] * inputs_dim)
    for i, inputs in enumerate(inputs_of_inputs):
        assert inputs_dim == np.asarray(inputs).shape[1]
        lb = np.amin(np.c_[inputs.min(0), lb], 1)
        ub = np.amax(np.c_[inputs.max(0), ub], 1)

    return lb, ub


def indice(batch_size: int, shuffle: bool = True, *ns):
    """For old code"""
    return indices(batch_size, shuffle, *ns)


def indices(batch_size: int, shuffle: bool = True, *ns):
    """Generator of indices for specified sizes"""
    n1 = ns[0]
    ns_remain = ns[1:] if len(ns) > 1 else []
    # First indices
    batch_steps = n1 // batch_size
    batch_steps = batch_steps + (n1 - 1) // (batch_steps * batch_size)
    # remaining indices
    indices_batch_size = [n_i // batch_steps for n_i in ns_remain]
    # indices_batch_size = [size + (batch_size // size) * (batch_size % size) for size in indices_batch_size]

    # indices
    indices = [np.array(list(range(n_i))) for n_i in ns]
    if shuffle:
        for arr in indices:
            np.random.shuffle(arr)

    for batch in range(batch_steps):
        # Observation start-end
        n1_start = batch * batch_size
        n1_end = (batch + 1) * batch_size
        n1_end = n1_end - (n1_end // n1) * (n1_end % n1)
        # remaining indices
        starts = [batch * size for size in indices_batch_size]
        ends = [(batch + 1) * size for size in indices_batch_size]
        # Correction for remining indices
        if batch == batch_steps - 1:
            ends = [ns[i + 1] if end != ns[i + 1] else end for i, end in enumerate(ns_remain)]
        # step's indices
        yield [indices[0][n1_start:n1_end]] + [
            indices[i + 1][star:end] for i, (star, end) in enumerate(zip(starts, ends))
        ]


def create_dataset(
    data,
    t_star,
    N,
    T,
    L,
    training_data_size,
    pde_data_size=None,
    boundary_data_size=None,
    signal_to_noise=0,
    shuffle=True,
):
    x_size = data.shape[1]
    y_size = data.shape[2]
    x_domain = L * np.linspace(0, 1, x_size)
    y_domain = L * np.linspace(0, 1, y_size)

    X, Y = np.meshgrid(x_domain, y_domain, sparse=False, indexing="ij")
    XX = np.tile(X.flatten(), T)  # N x T
    YY = np.tile(Y.flatten(), T)  # N x T
    TT = np.repeat(t_star[-T:], N)  # T x N

    AA = np.einsum("ijk->kij", data[0, :, :, -T:]).flatten()  # N x T
    SS = np.einsum("ijk->kij", data[1, :, :, -T:]).flatten()  # N x T

    # x = XX[:, np.newaxis]  # NT x 1
    # y = YY[:, np.newaxis]  # NT x 1
    # t = TT[:, np.newaxis]  # NT x 1

    # a = AA[:, np.newaxis]  # NT x 1
    # s = SS[:, np.newaxis]  # NT x 1
    if boundary_data_size is not None:
        boundary_x_LB = np.concatenate((x_domain, np.repeat(x_domain[0], y_size)))
        boundary_x_RT = np.concatenate((x_domain, np.repeat(x_domain[-1], y_size)))

        boundary_y_LB = np.concatenate((np.repeat(y_domain[0], x_size), y_domain))
        boundary_y_RT = np.concatenate((np.repeat(y_domain[-1], x_size), y_domain))

        boundary_XX_LB = np.tile(boundary_x_LB.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
        boundary_XX_RT = np.tile(boundary_x_RT.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
        boundary_YY_LB = np.tile(boundary_y_LB.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
        boundary_YY_RT = np.tile(boundary_y_RT.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
        # T x (x_size + y_size), 1
        boundary_TT = np.repeat(t_star[-T:], (x_size + y_size))[:, np.newaxis]
    ##########################################
    # Including noise
    if signal_to_noise > 0:
        signal_amp_a = (np.max(AA) - np.min(AA)) / 2.0
        signal_amp_s = (np.max(SS) - np.min(SS)) / 2.0
        sigma_a = signal_amp_a * signal_to_noise
        sigma_s = signal_amp_s * signal_to_noise
    # Observed data
    if shuffle:
        idx_data = np.random.choice(N * T, training_data_size, replace=False)
    else:
        idx_data = list(range(training_data_size))
    # PDE colocations
    if pde_data_size is not None:
        if shuffle:
            idx_pde = np.random.choice(N * T, pde_data_size, replace=False)
        else:
            idx_pde = list(range(pde_data_size))
    # Periodic boundary condition
    if boundary_data_size is not None:
        if shuffle:
            idx_boundary = np.random.choice((x_size + y_size) * T, boundary_data_size, replace=False)
        else:
            idx_boundary = list(range(boundary_data_size))

    # Lower/Upper bounds
    lb, ub = lower_upper_bounds([np.c_[XX, YY, TT]])

    ret = {
        "obs_input": np.c_[XX[idx_data], YY[idx_data], TT[idx_data]],
        "obs_output": np.c_[AA[idx_data], SS[idx_data]],
        "lb": lb,
        "ub": ub,
    }
    if pde_data_size is not None:
        ret = {**ret, **{"pde": np.c_[XX[idx_pde], YY[idx_pde], TT[idx_pde]]}}
    if signal_to_noise > 0:
        ret["obs_output"][:, 0] += sigma_a * np.random.randn(len(idx_data))
        ret["obs_output"][:, 1] += sigma_s * np.random.randn(len(idx_data))

    if boundary_data_size is not None:
        ret = {
            **ret,
            **{
                "boundary_LB": np.c_[
                    boundary_XX_LB[idx_boundary], boundary_YY_LB[idx_boundary], boundary_TT[idx_boundary]
                ],
                "boundary_RT": np.c_[
                    boundary_XX_RT[idx_boundary], boundary_YY_RT[idx_boundary], boundary_TT[idx_boundary]
                ],
            },
        }

    return ret


def create_dataset_mask(
    data,
    mask,
    t_star,
    N,
    T,
    L,
    training_data_size,
    pde_data_size,
    boundary_data_size,
    with_boundary=True,
    signal_to_noise=0,
    shuffle=True,
):
    x_size = data.shape[1]
    y_size = data.shape[2]
    x_domain = L * np.linspace(0, 1, x_size)
    y_domain = L * np.linspace(0, 1, y_size)

    X, Y = np.meshgrid(x_domain, y_domain, sparse=False, indexing="ij")
    XX = np.tile(X.flatten(), T)  # N x T
    YY = np.tile(Y.flatten(), T)  # N x T
    TT = np.repeat(t_star[-T:], N)  # T x N

    AA = np.einsum("ijk->kij", data[0, :, :, -T:]).flatten()  # N x T
    SS = np.einsum("ijk->kij", data[1, :, :, -T:]).flatten()  # N x T
    MASK = np.einsum("ijk->kij", mask[:, :, -T:]).flatten()  # N x T

    # x = XX[:, np.newaxis]  # NT x 1
    # y = YY[:, np.newaxis]  # NT x 1
    # t = TT[:, np.newaxis]  # NT x 1

    # a = AA[:, np.newaxis]  # NT x 1
    # s = SS[:, np.newaxis]  # NT x 1

    boundary_x_LB = np.concatenate((x_domain, np.repeat(x_domain[0], y_size)))
    boundary_x_RT = np.concatenate((x_domain, np.repeat(x_domain[-1], y_size)))

    boundary_y_LB = np.concatenate((np.repeat(y_domain[0], x_size), y_domain))
    boundary_y_RT = np.concatenate((np.repeat(y_domain[-1], x_size), y_domain))

    boundary_XX_LB = np.tile(boundary_x_LB.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_XX_RT = np.tile(boundary_x_RT.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_YY_LB = np.tile(boundary_y_LB.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_YY_RT = np.tile(boundary_y_RT.flatten(), T)[:, np.newaxis]  # (x_size + y_size) x T, 1
    # T x (x_size + y_size), 1
    boundary_TT = np.repeat(t_star[-T:], (x_size + y_size))[:, np.newaxis]
    ##########################################
    # Including noise
    if signal_to_noise > 0:
        signal_amp_a = (np.max(AA) - np.min(AA)) / 2.0
        signal_amp_s = (np.max(SS) - np.min(SS)) / 2.0
        sigma_a = signal_amp_a * signal_to_noise
        sigma_s = signal_amp_s * signal_to_noise
    # Observed data
    if shuffle:
        idx_data = np.random.choice(N * T, training_data_size, replace=False)
    else:
        idx_data = list(range(training_data_size))
    # PDE colocations
    if shuffle:
        idx_pde = np.random.choice(N * T, pde_data_size, replace=False)
    else:
        idx_pde = list(range(pde_data_size))
    # Periodic boundary condition
    if shuffle:
        idx_boundary = np.random.choice((x_size + y_size) * T, boundary_data_size, replace=False)
    else:
        idx_boundary = list(range(boundary_data_size))

    # Lower/Upper bounds
    lb, ub = lower_upper_bounds([np.c_[XX, YY, TT]])

    ret = {
        "obs_input": np.c_[XX[idx_data], YY[idx_data], TT[idx_data]],
        "obs_output": np.c_[AA[idx_data], SS[idx_data]],
        "obs_mask": MASK[idx_data],
        "pde": np.c_[XX[idx_pde], YY[idx_pde], TT[idx_pde]],
        "pde_mask": MASK[idx_pde],
        "lb": lb,
        "ub": ub,
    }
    if signal_to_noise > 0:
        ret["obs_output"][:, 0] += sigma_a * np.random.randn(len(idx_data))
        ret["obs_output"][:, 1] += sigma_s * np.random.randn(len(idx_data))

    if with_boundary:
        ret = {
            **ret,
            **{
                "boundary_LB": np.c_[
                    boundary_XX_LB[idx_boundary], boundary_YY_LB[idx_boundary], boundary_TT[idx_boundary]
                ],
                "boundary_RT": np.c_[
                    boundary_XX_RT[idx_boundary], boundary_YY_RT[idx_boundary], boundary_TT[idx_boundary]
                ],
            },
        }

    return ret


def create_dataset_multi_nodes_mask(
    data,
    mask,
    t_star,
    N,
    T,
    L,
    training_data_size,
    pde_data_size,
    signal_to_noise=0,
    shuffle=True,
):
    x_size = data.shape[1]
    y_size = data.shape[2]
    x_domain = L * np.linspace(0, 1, x_size)
    y_domain = L * np.linspace(0, 1, y_size)

    X, Y = np.meshgrid(x_domain, y_domain, sparse=False, indexing="ij")
    XX = np.tile(X.flatten(), T)  # N x T
    YY = np.tile(Y.flatten(), T)  # N x T
    TT = np.repeat(t_star[-T:], N)  # T x N

    UU = np.einsum("cijk->ckij", data[:, :, :, -T:])
    UU = np.array([UU[i, :, :, :].flatten() for i in range(UU.shape[0])])  # c , N x T
    MASK = np.einsum("ijk->kij", mask[:, :, -T:]).flatten()  # N x T

    ##########################################
    # Including noise
    if signal_to_noise > 0:
        signal_amp_u = (np.max(UU) - np.min(UU)) / 2.0
        sigma_u = signal_amp_u * signal_to_noise
    # Observed data
    if shuffle:
        idx_data = np.random.choice(N * T, training_data_size, replace=False)
    else:
        idx_data = list(range(training_data_size))
    # PDE colocations
    if shuffle:
        idx_pde = np.random.choice(N * T, pde_data_size, replace=False)
    else:
        idx_pde = list(range(pde_data_size))

    # Lower/Upper bounds
    lb, ub = lower_upper_bounds([np.c_[XX, YY, TT]])

    ret = {
        "obs_input": np.c_[XX[idx_data], YY[idx_data], TT[idx_data]],
        "obs_output": np.vstack([UU[i, idx_data] for i in range(UU.shape[0])]).T,
        "obs_mask": MASK[idx_data],
        "pde": np.c_[XX[idx_pde], YY[idx_pde], TT[idx_pde]],
        "pde_mask": MASK[idx_pde],
        "lb": lb,
        "ub": ub,
    }
    if signal_to_noise > 0:
        ret["obs_output"] += sigma_u * np.random.randn(len(idx_data))

    return ret


def merge_dict(dict_1, *dicts):
    """Imutable merge of dictionary objects"""
    ret = {}
    all_dicts = [dict_1, *dicts]
    for key in dict_1.keys():
        ret[key] = np.hstack([dict_i[key] for dict_i in all_dicts])

    return ret


def merge_dict_multi_nodes(dict_1, *dicts):
    """Imutable merge of dictionary objects"""
    ret = {}
    all_dicts = [dict_1, *dicts]
    for key in dict_1.keys():
        ret[key] = np.concatenate([dict_i[key] for dict_i in all_dicts])

    return ret


# def merge_dict_multi_nodes(node_names, *dicts):
#     """Imutable merge of dictionary objects"""
#     ret = {}
#     for d in dicts:
#         for key in d.keys():
#             # 1D arrays
#             if len(d[key].shape) == 1:
#                 if key not in ret.keys():
#                     ret[key] = d[key]
#                 else:
#                     ret[key] = np.concatenate((ret[key], d[key]), axis=0)
#             else:  # 2d Arrays
#                 if key not in ret.keys():
#                     d_2_d = d[key]
#                     for i, name in enumerate(node_names):
#                         ret[f"{key}_{name}"] = d_2_d[i]
#                 else:
#                     d_2_d = d[key]
#                     for i, name in enumerate(node_names):
#                         ret[f"{key}_{name}"] = np.concatenate((ret[f"{key}_{name}"], d_2_d[i]), axis=0)

#     return ret


# def merge_dict_multi_nodes2(dict_1, *dicts):
#     """Imutable merge of dictionary objects"""
#     ret = {}
#     all_dicts = [dict_1, *dicts]
#     for key in dict_1.keys():
#         print(key)
#         ret[key] = np.vstack([dict_i[key] for dict_i in all_dicts])

#     return ret


def plot_result(results, param_names=None, start=0, end=-1, node_names=["u", "v"], yscale="log", y_lims=None):
    import matplotlib.pyplot as plt

    def _closing_commands_():
        plt.legend()
        plt.grid()
        plt.xlabel("Iterations")
        plt.yscale(yscale)
        if y_lims is not None:
            plt.ylim(y_lims)
        plt.show()

    _ = plt.figure(figsize=(14, 5))
    plt.title("Training accuracy for observations")
    plt.plot(results["training_obs_accuracy"][start:end], label="accuracy")
    _closing_commands_()

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Real Loss")
        plt.plot(results["loss_total"][start:end], label="total")
        for name in node_names:
            plt.plot(results[f"loss_obs_{name}"][start:end], label=f"Obs {name}")
        for name in node_names:
            plt.plot(results[f"loss_pde_{name}"][start:end], label=f"PDE {name}")
        for key in [k for k in results.keys() if k.startswith("loss_extra_")]:
            plt.plot(results[key][start:end], label=f"{key}")
        _closing_commands_()

    if "loss_non_zero" in results.keys():
        _ = plt.figure(figsize=(14, 5))
        plt.plot(results["loss_non_zero"][start:end], label="loss_non_zero")

        _closing_commands_()

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Regularisd Loss")
        plt.plot(results["loss_regularisd_total"][start:end], label="total")
        if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
            for name in node_names:
                plt.plot(
                    results[f"lambda_obs_{name}"][start:end] * results[f"loss_obs_{name}"][start:end],
                    label=f"Obs {name}",
                )
            for name in node_names:
                plt.plot(
                    results[f"lambda_pde_{name}"][start:end] * results[f"loss_pde_{name}"][start:end],
                    label=f"PDE {name}",
                )
        _closing_commands_()

    if np.any([True if k.startswith("grads_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Gradient Norms")
        for name in node_names:
            plt.plot(results[f"grads_obs_{name}"][start:end], label=f"Grad obs {name}")
        for name in node_names:
            plt.plot(results[f"grads_pde_{name}"][start:end], label=f"Grad PDE {name}")
        if np.any([True if k.startswith("grad_norm_pde_params_") else False for k in results.keys()]):
            for name in node_names:
                plt.plot(results[f"grad_norm_pde_params_{name}"][start:end], label=f"Grad PDE params {name}")
        _closing_commands_()

    if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title(r"$\lambda$s")
        for name in node_names:
            plt.plot(results[f"lambda_obs_{name}"][start:end], label=r"$\lambda$" f" obs {name}")
        for name in node_names:
            plt.plot(results[f"lambda_pde_{name}"][start:end], label=r"$\lambda$" f" PDE {name}")
        if np.any([True if k.startswith("lambda_pde_params_") else False for k in results.keys()]):
            for name in node_names:
                plt.plot(results[f"lambda_pde_params_{name}"][start:end], label=r"$\lambda$" f" PDE params {name}")
        _closing_commands_()

    if param_names is not None:
        _ = plt.figure(figsize=(14, 5))
        plt.title(r"Estimated parameters")
        for name in param_names:
            plt.plot(results[f"{name}"][start:end], label=f"{name}")
        _closing_commands_()


def plot_result_multi_nodes(
    results,
    param_names=None,
    start=0,
    end=-1,
    node_names=["u", "v"],
    yscale="log",
    y_lims=None,
):
    import matplotlib.pyplot as plt

    def _closing_commands_():
        plt.legend()
        plt.grid()
        plt.xlabel("Iterations")
        plt.yscale(yscale)
        if y_lims is not None:
            plt.ylim(y_lims)
        plt.show()

    _ = plt.figure(figsize=(14, 5))
    plt.title("Training accuracy for observations")
    plt.plot(results["training_obs_accuracy"][start:end], label="accuracy")
    _closing_commands_()

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Real Loss")
        plt.plot(results["loss_total"][start:end], label="total")
        for i, name in enumerate(node_names):
            plt.plot(results["loss_obs"][start:end, i], label=f"Obs {name}")
        for i, name in enumerate(node_names):
            plt.plot(results["loss_pde"][start:end, i], label=f"PDE {name}")
        for key in [k for k in results.keys() if k.startswith("loss_extra_")]:
            plt.plot(results[key][start:end], label=f"{key}")

        _closing_commands_()

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Regularisd Loss")
        plt.plot(results["loss_regularisd_total"][start:end], label="total")
        if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
            for i, name in enumerate(node_names):
                plt.plot(
                    results["lambda_obs"][start:end, i] * results["loss_obs"][start:end, i],
                    label=f"Obs {name}",
                )
            for i, name in enumerate(node_names):
                plt.plot(
                    results["lambda_pde"][start:end, i] * results["loss_pde"][start:end, i],
                    label=f"PDE {name}",
                )
        _closing_commands_()

    if np.any([True if k.startswith("grads_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title("Gradient Norms")
        for i, name in enumerate(node_names):
            plt.plot(results["grads_obs"][start:end, i], label=f"Grad obs {name}")
        for i, name in enumerate(node_names):
            plt.plot(results["grads_pde"][start:end, i], label=f"Grad PDE {name}")
        _closing_commands_()

    if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
        _ = plt.figure(figsize=(14, 5))
        plt.title(r"$\lambda$s")
        for i, name in enumerate(node_names):
            plt.plot(results["lambda_obs"][start:end, i], label=r"$\lambda$" f" obs {name}")
        for i, name in enumerate(node_names):
            plt.plot(results["lambda_pde"][start:end, i], label=r"$\lambda$" f" PDE {name}")
        _closing_commands_()

    if param_names is not None:
        _ = plt.figure(figsize=(14, 5))
        plt.title(r"Estimated parameters")
        for name in param_names:
            plt.plot(results[f"{name}"][start:end], label=f"{name}")
        _closing_commands_()


# This code is originally from TF Source code
class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    ```
    Args:
        factor: factor by which the learning rate will be reduced.
          `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
          lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(
        self, optimizer, monitor="training_obs_accuracy", factor=0.1, patience=10, min_delta=1e-4, cooldown=0, min_lr=0
    ):

        if factor >= 1.0:
            raise ValueError(f"ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}")
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.monitor = monitor
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.best = np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_epoch_end(self, epoch, samples):
        current = samples[self.monitor][epoch]
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.optimizer.lr.numpy()
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    print("========================================================")
                    print("Update learining rate " f"from {old_lr:.3e} to {new_lr:.3e}")
                    print("========================================================")
                    self.optimizer.lr.assign(new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
