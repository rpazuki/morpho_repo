from collections import namedtuple
import numpy as np
from scipy.optimize import minimize


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
        "dt_arr",
        "parameters",
        "steady_state_func",
        "perturbation_size",
        "kinetic_func",
        "tol",
        "sample_parameters",
        "sample_parameters_num",
        "sample_parameters_std",
        "same_init",
        "c0",
    ],
    defaults=[
        "Brusselator",
        (128, 128),
        (2, 2),
        (0.002, 0.04),
        # 0.05, 0, 128, 64,
        0.001,
        0,
        128,
        128 + 1,
        None,
        {"A": 2, "B": 3},
        None,
        0.1,
        None,
        1e-3,
        True,
        30,
        (0.5, 0.5),
        False,
        None,
    ],
)


def diffusion(n, c):
    dc = np.zeros_like(c)
    for i in range(n[0]):
        for j in range(n[1]):
            # Periodic boundary condition
            i_prev = (i - 1) % n[0]
            i_next = (i + 1) % n[0]

            j_prev = (j - 1) % n[1]
            j_next = (j + 1) % n[1]
            dc[i, j] = c[i_prev, j] + c[i_next, j] + c[i, j_prev] + c[i, j_next] - 4.0 * c[i, j]
    return dc


def second_order_derivatives(n, c):
    dc_xx = np.zeros_like(c)
    dc_yy = np.zeros_like(c)
    for i in range(n[0]):
        for j in range(n[1]):
            # Periodic boundary condition
            i_prev = (i - 1) % n[0]
            i_next = (i + 1) % n[0]

            j_prev = (j - 1) % n[1]
            j_next = (j + 1) % n[1]
            dc_xx[i, j] = c[i_prev, j] + c[i_next, j] - 2.0 * c[i, j]
            dc_yy[i, j] = c[i, j_prev] + c[i, j_next] - 2.0 * c[i, j]
    return dc_xx, dc_yy


def first_order_derivatives(n, c, forward=True):
    dc_x = np.zeros_like(c)
    dc_y = np.zeros_like(c)
    for i in range(n[0]):
        for j in range(n[1]):
            # Periodic boundary condition
            i_prev = (i - 1) % n[0]
            i_next = (i + 1) % n[0]

            j_prev = (j - 1) % n[1]
            j_next = (j + 1) % n[1]
            if forward:
                dc_x[i, j] = c[i_next, j] - c[i, j]
                dc_y[i, j] = c[i, j_next] - c[i, j]
            else:
                dc_x[i, j] = c[i_prev, j] - c[i, j]
                dc_y[i, j] = c[i, j_prev] - c[i, j]
    return dc_x, dc_y


def lower_upper_bounds(inputs_2D):
    """Find the lower and upper bounds of inputs

    inputs_2D: a 2d ndarray that its maxs and mins are calcuated
               along axis 0.
    """

    lb = np.amin(inputs_2D, 0)
    ub = np.amax(inputs_2D, 0)
    return lb, ub


def lower_upper_bounds_old(inputs_of_inputs):
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
    diffusion=None,
    derivatives=None,
    idx_data=None,
):
    x_size = data.shape[1]
    y_size = data.shape[2]
    x_domain = L * np.linspace(0, 1, x_size)
    y_domain = L * np.linspace(0, 1, y_size)

    X, Y = np.meshgrid(x_domain, y_domain, sparse=False, indexing="ij")
    XX = np.tile(X.flatten(), T)  # N x T
    YY = np.tile(Y.flatten(), T)  # N x T
    TT = np.repeat(t_star[-T:], N)  # T x N

    UU = np.einsum("ijk->kij", data[0, :, :, -T:]).flatten()  # N x T
    VV = np.einsum("ijk->kij", data[1, :, :, -T:]).flatten()  # N x T

    if diffusion is not None:
        DUU = np.einsum("ijk->kij", diffusion[0, :, :, -T:]).flatten()  # N x T
        DVV = np.einsum("ijk->kij", diffusion[1, :, :, -T:]).flatten()

        X1, Y1 = np.meshgrid(
            np.r_[x_domain[1:], x_domain[0]], np.r_[y_domain[1:], y_domain[0]], sparse=False, indexing="ij"
        )
        XX_right = np.tile(X1.flatten(), T)  # N x T
        YY_bottom = np.tile(Y1.flatten(), T)  # N x T

        X2, Y2 = np.meshgrid(
            np.r_[x_domain[-1], x_domain[:-1]], np.r_[y_domain[-1], y_domain[:-1]], sparse=False, indexing="ij"
        )
        XX_left = np.tile(X2.flatten(), T)  # N x T
        YY_up = np.tile(Y2.flatten(), T)  # N x T
    if derivatives is not None:
        dd_us = np.array([np.einsum("ijk->kij", d[0, :, :, -T:]).flatten() for d in derivatives])
        dd_vs = np.array([np.einsum("ijk->kij", d[1, :, :, -T:]).flatten() for d in derivatives])

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
        signal_amp_a = (np.max(UU) - np.min(UU)) / 2.0
        signal_amp_s = (np.max(VV) - np.min(VV)) / 2.0
        sigma_a = signal_amp_a * signal_to_noise
        sigma_s = signal_amp_s * signal_to_noise
    # Observed data
    if idx_data is None:
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
    lb, ub = lower_upper_bounds(np.c_[XX, YY, TT])
    output_lb, output_ub = lower_upper_bounds(np.c_[UU, VV])
    if derivatives is not None:
        derivatives_u_lb, derivatives_u_ub = lower_upper_bounds(dd_us.T)

        derivatives_v_lb, derivatives_v_ub = lower_upper_bounds(dd_vs.T)
    ret = {
        "obs_input": np.c_[XX[idx_data], YY[idx_data], TT[idx_data]],
        "obs_output": np.c_[UU[idx_data], VV[idx_data]],
        "lb": lb,
        "ub": ub,
        "output_lb": output_lb,
        "output_ub": output_ub,
        "idx_data": idx_data,
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
                "idx_boundary": idx_boundary,
            },
        }
    if derivatives is not None:
        ret = {
            **ret,
            **{
                "der_u": dd_us[:, idx_data],
                "der_v": dd_vs[:, idx_data],
                "derivatives_u_lb": derivatives_u_lb,
                "derivatives_u_ub": derivatives_u_ub,
                "derivatives_v_lb": derivatives_v_lb,
                "derivatives_v_ub": derivatives_v_ub,
            },
        }

    if diffusion is not None:
        ret = {
            **ret,
            **{
                "diff_input": np.c_[
                    XX[idx_data],
                    YY[idx_data],
                    TT[idx_data],
                    XX_left[idx_data],
                    XX_right[idx_data],
                    YY_bottom[idx_data],
                    YY_up[idx_data],
                ],
                "diff_output": np.c_[DUU[idx_data], DVV[idx_data]],
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
    lb, ub = lower_upper_bounds(np.c_[XX, YY, TT])

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
    lb, ub = lower_upper_bounds(np.c_[XX, YY, TT])

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
    for key in ret.keys():
        ret[key] = ret[key].T
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
def plot_result(
    results,
    model,
    param_names=None,
    start=0,
    end=-1,
    skip=1,
    yscale="log",
    y_lims=None,
    figsize=(14, 5),
    file_name=None,
):
    import matplotlib.pyplot as plt

    def _closing_commands_(plot_name=""):
        plt.legend()
        plt.grid()
        plt.xlabel("Iterations")
        plt.yscale(yscale)
        if y_lims is not None:
            plt.ylim(y_lims)
        if file_name is not None:
            plt.savefig((f"{file_name}_{plot_name}.png"), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # _ = plt.figure(figsize=figsize)
    # plt.title("Training accuracy for observations")
    # plt.plot(results["training_obs_accuracy"][start:end], label="accuracy")
    # _closing_commands_("training_accuracy")
    _ = plt.figure(figsize=figsize)
    plt.title("Total Loss")
    plt.plot(results["loss_total"][start:end:skip], label="Real")
    plt.plot(results["loss_regularisd_total"][start:end:skip], label="Regularisd")
    _closing_commands_("Total losses")

    for i, loss in enumerate(model.losses):
        _ = plt.figure(figsize=figsize)
        plt.title(loss.name)
        ts = results[f"{loss.name}_values"][start:end:skip, :]
        for j in range(ts.shape[1]):
            plt.plot(ts[:, j], label=f"({j+1}) {loss.residual_ret_names[j]}")
        _closing_commands_("Losses")

    for i, loss in enumerate(model.no_input_losses):
        _ = plt.figure(figsize=figsize)
        plt.title(loss.name)
        ts = results[f"{loss.name}_values"][start:end:skip, :]
        for j in range(ts.shape[1]):
            plt.plot(ts[:, j], label=f"({j+1}) {loss.residual_ret_names[j]}")
        _closing_commands_("No input losses")

    if "lambdas" in results.keys():
        _ = plt.figure(figsize=figsize)
        plt.title(r"$\lambda$")
        ts = results["lambdas"][start:end:skip, :]
        for j in range(ts.shape[1]):
            plt.plot(ts[:, j], label=r"$\lambda_{" f"{j+1}" r"}$")
        _closing_commands_("Lambdas")

    if "grads" in results.keys():
        _ = plt.figure(figsize=figsize)
        plt.title("Gradients")
        ts = results["grads"][start:end:skip, :]
        for j in range(ts.shape[1]):
            plt.plot(ts[:, j], label=f"{j+1}")
        _closing_commands_("Gradients")

    if param_names is not None:
        _ = plt.figure(figsize=figsize)
        plt.title(r"Estimated parameters")
        for name in param_names:
            plt.plot(results[f"{name}"][start:end:skip], label=f"{name}")
        _closing_commands_("parameters")


def plot_result_old(
    results,
    param_names=None,
    start=0,
    end=-1,
    node_names=["u", "v"],
    yscale="log",
    y_lims=None,
    figsize=(14, 5),
    file_name=None,
):
    import matplotlib.pyplot as plt

    def _closing_commands_(plot_name=""):
        plt.legend()
        plt.grid()
        plt.xlabel("Iterations")
        plt.yscale(yscale)
        if y_lims is not None:
            plt.ylim(y_lims)
        if file_name is not None:
            plt.savefig((f"{file_name}_{plot_name}.png"), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    _ = plt.figure(figsize=figsize)
    plt.title("Training accuracy for observations")
    plt.plot(results["training_obs_accuracy"][start:end], label="accuracy")
    _closing_commands_("training_accuracy")

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=figsize)
        plt.title("Real Loss")
        plt.plot(results["loss_total"][start:end], label="total")
        for name in node_names:
            plt.plot(results[f"loss_obs_{name}"][start:end], label=f"Obs {name}")
        for name in node_names:
            plt.plot(results[f"loss_pde_{name}"][start:end], label=f"PDE {name}")
        for key in [k for k in results.keys() if k.startswith("loss_extra_")]:
            plt.plot(results[key][start:end], label=f"{key}")
        _closing_commands_("losses")

    if "loss_non_zero" in results.keys():
        _ = plt.figure(figsize=figsize)
        plt.plot(results["loss_non_zero"][start:end], label="loss_non_zero")

        _closing_commands_("non_zero_loss")

    if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
        _ = plt.figure(figsize=figsize)
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
        _closing_commands_("regularisd_losses")

    if np.any([True if k.startswith("grads_") else False for k in results.keys()]):
        _ = plt.figure(figsize=figsize)
        plt.title("Gradient Norms")
        for name in node_names:
            plt.plot(results[f"grads_obs_{name}"][start:end], label=f"Grad obs {name}")
        for name in node_names:
            plt.plot(results[f"grads_pde_{name}"][start:end], label=f"Grad PDE {name}")
        if np.any([True if k.startswith("grad_norm_pde_params_") else False for k in results.keys()]):
            for name in node_names:
                plt.plot(results[f"grad_norm_pde_params_{name}"][start:end], label=f"Grad PDE params {name}")
        _closing_commands_("gradient_norms")

    if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
        _ = plt.figure(figsize=figsize)
        plt.title(r"$\lambda$s")
        for name in node_names:
            plt.plot(results[f"lambda_obs_{name}"][start:end], label=r"$\lambda$" f" obs {name}")
        for name in node_names:
            plt.plot(results[f"lambda_pde_{name}"][start:end], label=r"$\lambda$" f" PDE {name}")
        if np.any([True if k.startswith("lambda_pde_params_") else False for k in results.keys()]):
            for name in node_names:
                plt.plot(results[f"lambda_pde_params_{name}"][start:end], label=r"$\lambda$" f" PDE params {name}")
        _closing_commands_("lambdas")

    if param_names is not None:
        _ = plt.figure(figsize=figsize)
        plt.title(r"Estimated parameters")
        for name in param_names:
            plt.plot(results[f"{name}"][start:end], label=f"{name}")
        _closing_commands_("parameters")


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
