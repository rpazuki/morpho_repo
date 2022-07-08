import copy
import numpy as np
#  import matplotlib.pyplot as plt


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
    indices_batch_size = [size + (batch_size // size) * (batch_size % size)
                          for size in indices_batch_size]

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
            ends = [ns[i + 1] if end != ns[i + 1]
                    else end for i, end in enumerate(ns_remain)]
        # step's indices
        yield [indices[0][n1_start:n1_end]] + \
              [indices[i + 1][star:end]
                  for i, (star, end) in enumerate(zip(starts, ends))]


def create_dataset(data,
                   t_star,
                   N,
                   T,
                   L,
                   training_data_size,
                   pde_data_size,
                   boundary_data_size,
                   with_boundary=True,
                   signal_to_noise=0):
    x_size = data.shape[1]
    y_size = data.shape[2]
    x_domain = L * np.linspace(0, 1, x_size)
    y_domain = L * np.linspace(0, 1, y_size)

    X, Y = np.meshgrid(x_domain, y_domain, sparse=False, indexing='ij')
    XX = np.tile(X.flatten(), T)  # N x T
    YY = np.tile(Y.flatten(), T)  # N x T
    TT = np.repeat(t_star[-T:], N)  # T x N

    AA = np.einsum('ijk->kij', data[0, :, :, -T:]).flatten()  # N x T
    SS = np.einsum('ijk->kij', data[1, :, :, -T:]).flatten()  # N x T

    x = XX[:, np.newaxis]  # NT x 1
    y = YY[:, np.newaxis]  # NT x 1
    t = TT[:, np.newaxis]  # NT x 1

    a = AA[:, np.newaxis]  # NT x 1
    s = SS[:, np.newaxis]  # NT x 1

    boundary_x_LB = np.concatenate((x_domain,
                                    np.repeat(x_domain[0], y_size)))
    boundary_x_RT = np.concatenate((x_domain,
                                    np.repeat(x_domain[-1], y_size)))

    boundary_y_LB = np.concatenate((np.repeat(y_domain[0], x_size),
                                    y_domain))
    boundary_y_RT = np.concatenate((np.repeat(y_domain[-1], x_size),
                                    y_domain))

    boundary_XX_LB = np.tile(boundary_x_LB.flatten(), T)[
        :, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_XX_RT = np.tile(boundary_x_RT.flatten(), T)[
        :, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_YY_LB = np.tile(boundary_y_LB.flatten(), T)[
        :, np.newaxis]  # (x_size + y_size) x T, 1
    boundary_YY_RT = np.tile(boundary_y_RT.flatten(), T)[
        :, np.newaxis]  # (x_size + y_size) x T, 1
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
    idx_data = np.random.choice(N * T, training_data_size, replace=False)
    # PDE colocations
    idx_pde = np.random.choice(N * T, pde_data_size, replace=False)
    # Periodic boundary condition
    idx_boundary = np.random.choice(
        (x_size + y_size) * T, boundary_data_size, replace=False)

    # Lower/Upper bounds
    lb, ub = lower_upper_bounds([np.c_[XX, YY, TT]])

    ret = {'obs_input': np.c_[XX[idx_data], YY[idx_data], TT[idx_data]],
           'obs_output': np.c_[AA[idx_data], SS[idx_data]],
           'pde': np.c_[XX[idx_pde], YY[idx_pde], TT[idx_pde]],
           'lb': lb,
           'ub': ub}

    if signal_to_noise > 0:
        ret['obs_output'][0] += sigma_a * \
            np.random.randn(len(idx_data), a.shape[1])
        ret['obs_output'][1] += sigma_s * \
            np.random.randn(len(idx_data), s.shape[1])

    if with_boundary:
        ret = {**ret,
               **{'boundary_LB': np.c_[boundary_XX_LB[idx_boundary],
                                       boundary_YY_LB[idx_boundary],
                                       boundary_TT[idx_boundary]],
                  'boundary_RT': np.c_[boundary_XX_RT[idx_boundary],
                                       boundary_YY_RT[idx_boundary],
                                       boundary_TT[idx_boundary]]}
               }

    return ret


def merge_dict(dict_1, * dicts):
    """Imutable merge of dictionary objects"""
    ret = {}
    all_dicts = [dict_1, * dicts]
    for key in dict_1.keys():
        ret[key] = np.hstack([dict_i[key] for dict_i in all_dicts])

    return ret


# def plot_result(results, start=0, end=-1, node_names=['u', 'v'], yscale='log', y_lims=None):

#     def _closing_commands_():
#         plt.legend()
#         plt.grid()
#         plt.yscale(yscale)
#         if y_lims is not None:
#             plt.ylim(y_lims)
#         plt.show()

#     _ = plt.figure(figsize=(14, 5))
#     plt.title("Training accuracy for observations")
#     plt.plot(results["training_obs_accuracy"][start:end], label="accuracy")
#     _closing_commands_()

#     if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
#         _ = plt.figure(figsize=(14, 5))
#         plt.title("Real Loss")
#         plt.plot(results["loss_total"][start:end], label="total")
#         for name in node_names:
#             plt.plot(results[f"loss_obs_{name}"]
#                      [start:end], label=f"Obs {name}")
#         for name in node_names:
#             plt.plot(results[f"loss_pde_{name}"]
#                      [start:end], label=f"PDE {name}")
#         for key in [k for k in results.keys() if k.startswith("loss_extra_")]:
#             plt.plot(results[key][start:end], label=f"{key}")

#         _closing_commands_()

#     if np.any([True if k.startswith("loss_") else False for k in results.keys()]):
#         _ = plt.figure(figsize=(14, 5))
#         plt.title("Regularisd Loss")
#         plt.plot(results["loss_regularisd_total"][start:end], label="total")
#         if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
#             for name in node_names:
#                 plt.plot(results[f"lambda_obs_{name}"][start:end] * results[f"loss_obs_{name}"][start:end],
#                          label=f"Obs {name}")
#             for name in node_names:
#                 plt.plot(results[f"lambda_pde_{name}"][start:end] * results[f"loss_pde_{name}"][start:end],
#                          label=f"PDE {name}")
#         _closing_commands_()

#     if np.any([True if k.startswith("grads_") else False for k in results.keys()]):
#         _ = plt.figure(figsize=(14, 5))
#         plt.title("Gradient Norms")
#         for name in node_names:
#             plt.plot(results[f"grads_obs_{name}"]
#                      [start:end], label=f"Grad obs {name}")
#         for name in node_names:
#             plt.plot(results[f"grads_pde_{name}"]
#                      [start:end], label=f"Grad PDE {name}")
#         _closing_commands_()

#     if np.any([True if k.startswith("lambda_") else False for k in results.keys()]):
#         _ = plt.figure(figsize=(14, 5))
#         plt.title(r"$\lambda$s")
#         for name in node_names:
#             plt.plot(
#                 results[f"lambda_obs_{name}"][start:end], label=r"$\lambda$" f" obs {name}")
#         for name in node_names:
#             plt.plot(
#                 results[f"lambda_pde_{name}"][start:end], label=r"$\lambda$" f" PDE {name}")
#         _closing_commands_()
