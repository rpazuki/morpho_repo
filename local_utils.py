from itertools import combinations
from collections import namedtuple
import os
from pathlib import Path
import numpy as np
import sys
sys.path.append(f"{Path.home()}/morpho_repo/turing_codebase")
from turing.utils import Simulation, create_dataset, second_order_derivatives, diffusion
import matplotlib.pyplot as plt

Pattern = namedtuple('Pattern', ['simulation', 't_star', 'x_size', 'y_size', 
                                 'c', 'c_xx', 'c_yy', 'c_t',
                                 'c_domain', 'c_xx_domain', 'c_yy_domain', 'c_t_domain',
                                 'params', 'u_params', 'v_params',
                                 'block_x', 'block_y', 'dataset'])     
def load(sim_name, 
         x_slice,
         y_slice,
         data_time_start=-3, 
         data_time_len=2,
         data_time_step=1,
         verbose=False
        ):
    data_path = os.path.abspath(f"{Path.home()}/test/outputs_Koch_Meinhardt_100_by_100/{sim_name}/{sim_name}.npy")
    with open(data_path, 'rb') as f:
        data = np.load(f)

    data_path = os.path.abspath(f"{Path.home()}/test/outputs_Koch_Meinhardt_100_by_100/{sim_name}/simulation.txt")
    with open(data_path, 'r') as f:
        simulation = eval(f.read())
        
    t_star = np.linspace(simulation.t_start, 
                     simulation.t_end, 
                     simulation.t_steps)
    
    if verbose:
        print(f"Initial dataset shape was: {data.shape}.")
    
    L = simulation.L[0]
    x_size = data.shape[1]
    y_size = data.shape[2]
    N = x_size*y_size    
    dxdy = x_size*y_size/L**2
    
       
    if data_time_start+data_time_len+1 < 0:
        time_slice=slice(data_time_start, data_time_start+data_time_len+1, data_time_step)
    else:
        time_slice=slice(data_time_start, data.shape[3], data_time_step)
    data_time_slice=slice(data_time_start, data_time_start+data_time_len, data_time_step)
    
    data_time = data[:, :, :, time_slice].copy()
    data = data[:, :, :, data_time_slice] 
    
    if verbose:
        print(f"It reduced to: {data.shape}.")
    
    t_star_time = t_star[time_slice].copy()
    t_star = t_star[data_time_slice]
    
    assert len(t_star) > 1, f"There must be at least two time steps. There are '{len(t_star)}'."
    
    c_xx =  np.array([[ dxdy  * second_order_derivatives((x_size,y_size), data[c, ..., t])[0] 
                        for t in range(data.shape[3])]
                        for c in range(data.shape[0])
                       ])
    c_xx = np.einsum("ctij -> cijt", c_xx)
    c_yy = np.array([[ dxdy * second_order_derivatives((x_size,y_size), data[c, ..., t])[1] 
                        for t in range(data.shape[3])]
                        for c in range(data.shape[0])
                       ])
    c_yy = np.einsum("ctij -> cijt", c_yy)

    c_t = np.array([[(data_time[c, ..., t+1] -  data_time[c, ..., t])/(t_star_time[t+1] - t_star_time[t])
                        for t in range(data.shape[3])]
                        for c in range(data.shape[0])
                       ])
    c_t = np.einsum("ctij -> cijt", c_t)
    
    diff_Y = np.array([[dxdy * diffusion((x_size,y_size), data[c, ..., t]) 
                        for t in range(data.shape[3])]
                        for c in range(data.shape[0])
                       ])
    diff_Y = np.einsum("ctij -> cijt", diff_Y)
        
    if verbose:
        print(f"And its X-Y size reduced to: {data[:, x_slice, y_slice, :].shape}.")
        
        
    
        
    params = {"D_u": simulation.Ds[0],
          "D_v": simulation.Ds[1],
          "kappa_u": simulation.parameters["kappa_u"],
          "rho_u": simulation.parameters["rho_u"],
          "mu_u": simulation.parameters["mu_u"],
          "sigma_u": simulation.parameters["sigma_u"],
          "rho_v": simulation.parameters["rho_v"],
          "sigma_v": simulation.parameters["sigma_v"]}
    
    u_params = {k: params[k] for k in ("D_u", "kappa_u", "rho_u", "mu_u", "sigma_u")}
    v_params = {k: params[k] for k in ("D_v", "kappa_u", "rho_v", "sigma_v")}
    
    c_restricted = data[:, x_slice, y_slice, :].copy()
    c_xx_restricted = c_xx[:, x_slice, y_slice, :].copy()
    c_yy_restricted = c_yy[:, x_slice, y_slice, :].copy() 
    c_t_restricted = c_t[:, x_slice, y_slice, :].copy()
    T = t_star.shape[0]
    indices_all = np.arange(0, x_size*y_size*T, 1).reshape((T, x_size,y_size))
    indices_sub = indices_all[:, x_slice, y_slice]
    block_x = indices_sub.shape[1]
    block_y = indices_sub.shape[2]
    block_size = indices_sub.shape[1]*indices_sub.shape[2]

    model_params = {'training_data_size': x_size*y_size*T,
                    'pde_data_size': x_size*y_size*T,
                    'boundary_data_size':((x_size + y_size)*T),
                    'diffusion': diff_Y,
                    'derivatives':[c_xx, c_yy, c_t], 
                    'signal_to_noise':0.0,
                    'shuffle':False,
                    'idx_data':indices_sub.flatten()}

    dataset = create_dataset(data, t_star, N, T, L, **model_params)
    
            
    pattern = Pattern(simulation, t_star, x_size, y_size, 
                      c_restricted, c_xx_restricted, c_yy_restricted, c_t_restricted, 
                      data, c_xx, c_yy, c_t,
                      params, u_params, v_params,
                      block_x, block_y, dataset)
        
    return pattern


########################################
# Lines
def Euc_L(vec):
    return np.sqrt(np.sum(vec**2))

def normalise(vec):
    return vec/(Euc_L(vec) + 1e-100)

def creat_line(x_0, s):
    def line(alpha):
        return x_0 + alpha * s
    return line

##########################################
# Ploting Helpers
def plot_n_im(arrays, titles=None, add_colorbar=True, figsize=(12, 8), fraction=0.15, shrink=1.0):
    cols = len(arrays)
    if titles is not None:
        assert len(titles) == cols, f"Titles len'{len(titles)}' is not equal to arrays '{col}'."
    plt.figure(figsize=figsize)
    for i in range(cols):
        ax = plt.subplot(1, cols, i+1)
        if titles is not None:
            ax.set_title(titles[i])
        img = plt.imshow(arrays[i])
        if add_colorbar:
            plt.colorbar(img, fraction=fraction, shrink=shrink)
