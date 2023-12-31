import numpy as np
import pandas as pd


df = pd.read_csv("../../Three_nodes_models/circuit_3954/with_diffusions_second_search/df_network_analysis_full_topology_with_estimates.csv")
df["adj_tup"] = df["adj_tup"].apply(lambda x: eval(f"tuple({x})"))
df["Adj"] = df["adj_tup"].apply(lambda x: np.array(x).reshape((3,3)))

adj=np.array([[1, 1, -1], [-1, 0, -1], [0, -1, 1]])
subnet_list = [g[1] for g in df.groupby("adj_tup") if g[0] == tuple(adj.flatten())]
subnet_df = None
if len(subnet_list) == 0:
    print("================================")
    print("There is no adjacancy matrix as: ", adj)
    print("================================")
else:
    subnet_df = subnet_list[0]
    
def load_dataset(path):
    with open(f"../../Three_nodes_models/circuit_3954/{path}", "rb") as f:
        k_max, params, res = np.load(f, allow_pickle=True)
    (n_val, 
     b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
     b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
     b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val) = params
    params = {
              'D_A':0.01,
              'D_B':1.0,
              'n':n_val, 
              'b_A':b_A_val, 
              'mu_A':mu_A_val, 
              'V_A':V_A_val,
              'K_AA':K_AA_val, 
              'K_AB':K_AB_val,  
              'K_AC':K_AC_val,
              'b_B':b_B_val, 
              'mu_B':mu_B_val, 
              'V_B':V_B_val,
              'K_BA':K_BA_val, 
              'K_BC':K_BC_val,  
              'b_C':b_C_val, 
              'mu_C':mu_C_val, 
              'V_C':V_C_val,
              'K_CB':K_CB_val, 
              'K_CC':K_CC_val
             }
           
    return (params, res, k_max)
def to(arr):
    return arr.reshape(128, 128) 

def reshape(arr, steps=1):
    T = arr.shape[0]
    ret = np.array([
        [to(arr[i, 0, :]), to(arr[i, 1, :]), to(arr[i, 2, :])]
        for i in range(T-steps, T)
    ])
    return np.einsum("tcxy -> cxyt", ret)

def rmse(arr1, arr2):
    return np.sqrt(np.mean((arr1-arr2)**2))

def grad_diff(c_c):
    dc = np.zeros_like(c_c)
    for i in range(c_c.shape[0]):
        for j in range(c_c.shape[1]):
            # Periodic boundary condition
            i_prev = (i - 1) % c_c.shape[0]
            i_next = (i + 1) % c_c.shape[0]

            j_prev = (j - 1) % c_c.shape[1]
            j_next = (j + 1) % c_c.shape[1]
            dc[i, j] = (
                    c_c[i_prev, j]
                    + c_c[i_next, j]
                    + c_c[i, j_prev]
                    + c_c[i, j_next]
                    - 4.0 * c_c[i, j]
                ) 
    return dc

def directional_linspace(origin, direction, steps):
    return origin[np.newaxis, ...] + steps[...,np.newaxis]*direction[np.newaxis, ...]

def MCMC(original_params, L2_func, max_steps = 1000000):
    current_point = np.random.normal(original_params, scale=original_params*1)
    current_point[current_point <= 0] = 1e-5
    
    L2_samples = np.zeros(max_steps)
    point_samples = np.zeros((max_steps,current_point.shape[0]))
    L2_samples[0] = L2_func(*current_point)
    point_samples[0, :] = current_point 
    for step in range(1, max_steps):
        next_point = np.random.normal(original_params, scale=original_params*.1)
        next_point[next_point <= 0] = 1e-5
        next_L2_sample = L2_func(*next_point)
        if min(1, L2_samples[step-1]/next_L2_sample) > np.random.rand():
            current_point = next_point
            L2_samples[step] = next_L2_sample
        else:
            L2_samples[step] = L2_samples[step-1]
        point_samples[step, :] = current_point 
    return L2_samples, point_samples 