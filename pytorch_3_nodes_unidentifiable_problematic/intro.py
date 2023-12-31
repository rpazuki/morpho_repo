import numpy as np
from scipy.optimize import fsolve
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt

# Use this class to convert dict to objet
class Struct(dict):
    def __init__(self, **entries):
        super().__init__(entries)
        self.__dict__.update(entries)
        
def to(arr):
    return arr.reshape(128, 128) 

def  init_values(model, A_star, B_star, C_star, scale_factor=.1):
    A = np.random.normal(scale=A_star*scale_factor, size=(model.Ix*model.Jy))
    A += A_star
    #
    B = np.random.normal(scale=B_star*scale_factor, size=(model.Ix*model.Jy))
    B += B_star
    #
    C = np.random.normal(scale=C_star*scale_factor, size=(model.Ix*model.Jy))
    C += C_star
    return (A, B, C)

def find_roots(kinetics, verbose=False):
    def k(args):
        A,B,C = args
        return kinetics(A,B,C)
    func = k
    #
    roots_res = []
    for init_vals in product([.1,-.1], repeat=3):        
        roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
        (a_1, b_1, c_1) = func(roots)
        if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
            roots_res.append(roots)
            return roots_res
    for init_vals in product([1,-1], repeat=3):        
        roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
        (a_1, b_1, c_1) = func(roots)
        if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
            roots_res.append(roots)
            return roots_res

    for init_vals in product([10,-10], repeat=3):        
        roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
        (a_1, b_1, c_1) = func(roots)
        if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
            roots_res.append(roots)
            return roots_res

    for init_vals in product([100,-100], repeat=3):        
        roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
        (a_1, b_1, c_1) = func(roots)
        if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
            roots_res.append(roots)
            return roots_res

    if len(roots_res) == 0:
        if verbose:
            print(msg)
            print("roots:", roots)
            print("values", a_1, b_1, c_1)
        return roots_res 
    else:
        roots_res2 = [roots_res[0]]
        for item in roots_res[1:]:
            if np.any(item < 0):
                continue
            if not np.any([np.isclose(item2, item) for item2 in roots_res2]):
                roots_res2.append(item)
        return roots_res2 

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
def find_act_bound(func, ranges, level, args=(), is_lower=True):
    values_indices = np.nonzero(func(ranges, *args)  < level) if is_lower else np.nonzero(func(ranges, *args) > level)    
    return ranges[values_indices][-1] if is_lower else ranges[values_indices][0]

def find_inh_bound(func, ranges, level, args=(), is_lower=True):
    values_indices = np.nonzero(func(ranges, *args)  > level) if is_lower else np.nonzero(func(ranges, *args) < level)    
    return ranges[values_indices][-1] if is_lower else ranges[values_indices][0]

def find_bounds(func, ranges, level, args=(), is_act=True):
    if is_act:
        lower_bound = find_act_bound(func, ranges, level, args)
        upper_bound = find_act_bound(func, ranges, 1-level, args, is_lower=False)
    else:
        lower_bound = find_inh_bound(func, ranges, 1-level, args, is_lower=False)
        upper_bound = find_inh_bound(func, ranges, level, args)
    return (lower_bound, upper_bound)

def act(X, K, n):
    """Activatrion"""
    return X**n / (X**n + K**n)


def inh(X, K, n):
    """Inhibition"""
    return K**n / (X**n + K**n)

def creat_topology(params, kinetics, A, B, C, threshold = 0.001):
    (D_A, D_B, n, 
     b_A, mu_A, V_A, K_AA, K_AB, K_AC,
     b_B, mu_B, V_B, K_BA, K_BC,
     b_C, mu_C, V_C, K_CB, K_CC) = tuple(params.values())    
    ###############################################################
    A_act_AA = lambda A, K_AA: act(A, K_AA, 4)
    A_inh_BA = lambda B, K_BA: inh(B, K_BA, 4)
    A_prod = lambda A, K_AA, B, K_BA: A_act_BA(A, K_AA)*A_act_BA(B, K_BA)

    B_act_AB = lambda A, K_AB: act(A, K_AB, 4)
    B_inh_CB = lambda C, K_CB: inh(C, K_CB, 4)
    B_prod = lambda A, K_AB, C, K_CB: B_act_AB(A, K_AB)*B_inh_CB(C, K_CB)

    C_inh_AC = lambda A, K_AC: inh(A, K_AC, 4)
    C_inh_BC = lambda B, K_BC: inh(B, K_BC, 4)
    C_act_CC = lambda C, K_CC: act(C, K_CC, 4)
    C_prod = lambda A, K_AC, B, K_BC, C, K_CC: C_inh_AC(A, K_AC)*C_inh_BC(B, K_BC)*C_act_CC(C, K_CC)
    ###############################################################
    def analysis_act(min_, avg_, max_, lb, up):
        if max_ < lb:
            return "ZERO"
        elif min_ > up:
            return "ONE"
        else:
            return "Active"
        
    def analysis_inh(min_, avg_, max_, lb, up):
        if max_ < lb:
            return "ONE"
        elif min_ > up:
            return "ZERO"
        else:
            return "Active"       
    xs = np.linspace(0, 1000, 100000)
    # circuit 3954 adjacency matrix    
    Adj=np.array([[1, 1, -1], [-1, 0, -1], [0, -1, 1]])    
    ###############################################################
    min_A, min_B, min_C = np.min(A), np.min(B), np.min(C)
    avg_A, avg_B, avg_C = np.mean(A), np.mean(B), np.mean(C)
    max_A, max_B, max_C = np.max(A), np.max(B), np.max(C)
    std_A, std_B, std_C = np.std(A), np.std(B), np.std(C)
    
    lb_AA, ub_AA = find_bounds(A_act_AA, xs, threshold, (K_AA,), True)
    analysis_AA = analysis_act(min_A, avg_A, max_A, lb_AA, ub_AA)
    if analysis_AA == "ONE":
        Adj[0,0]=0
        
    
    lb_BA, ub_BA = find_bounds(A_inh_BA, xs, threshold, (K_BA,), False)
    analysis_BA = analysis_inh(min_B, avg_B, max_B, lb_BA, ub_BA)
    if analysis_BA == "ONE":
        Adj[1,0]=0
    if analysis_AA == "ZERO" or analysis_BA == "ZERO":
        Adj[0,0]=Adj[1,0]=0 
    
    lb_AB, ub_AB = find_bounds(B_act_AB, xs, threshold, (K_AB,), True)
    analysis_AB = analysis_act(min_A, avg_A, max_A, lb_AB, ub_AB)
    if analysis_AB == "ONE":
        Adj[0,1]=0
    
    lb_CB, ub_CB = find_bounds(B_inh_CB, xs, threshold, (K_CB,), False)
    analysis_CB = analysis_inh(min_C, avg_C, max_C, lb_CB, ub_CB)
    if analysis_CB == "ONE":
        Adj[2,1]=0
    if analysis_AB == "ZERO" or analysis_CB == "ZERO":
        Adj[0,1]=Adj[2,1]=0
    
    lb_AC, ub_AC = find_bounds(C_inh_AC, xs, threshold, (K_AC,), False)
    analysis_AC = analysis_inh(min_A, avg_A, max_A, lb_AC, ub_AC)
    if analysis_AC == "ONE":
        Adj[0,2]=0
    
    lb_BC, ub_BC = find_bounds(C_inh_BC, xs, threshold, (K_BC,), False)
    analysis_BC = analysis_inh(min_B, avg_B, max_B, lb_BC, ub_BC)
    if analysis_BC == "ONE":
        Adj[1,2]=0
    
    lb_CC, ub_CC = find_bounds(C_act_CC, xs, threshold, (K_CC,), True)
    analysis_CC = analysis_act(min_C, avg_C, max_C, lb_CC, ub_CC)
    if analysis_CC == "ONE":
        Adj[2,2]=0
    if analysis_AC == "ZERO" or analysis_BC == "ZERO" or analysis_CC == "ZERO":
        Adj[0,2]=Adj[1,2]=Adj[2,2]=0
    
    return (Adj,
            analysis_AA, lb_AA, ub_AA,
            analysis_AB, lb_AB, ub_AB,
            analysis_AC, lb_AC, ub_AC,
            analysis_BA, lb_BA, ub_BA,
            analysis_BC, lb_BC, ub_BC,
            analysis_CB, lb_CB, ub_CB,
            analysis_CC, lb_CC, ub_CC)
def create_by_adj(A):    
    G=nx.from_numpy_array(A, create_using=nx.DiGraph)
    return nx.relabel_nodes(G, {0:'A', 1:'B', 2:'C'}, True)

def create_net(edges={}):
    A = np.array([[1, 1, -1], [-1, 0, -1], [0, -1, 1]])
    for loc,v in edges.items():
        A[loc] = v
    G=nx.from_numpy_array(A, create_using=nx.DiGraph)
    return nx.relabel_nodes(G, {0:'A', 1:'B', 2:'C'}, True)

def plot_net(A):
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    nx.draw_circular(create_by_adj(A), ax=ax, with_labels=True, font_weight='bold')
    plt.show()
    
def MCMC(original_params, L2_func, max_steps = 1000000, scale_factor=.1):
    current_point = np.random.normal(original_params, scale=original_params*1)
    current_point[current_point <= 0] = 1e-5
    
    L2_samples = np.zeros(max_steps)
    point_samples = np.zeros((max_steps,current_point.shape[0]))
    L2_samples[0] = L2_func(*current_point)
    point_samples[0, :] = current_point 
    for step in range(1, max_steps):
        next_point = np.random.normal(original_params, scale=original_params*scale_factor)
        next_point[next_point <= 0] = 1e-5
        next_L2_sample = L2_func(*next_point)
        if min(1, L2_samples[step-1]/next_L2_sample) > np.random.rand():
            current_point = next_point
            L2_samples[step] = next_L2_sample
        else:
            L2_samples[step] = L2_samples[step-1]
        point_samples[step, :] = current_point 
    return L2_samples, point_samples

def plot_single_minimum(original_params, L2_func, pca, component_index, steps = np.linspace(-1, 1, 200)):
    direction = pca.components_[component_index]
    test_points = directional_linspace(original_params, direction, steps)
    plt.plot(steps, [L2_func(*p) for p in  test_points], label=f"eigvec {component_index+1}")
    
def plot_minimums(original_params, L2_func, pca, steps = np.linspace(-1, 1, 200), per_plot=4, y_scale='log'):
    for i,direction in enumerate(pca.components_):
        plot_single_minimum(original_params, L2_func, pca, i, steps)
        if (i+1)%per_plot == 0 or (i+1) == len(pca.components_):
            original_L2 = L2_func(*original_params)
            plt.scatter(0, original_L2, marker='x')
            plt.yscale(y_scale);plt.legend();plt.grid();plt.show()
def directional_linspace(origin, direction, steps):
    return origin[np.newaxis, ...] + steps[...,np.newaxis]*direction[np.newaxis, ...]