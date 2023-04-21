import numpy as np
from itertools import product
from scipy.optimize import fsolve

    
def create_circuit_3954_dispersions():
    from sympy import symbols, Matrix, lambdify
    A, B, C = symbols('A, B, C', real=True, positive = True)
    D_A, D_B = symbols('D_A, D_B', real=True, positive = True)
    (b_A, b_B, b_C,
     V_A, V_B, V_C,
    K_AA, K_AB, K_AC, K_BA, K_BC, K_CB, K_CC,
    μ_A, μ_B, μ_C) = symbols(
        'b_A, b_B, b_C, V_A, V_B, V_C, K_AA, K_AB, K_AC, K_BA, K_BC, K_CB, K_CC, mu_A, mu_B, mu_C', 
        real=True, positive = True)
    k, n = symbols('k, n', integer=True)
    ###############################################
    def act(x, K, n):
        return 1/(1 + (K/x)**n)

    def inh(x, K, n):
        return 1/(1 + (x/K)**n)


    fA = b_A + V_A*act(A, K_AA, n)*inh(B, K_BA, n) - μ_A * A
    fB = b_B + V_B*act(A, K_AB, n)*inh(C, K_CB, n) - μ_B * B
    fC = b_C + V_C*inh(A, K_AC, n)*inh(B, K_BC, n)*act(C, K_CC, n) - μ_C * C
    
    Kinetic = Matrix([[fA], [fB], [fC]])
    ##############################################
    f1 = lambdify([n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                   b_B, μ_B, V_B, K_BA, K_BC,
                   b_C, μ_C, V_C, K_CB, K_CC,
                   A,B,C], 
                   fA,
                  modules='numpy')

    f2 = lambdify([n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                   b_B, μ_B, V_B, K_BA, K_BC,
                   b_C, μ_C, V_C, K_CB, K_CC,
                   A,B,C], 
                  fB,
                  modules='numpy')

    f3 = lambdify([n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                   b_B, μ_B, V_B, K_BA, K_BC,
                   b_C, μ_C, V_C, K_CB, K_CC,
                   A,B,C], 
                   fC,
                  modules='numpy')

    def create_func(n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                   b_B, μ_B, V_B, K_BA, K_BC,
                   b_C, μ_C, V_C, K_CB, K_CC):
        def f_1(args):
            A,B,C = args
            return (f1(n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                       b_B, μ_B, V_B, K_BA, K_BC,
                       b_C, μ_C, V_C, K_CB, K_CC, A,B,C),
                    f2(n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                       b_B, μ_B, V_B, K_BA, K_BC,
                       b_C, μ_C, V_C, K_CB, K_CC, A,B,C),
                    f3(n, b_A, μ_A, V_A, K_AA, K_AB, K_AC,
                       b_B, μ_B, V_B, K_BA, K_BC,
                       b_C, μ_C, V_C, K_CB, K_CC, A,B,C)) 

        return f_1
    ###############################################
    J_jac = Kinetic.jacobian([A, B, C])
    J_jac_diff = J_jac - Matrix([[D_A*k**2, 0,        0], 
                             [0,        D_B*k**2, 0],
                             [0,        0,        0],
                           ])
    ###############################################
    def find_root_original(n_val, 
                   b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
                   b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
                   b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val):    
        func = create_func(n_val, 
                           b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
                           b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
                           b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val)        
        roots, d, ier, msg = fsolve(func, [10, 10, 10], xtol=1e-10, maxfev=100000,  full_output=1)        
        # check the solution is valid
        (a_1, b_1, c_1) = func(roots)
        if ier != 1 or a_1 > 1e-8 or b_1 > 1e-8 or c_1 > 1e-8 :
            roots, d, ier, msg = fsolve(func, [-1, -1, -1], xtol=1e-10, maxfev=100000,  full_output=1)
            # check the solution is valid
            (a_1, b_1, c_1) = func(roots)
            if ier != 1 or a_1 > 1e-8 or b_1 > 1e-8 or c_1 > 1e-8 :
                print("================================================")
                print(f"     roots original finds failed.")
                print("================================================")
        return roots

    def find_roots(n_val, 
                   b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
                   b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
                   b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val,
                   verbose=False):    
        func = create_func(n_val, 
                           b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
                           b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
                           b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val)
        #
        roots_res = []
        for init_vals in product([.1,-.1], repeat=3):        
            roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
            (a_1, b_1, c_1) = func(roots)
            if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
                roots_res.append(roots)
        for init_vals in product([1,-1], repeat=3):        
            roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
            (a_1, b_1, c_1) = func(roots)
            if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
                roots_res.append(roots)

        for init_vals in product([10,-10], repeat=3):        
            roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
            (a_1, b_1, c_1) = func(roots)
            if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
                roots_res.append(roots)

        for init_vals in product([100,-100], repeat=3):        
            roots, d, ier, msg = fsolve(func, init_vals, xtol=1e-7, maxfev=10000,  full_output=1)
            (a_1, b_1, c_1) = func(roots)
            if ier == 1 and a_1 < 1e-7 and b_1 < 1e-7 and c_1 < 1e-7:
                roots_res.append(roots)

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
    ###############################################
    def get_dispersions(ks, 
                        n_val, D_A_val, D_B_val,
                        b_A_val, mu_A_val, V_A_val, K_AA_val, K_AB_val, K_AC_val,
                        b_B_val, mu_B_val, V_B_val, K_BA_val, K_BC_val,
                        b_C_val, mu_C_val, V_C_val, K_CB_val, K_CC_val,
                        A_star, B_star, C_star):    
        λ_1, λ_2, λ_3 = list(J_jac_diff.subs(
                              {n:n_val, 
                               b_A:b_A_val, 
                               μ_A:mu_A_val, 
                               V_A:V_A_val,
                               K_AA:K_AA_val, 
                               K_AB:K_AB_val,  
                               K_AC:K_AC_val,
                               b_B:b_B_val, 
                               μ_B:mu_B_val, 
                               V_B:V_B_val,
                               K_BA:K_BA_val, 
                               K_BC:K_BC_val,  
                               b_C:b_C_val, 
                               μ_C:mu_C_val, 
                               V_C:V_C_val,
                               K_CB:K_CB_val, 
                               K_CC:K_CC_val,  
                               A:A_star, 
                               B:B_star, 
                               C:C_star}
                         ).eigenvals().keys())

        λ_1_func = lambdify([k, D_A, D_B], 
                             λ_1,
                             modules='numpy')
        λ_2_func = lambdify([k, D_A, D_B], 
                             λ_2,
                             modules='numpy')
        λ_3_func = lambdify([k, D_A, D_B], 
                             λ_3,
                             modules='numpy')

        dis1 = λ_1_func(ks, D_A_val, D_B_val)
        dis2 = λ_2_func(ks, D_A_val, D_B_val)    
        dis3 = λ_3_func(ks, D_A_val, D_B_val)   
        return dis1, dis2,dis3
    
    return get_dispersions, find_roots, find_root_original

