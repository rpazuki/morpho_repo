import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import sys
import argparse
import pathlib
import pickle
from pathlib import Path
sys.path.append(f"{Path.home()}/morpho_repo")
sys.path.append(f"{Path.home()}/morpho_repo/pytorch")
from pinns import *
from local_utils import *

from pinns import *
from intro import *



def main() -> int:
    """"""
    parser = argparse.ArgumentParser(
                    prog='TINN',
                    description='Run the full parameter estimation',
                    epilog=''
                    )
    
    # Optional device number argument
    parser.add_argument('--run_name', type=str, required=True,
                        help='The prefix name of the outputs -- requred argument')
    
    parser.add_argument('--device_id', type=int, default=0, nargs='?', metavar='',
                        help='An optional integer device number argument')
    
    parser.add_argument('--input', type=str, default='./outputs/solution_1.pkl', nargs='?', metavar='',
                       help='A pickled file that contains the solution of the raction-diffusion model.')
    
    parser.add_argument('--epochs', type=int, default=5000, nargs='?', metavar='',
                       help='Maximum number of epochs, if the early stop condition does not satisfy.')
    
    parser.add_argument('--noise_level', type=float, default=0.0, nargs='?', metavar='',
                       help='Nose level in percentage.')
    
    parser.add_argument('--train_split_ratio', type=float, default=1/16, nargs='?', metavar='ratio',
                       help='Train/validation split (between zero and one).')
    
    parser.add_argument('--early_stop_window', type=int, default=50, nargs='?', metavar='window',
                       help='Early stop moving average window size.')
    
    parser.add_argument('--output', type=str, default='./outputs', nargs='?', metavar='',
                       help='The folder name that the outputs will be saved in.')
    # Pase the arguments
    args = parser.parse_args()
    if args.device_id is None:
        args.device_id = 0
    if args.input is None:
        args.input = './outputs/solution_1.pkl'
    if args.epochs is None:
        args.epochs = 5000
    if args.noise_level is None:
        args.noise_level = 0.0
    if args.train_split_ratio is None:
        args.train_split_ratio = 1/16
    if args.early_stop_window is None:
        args.early_stop_window = 50
    if args.output is None:
        args.output = './outputs'
    ######################################
    print("~"*45)
    print(f"      start running model '{args.run_name}'.")
    print("~"*45)
    ######################################
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    dev_str = f"{device.type}:{device.index}"
    print(f"The selected device is {dev_str}")
    ######################################
    with open(args.input, "rb") as f:
        (run_params, params, solution) = pickle.load(f)

    run_params = Struct(**run_params)
    params = Struct(**params)
    #####################################
    A_n = solution[1, 0, :, :]
    B_n = solution[1, 1, :, :]
    C_n = solution[1, 2, :, :]
    #############################################################
    #    Inner boundaries
    # By using these slices, we can remove the boundary effects
    # or select a smaller portion of the domain
    #
    #   Important: slices must not be larger than slices(1,-1,1).
    #              Otherwise, the boundary effects of finit difference
    #              Laplacian cannot be removed
    #
    x_slice = slice(1,-1, 1)
    y_slice = slice(1,-1, 1)

    to = create_to(A_n[x_slice,y_slice])
    ########################################
    # Take the average of data points
    kernel =np.array([[1, 1],
                      [1, 1]])
    A_n = ndimage.convolve(A_n, kernel)/4
    B_n = ndimage.convolve(B_n, kernel)/4
    C_n = ndimage.convolve(C_n, kernel)/4
    ######################################
    # Add noise to averaged mesh
    if args.noise_level > 0:
        ampA = np.max(A_n) - np.min(A_n)
        ampB = np.max(B_n) - np.min(B_n)
        ampC = np.max(C_n) - np.min(C_n)
        print(f"noise level:{args.noise_level}, A amplitude:{ampA}, B amplitude:{ampB}, C amplitude:{ampC},")
        print(f"A noise std:{ampA*args.noise_level}, B noise std:{ampB*args.noise_level}, C noise std:{ampC*args.noise_level}.")
        np.random.seed(42)
        A_n += np.random.normal(0,ampA*args.noise_level ,A_n.shape)
        B_n += np.random.normal(0,ampB*args.noise_level ,B_n.shape)
        C_n += np.random.normal(0,ampC*args.noise_level ,C_n.shape)
    #######################################
    torch.manual_seed(42)
    np.random.seed(42)
    ###########################################################
    # Inputs
    # restrict to inner boundaries
    x = torch.linspace(0, run_params.Lx, run_params.Ix)[x_slice].to(device)
    y = torch.linspace(0, run_params.Ly, run_params.Jy)[y_slice].to(device)
    X,Y = torch.meshgrid(x, y, indexing='ij')
    data_X = torch.vstack([X.flatten(), Y.flatten()]).T.requires_grad_(True).to(device)
    ##########################################################
    #   Data
    # restrict to inner boundaries
    data_A = torch.from_numpy(A_n[x_slice,y_slice].flatten()).to(device)
    data_B = torch.from_numpy(B_n[x_slice,y_slice].flatten()).to(device)
    data_C = torch.from_numpy(C_n[x_slice,y_slice].flatten()).to(device)
    ###########################################################
    # LoG diffusion instead of Laplacians
    # First, find the Laplacian of the Gaussian on the whole domain,
    # then, select the inner boundaries. This way, the boundaries 
    # effets removed
    diffusion_scale_factor = (run_params.Ix*run_params.Jy/(run_params.Lx*run_params.Ly))
    laplacianA = diffusion_scale_factor * ndimage.gaussian_laplace(A_n, sigma=2.2)
    laplacianB = diffusion_scale_factor * ndimage.gaussian_laplace(B_n, sigma=2.2)

    laplacianA = torch.tensor(laplacianA[x_slice,y_slice].flatten()).to(device)
    laplacianB = torch.tensor(laplacianB[x_slice,y_slice].flatten()).to(device)
    ###########################################################
    # Model
    lb = torch.tensor([torch.min(x).item(), torch.min(y).item()]).to(device)
    ub = torch.tensor([torch.max(x).item(), torch.max(y).item()]).to(device)
    model = Net_dense_normalised([2, 64, 64, 3], lb, ub).to(device)
    ###########################################################
    # optimizer
    optimizer = torch.optim.LBFGS([*model.parameters()], lr=1
                                 ,line_search_fn='strong_wolfe')#.Adam([*model.parameters()], lr=1e-3)#
    ###########################################################
    # Train/Validation splits
    #
    split_ratio = args.train_split_ratio
    # Data
    max_index = data_X.shape[0]
    All_indices = torch.randperm(max_index)
    train_last_index = int(split_ratio*max_index)
    train_indices = All_indices[:train_last_index]
    validation_indices = All_indices[train_last_index:]


    data_X_train = data_X[train_indices,:]
    data_X_validation = data_X[validation_indices,:]
    data_A_train = data_A[train_indices]
    data_A_validation = data_A[validation_indices]
    data_B_train = data_B[train_indices]
    data_B_validation = data_B[validation_indices]
    data_C_train = data_C[train_indices]
    data_C_validation = data_C[validation_indices]

    # Laplacian
    max_Laplacian_index = laplacianA.shape[0]
    All_Laplacian_indices = torch.randperm(max_Laplacian_index)
    laplacian_last_index = int(split_ratio*max_Laplacian_index)
    train_Laplacian_indices = All_Laplacian_indices[:laplacian_last_index]
    validation_Laplacian_indices = All_Laplacian_indices[laplacian_last_index:]

    laplacianA_train = laplacianA[train_Laplacian_indices].to(device)
    laplacianA_validation = laplacianA[validation_Laplacian_indices].to(device)
    laplacianB_train = laplacianB[train_Laplacian_indices].to(device)
    laplacianB_validation = laplacianB[validation_Laplacian_indices].to(device)


    epochs = args.epochs#3000
    loss_data = 0.0
    lambda_data = 1.0
    loss_physics = 0.0
    lambda_physics_laplacian = 1e-2

    losses = np.zeros((5, epochs))
    validations = np.zeros(epochs)

    def act(x, km, n=2):
        return x**n / (x**n + km**n)

    def inh(x, km, n=2):
        return  km**n / (x**n + km**n)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    for i in range(epochs):    
        def find_data_losses(data_X_input, data_A, data_B, data_C):
            #################################
            # data loss       
            data_f_hat = model(data_X_input).squeeze() 
            data_A_hat = data_f_hat[:,0]
            data_B_hat = data_f_hat[:,1]
            data_C_hat = data_f_hat[:,2]
            loss_A_data = torch.mean((data_A - data_A_hat)**2)
            loss_B_data = torch.mean((data_B - data_B_hat)**2)
            loss_C_data = torch.mean((data_C - data_C_hat)**2)
            return (loss_A_data,loss_B_data,loss_C_data)

        def find_Laplacian_losses(data_X_input, laplacianA, laplacianB,
                                  laplacian_indices):
            #################################
            # physics derivatives
            #        
            physics_f = model(data_X_input).squeeze()         
            A_hat = physics_f[:,0]
            B_hat = physics_f[:,1]        
            # Note: The Laplacian is taken first on all points
            #       and then are selected for the given index
            #       The reason to do it this way is becuase of
            #       The laplacian bounderies effect that we need to remove
            laplacianA_hat = Laplacian(A_hat, data_X_input)
            laplacianB_hat = Laplacian(B_hat, data_X_input)

            laplacianA_hat = laplacianA_hat[laplacian_indices]
            laplacianB_hat = laplacianB_hat[laplacian_indices]

            A_loss_laplacian = torch.mean((laplacianA_hat-laplacianA)**2)
            B_loss_laplacian = torch.mean((laplacianB_hat-laplacianB)**2)                 
            ###############################
            return (A_loss_laplacian, B_loss_laplacian)

        # L-BFGS
        def closure():

            optimizer.zero_grad()
            (loss_A_data,loss_B_data,loss_C_data
            ) = find_data_losses(data_X_train,data_A_train,data_B_train,data_C_train)

            (A_loss_laplacian, B_loss_laplacian
            ) = find_Laplacian_losses(data_X, laplacianA_train,laplacianB_train,
                                      train_Laplacian_indices)

            loss_data = (loss_A_data + loss_B_data + loss_C_data)/3
            loss_laplacian_physics = A_loss_laplacian + B_loss_laplacian
            total_loss = (
                +lambda_data*loss_data            
                +lambda_physics_laplacian*loss_laplacian_physics
                            )

            total_loss.backward(retain_graph=True)


            losses[:,i] = (loss_A_data.item(), loss_B_data.item(), loss_C_data.item(), 
                           A_loss_laplacian.item(), B_loss_laplacian.item())                    

            return total_loss



        optimizer.step(closure)


        (validation_A_data,validation_B_data,validation_C_data,
         ) = find_data_losses(data_X_validation, 
                              data_A_validation,data_B_validation,data_C_validation)

        (A_validation_laplacian, B_validation_laplacian
        ) = find_Laplacian_losses(data_X, laplacianA_validation, laplacianB_validation,
                                  validation_Laplacian_indices)

        validation_loss = (validation_A_data+validation_B_data+validation_C_data+
                           A_validation_laplacian+B_validation_laplacian).item()
        validations[i] = validation_loss

        # Early stop
        w = args.early_stop_window
        if i > w:
            if validations[i] > moving_average(validations[:i], w)[-1]:
                print(f"Earlt stop at epoch:{i+1}.\n"
                      f" Validation:{validations[i]}, averged validation:{moving_average(validations[:i], w)[-1]}")
                break    

        if (i+1)%200 == 0 or i==0 :
            print("============================================")
            print(f"Epoch: {i+1} \n data loss:{np.sum(losses[0:3,i]):.6f}, \n"
                  f"data A loss:{losses[0,i]:.6f}, data B loss:{losses[1,i]:.6f}, data C loss:{losses[2,i]:.6f}, \n"
                  f"Laplacian A loss:{losses[3,i]:.6f}, Laplacian B loss:{losses[4,i]:.6f}\n"
                  f"\n"
                  f"Validation loss:{validation_loss:.6f}"
                 )
    #######################################
    # Check the trained model
    print("~"*45)
    print(f"      Check the trained model '{args.run_name}' outputs.")
    print("~"*45)
    physics_f = model(data_X).squeeze()
    A_hat = physics_f[:,0]
    B_hat = physics_f[:,1]
    C_hat = physics_f[:,2]

    laplacianA_hat = Laplacian(A_hat, data_X)
    laplacianB_hat = Laplacian(B_hat, data_X)

    A_hat = to(A_hat.cpu().detach().numpy())
    B_hat = to(B_hat.cpu().detach().numpy())
    C_hat = to(C_hat.cpu().detach().numpy())
    laplacianA_hat = to(laplacianA_hat.cpu().detach().numpy())
    laplacianB_hat = to(laplacianB_hat.cpu().detach().numpy())


    laplacianA = diffusion_scale_factor * grad_diff(A_n)[x_slice,y_slice]
    laplacianB = diffusion_scale_factor * grad_diff(B_n)[x_slice,y_slice]

    n = params["n"]

    A_n = A_n[x_slice,y_slice]
    B_n = B_n[x_slice,y_slice]
    C_n = C_n[x_slice,y_slice]

    F_A =  params["b_A"] + params["V_A"]*act(A_n, params["K_AA"], n)*inh(B_n, params["K_BA"], n) - params["mu_A"] * A_n
    F_A_hat =  params["b_A"] + params["V_A"]*act(A_hat, params["K_AA"], n)*inh(B_hat, params["K_BA"], n) - params["mu_A"] * A_hat
    F_B =  params["b_B"] + params["V_B"]*act(A_n, params["K_AB"], n)*inh(C_n, params["K_CB"], n) - params["mu_B"] * B_n
    F_B_hat =  params["b_B"] + params["V_B"]*act(A_hat, params["K_AB"], n)*inh(C_hat, params["K_CB"], n) - params["mu_B"] * B_hat
    F_C =  params["b_C"] + params["V_C"]*inh(A_n, params["K_AC"], n)*inh(B_n, params["K_BC"], n)*act(C_n, params["K_CC"], n) - params["mu_C"] * C_n
    F_C_hat =  params["b_C"] + params["V_C"]*inh(A_hat, params["K_AC"], n)*inh(B_hat, params["K_BC"], n)*act(C_hat, params["K_CC"], n) - params["mu_C"] * C_hat


    dAdt = params["D_A"]*laplacianA + F_A
    dAdt_hat = params["D_A"]*laplacianA_hat + F_A_hat
    dBdt = params["D_B"]*laplacianB + F_B
    dBdt_hat = params["D_B"]*laplacianB_hat + F_B_hat

    print(f"A diff:{np.mean((A_hat-A_n)**2)}",
          f"B diff:{np.mean((B_hat-B_n)**2)}",
          f"C diff:{np.mean((C_hat-C_n)**2)}")

    print(f"F_A actual:{np.mean(F_A**2)}, F_A:{np.mean(F_A_hat**2)}, RMSE: {np.sqrt(np.mean((F_A - F_A_hat)**2))}")
    print(f"F_B actual:{np.mean(F_B**2)}, F_B:{np.mean(F_B_hat**2)}, RMSE: {np.sqrt(np.mean((F_B - F_B_hat)**2))}")
    print(f"F_C actual:{np.mean(F_C**2)}, F_C:{np.mean(F_C_hat**2)}, RMSE: {np.sqrt(np.mean((F_C - F_C_hat)**2))}")


    print()
    print(f"Laplacian A RMSE: {np.sqrt(np.mean((laplacianA_hat-laplacianA)**2))}")
    print(f"Laplacian B RMSE: {np.sqrt(np.mean((laplacianB_hat-laplacianB)**2))}")

    print()
    print(f"dAdt RMSE: {np.sqrt(np.mean((dAdt-dAdt_hat)**2))}")
    print(f"dBdt RMSE: {np.sqrt(np.mean((dBdt-dBdt_hat)**2))}")
    print(f"dCdt RMSE: {np.sqrt(np.mean((F_C-F_C_hat)**2))}")
    ############################################################
    # Find the parameters
    def create_var(init=None):        
        if init is None:
            return torch.nn.Parameter(torch.ones(1, requires_grad=True, device=dev_str))
        else:
            return torch.nn.Parameter(init*torch.ones(1, requires_grad=True, device=dev_str))


    D_A=params["D_A"]
    D_B=params["D_B"]
    b_A = create_var()
    b_B = create_var()
    b_C = create_var()
    V_A = create_var()
    V_B = create_var()
    V_C = create_var()
    mu_A = create_var()
    mu_B = create_var()
    mu_C=params["D_A"]
    K_AA = create_var()
    K_AB = create_var()
    K_AC = create_var()
    K_BA = create_var()
    K_BC = create_var()
    K_CB = create_var()
    K_CC = create_var()
    n = params['n']

    params_name_list = [#"D_A", "D_B", 
                        "b_A", "b_B", "b_C", "V_A", "V_B", "V_C", "mu_A", "mu_B", #"mu_C",
                        "K_AA", "K_AB", "K_AC", "K_BA", "K_BC", "K_CB", 
                        "K_CC"
    ]

    params_list = [#D_A, D_B, 
                   b_A, b_B, b_C, V_A, V_B, V_C, mu_A, mu_B, #mu_C,
                   K_AA, K_AB, K_AC, K_BA, K_BC, K_CB, 
                   K_CC
    ]
    def physics_loss():            
        physics_f = model(data_X).squeeze()
        A_hat = physics_f[:,0]
        B_hat = physics_f[:,1]
        C_hat = physics_f[:,2]

        laplacianA_hat = Laplacian(A_hat, data_X)
        laplacianB_hat = Laplacian(B_hat, data_X)    
        # To make sure the parameters stay positive, we use the exponential function    
        e = torch.exp
        F_A_hat =  e(b_A) + e(V_A)*act(A_hat, e(K_AA), n)*inh(B_hat, e(K_BA), n) - e(mu_A) * A_hat
        #f_A =  e(b_A)/(e(D_A)+1e-6) + e(V_A)*act(A, e(K_AA), n)*inh(B, e(K_BA), n)/(e(D_A)+1e-6) - e(mu_A)*A/(e(D_A)+1e-6)
        F_B_hat =  e(b_B) + e(V_B)*act(A_hat, e(K_AB), n)*inh(C_hat, e(K_CB), n) - e(mu_B) * B_hat
        #f_B =  e(b_B)/(e(D_B)+1e-6) + e(V_B)*act(A, e(K_AB), n)*inh(C, e(K_CB), n)/(e(D_B)+1e-6) - e(mu_B)*B/(e(D_B)+1e-6)
        F_C_hat =  e(b_C) + e(V_C)*inh(A_hat, e(K_AC), n)*inh(B_hat, e(K_BC), n)*act(C_hat, e(K_CC), n) - mu_C * C_hat


        #dAdt = e(D_A) * laplacianA + F_A
        dAdt = D_A * laplacianA_hat + F_A_hat
        #dAdt2 = laplacianA + f_A
        #dBdt = e(D_B) * laplacianB + F_B
        dBdt = D_B * laplacianB_hat + F_B_hat
        #dBdt2 = laplacianB + f_B
        dCdt = F_C_hat
        ################################
        # physics loss
        # Construct the physics loss here
        A_loss_physics = torch.mean(dAdt**2)
        #A2_loss_physics = torch.mean(dAdt2**2)
        B_loss_physics = torch.mean(dBdt**2)
        #B2_loss_physics = torch.mean(dBdt2**2)
        C_loss_physics = torch.mean(dCdt**2)
        return (A_loss_physics + B_loss_physics + C_loss_physics)


    optimizer2 = torch.optim.LBFGS(params_list,
                                   lr=.5,
                                   history_size=10, 
                                   max_iter=20, 
                                   line_search_fn="strong_wolfe")

    # L-BFGS
    def closure():
        if torch.is_grad_enabled():
            optimizer2.zero_grad()
        loss = physics_loss()
        if loss.requires_grad:
            loss.backward()
        return loss
    
    print("~"*50)
    print(f"      Estimates the parameters of the model '{args.run_name}'.")
    print("~"*50)
    history_lbfgs = []
    stored_parameters = np.zeros(len(params_list))
    for i in range(20):
        history_lbfgs.append(physics_loss().item())
        if np.all([not np.isnan(p.item()) for p in params_list]):
            stored_parameters = [np.exp(p.item()) for p in params_list]
        else:
            print("Nan -- (maybe large learning rate)")
            break

        if i%1 ==0:
            print(history_lbfgs[-1],", ".join([f"{name}={np.exp(l.item()):.4f}" 
                     for l, name in zip(params_list, params_name_list)]))
            print()
        optimizer2.step(closure)
    ########################################
    estimated_params = dict()
    for l, name in zip(params_list, params_name_list):    
        estimated_params[name] = np.exp(l.item())
    
    
    print("name \t Originl \t Estimated \t Difference")
    print("-------------------------------------------------------------------------------------------------------------")
    print("\n".join([ f"{name}:\t{params[name]:.3f}\t\t"
                      f"{estimated_params[name]:.3f}\t\t"
                     f"{np.abs(params[name]-estimated_params[name]):.3f}"
          for name in params_name_list
    ]))
    #######################################
    print("~"*45)
    print(f"      Save the results at '{args.output}'.")
    print("~"*45)
#     torch.save(model.state_dict(), f"{args.output}/model_{args.run_name}")
#     with open(f"{args.output}/model_{args.run_name}_params_name_list.npy", 'wb') as f:
#         np.save(f, params_name_list)    
#     with open(f"{args.output}/model_{args.run_name}_losses.npy", 'wb') as f:
#         np.save(f, losses)    
#     with open(f"{args.output}/model_{args.run_name}_estimated_params.pkl", "wb") as f:
#         pickle.dump((dict(params), dict(estimated_params)), f) 
    print("~"*45)
    print(f"      End of the model '{args.output}'.")
    print("~"*45)


    return 0

if __name__ == '__main__':
    sys.exit(main())  
