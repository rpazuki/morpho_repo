import numpy as np

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
