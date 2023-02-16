"""
torch-specific functions.

"""
import numpy as np
import torch

#=========================================================================================
# Activation functions and some of their derivatives
#=========================================================================================

# Newer version of Theano has built-in ReLU
if hasattr(torch.nn, 'ReLU'):
    rectify = torch.nn.ReLU
else:
    def rectify(x):
        return torch.where(x > 0, x, 0)

def d_rectify(x):
    return torch.where(x > 0, 1, 0)

def rectify_power(x, n=2):
    return torch.where(x > 0, x**n, 0)

def d_rectify_power(x, n=2):
    return torch.where(x > 0, n*x**(n-1), 0)

sigmoid = torch.sigmoid

def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

tanh = torch.tanh

def d_tanh(x):
    return 1 - tanh(x)**2

def rtanh(x):
    return rectify(tanh(x))

def d_rtanh(x):
    return torch.where(x > 0, d_tanh(x), 0)

def softplus(x):
    return torch.log(1 + torch.exp(x))

d_softplus = sigmoid

def softmax(x):
    """
    Softmax function.

    Parameters
    ----------

    x : theano.tensor.tensor3
        This function assumes the outputs are the third dimension of x.

    """
    sh = x.shape
    x  = x.reshape((sh[0]*sh[1], sh[2]))
    fx = torch.nn.softmax(x)
    fx = fx.reshape(sh)

    return fx

#-----------------------------------------------------------------------------------------
# Gather all functions into a convenient dictionary.
#-----------------------------------------------------------------------------------------

hidden_activations = {
    'linear':        (lambda x: x,   lambda x: 1),
    'rectify':       (rectify,       d_rectify),
    'rectify_power': (rectify_power, d_rectify_power),
    'sigmoid':       (sigmoid,       d_sigmoid),
    'tanh':          (tanh,          d_tanh),
    'rtanh':         (rtanh,         d_rtanh),
    'softplus':      (softplus,      d_softplus)
}

output_activations = {
    'linear':        (lambda x: x),
    'rectify':       rectify,
    'rectify_power': rectify_power,
    'sigmoid':       sigmoid,
    'softmax':       softmax
    }

#=========================================================================================
# Loss functions
#=========================================================================================

epsilon = 1e-10

def binary_crossentropy(y, t):
    return -t*torch.log(y + epsilon) - (1-t)*torch.log((1-y) + epsilon)

def categorical_crossentropy(y, t):
    return -t*torch.log(y + epsilon)

def L2(y, t):
    return (y - t)**2

#=========================================================================================
# Theano
#=========================================================================================

# def grad(*args, **kwargs):
#     kwargs.setdefault('disconnected_inputs', 'warn')

#     return T.grad(*args, **kwargs)

# def function(*args, **kwargs):
#     kwargs.setdefault('on_unused_input', 'warn')

#     return theano.function(*args, **kwargs)

#=========================================================================================
# NumPy to Torch
#=========================================================================================

def shared(x, device, dtype=np.float64):
    if x.dtype == dtype:
        return torch.tensor(x)
    return torch.tensor(np.asarray(x, dtype=dtype),device=device )

def shared_scalar(x, device, dtype=np.float64):
    return torch.tensor(np.cast[dtype](x),device=device)

def shared_zeros(shape,device, dtype=np.float64):
    return torch.zeros(shape,dtype,device=device)

#=========================================================================================
# GPU
#=========================================================================================

def get_processor_type():
    """
    Test whether the GPU can be used

    """
    if torch.cuda.is_available():
        device='gpu'
    else:
        device='cpu'
    return device