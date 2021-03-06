import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    HH = WW = filter_size

    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H/2 * W/2, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    A = X  # Activations
    cache = [0] * 3

    # conv - relu - 2x2 max pool
    A, cache[0] = conv_relu_pool_forward(A, W1, b1, conv_param, pool_param)

    # affine - relu
    A, cache[1] = affine_relu_forward(A, W2, b2)

    # affine - softmax
    scores, cache[2] = affine_forward(A, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dA = softmax_loss(scores, y)

    dA, grads['W3'], grads['b3'] = affine_backward(dA, cache[2])
    dA, grads['W2'], grads['b2'] = affine_relu_backward(dA, cache[1])
    dA, grads['W1'], grads['b1'] = conv_relu_pool_backward(dA, cache[0])

    # Regularisation
    def sumsquare(x):
        return np.sum(np.square(x))
    # * [for W in [W1, W2, W3]]
    regularization_loss = 0.5 * self.reg * (
        sumsquare(W1) + sumsquare(W2) + sumsquare(W3))
    loss += regularization_loss
    grads['W3'] += self.reg * self.params['W3']
    grads['W2'] += self.reg * self.params['W2']
    grads['W1'] += self.reg * self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class DeepConvNet(object):
  """
  A five-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.

  TODO: Generalise this to take arbitrary shapes.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    HH = WW = filter_size

    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters, num_filters, HH, WW))
    self.params['b2'] = np.zeros(num_filters)

    self.params['W3'] = np.random.normal(0, weight_scale, (num_filters, num_filters, HH, WW))
    self.params['b3'] = np.zeros(num_filters)

    self.params['W4'] = np.random.normal(0, weight_scale, (num_filters * H/8 * W/8, hidden_dim))
    self.params['b4'] = np.zeros(hidden_dim)

    self.params['W5'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b5'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    A = X  # Activations
    cache = [0] * 5

    # conv - relu - 2x2 max pool
    A, cache[0] = conv_relu_pool_forward(A, W1, b1, conv_param, pool_param)

    # conv - relu - 2x2 max pool
    A, cache[1] = conv_relu_pool_forward(A, W2, b2, conv_param, pool_param)

    # conv - relu - 2x2 max pool
    A, cache[2] = conv_relu_pool_forward(A, W3, b3, conv_param, pool_param)

    # affine - relu
    A, cache[3] = affine_relu_forward(A, W4, b4)

    # affine - softmax
    scores, cache[4] = affine_forward(A, W5, b5)

    if y is None:
      return scores

    loss, grads = 0, {}
    loss, dA = softmax_loss(scores, y)

    dA, grads['W5'], grads['b5'] = affine_backward(dA, cache[4])
    dA, grads['W4'], grads['b4'] = affine_relu_backward(dA, cache[3])
    dA, grads['W3'], grads['b3'] = conv_relu_pool_backward(dA, cache[2])
    dA, grads['W2'], grads['b2'] = conv_relu_pool_backward(dA, cache[1])
    dA, grads['W1'], grads['b1'] = conv_relu_pool_backward(dA, cache[0])

    # Regularisation
    def sumsquare(x):
        return np.sum(np.square(x))
    regularization_loss = 0.5 * self.reg * np.sum([sumsquare(W) for W in [W1, W2, W3, W4, W5]])
    loss += regularization_loss
    grads['W5'] += self.reg * self.params['W5']
    grads['W4'] += self.reg * self.params['W4']
    grads['W3'] += self.reg * self.params['W3']
    grads['W2'] += self.reg * self.params['W2']
    grads['W1'] += self.reg * self.params['W1']

    return loss, grads


pass
