import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MySixLayerConvNet(object):
  """
  A six-layer convolutional network with the following architecture:
  
  (conv - spatial batchnorm - relu - 2x2 max pool) x 4 - (affine - batchnorm- relu) - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=(32, 64, 32, 64), filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in different convolutional layers
    - filter_size: Size of filters to use in the convolutional layers
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
    self.use_batchnorm = use_batchnorm
    
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
    self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size)*weight_scale
    self.params['b1'] = np.zeros((num_filters[0],))
    self.params['W2'] = np.random.randn(num_filters[1], num_filters[0], filter_size, filter_size)*weight_scale
    self.params['b2'] = np.zeros((num_filters[1],))
    self.params['W3'] = np.random.randn(num_filters[2], num_filters[1], filter_size, filter_size)*weight_scale
    self.params['b3'] = np.zeros((num_filters[2],))
    self.params['W4'] = np.random.randn(num_filters[3], num_filters[2], filter_size, filter_size)*weight_scale
    self.params['b4'] = np.zeros((num_filters[3],))
    self.params['W5'] = np.random.randn(num_filters[3]*(input_dim[1]/16)*(input_dim[2]/16),hidden_dim)*weight_scale
    self.params['b5'] = np.zeros((hidden_dim,))
    self.params['W6'] = np.random.randn(hidden_dim,num_classes)*weight_scale
    self.params['b6'] = np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    num_layers = 6
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(num_layers - 1)]
      self.params['gamma1'] = np.ones((num_filters[0],))
      self.params['beta1'] = np.zeros((num_filters[0],))
      self.params['gamma2'] = np.ones((num_filters[1],))
      self.params['beta2'] = np.zeros((num_filters[1],))
      self.params['gamma3'] = np.ones((num_filters[2],))
      self.params['beta3'] = np.zeros((num_filters[2],))
      self.params['gamma4'] = np.ones((num_filters[3],))
      self.params['beta4'] = np.zeros((num_filters[3],))
      self.params['gamma5'] = np.ones((hidden_dim,))
      self.params['beta5'] = np.zeros((hidden_dim,))
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
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
    [out1, cache_layer1] = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    [out2, cache_layer2] = conv_batchnorm_relu_pool_forward(out1, W2, b2, conv_param, pool_param, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
    [out3, cache_layer3] = conv_batchnorm_relu_pool_forward(out2, W3, b3, conv_param, pool_param, self.params['gamma3'], self.params['beta3'], self.bn_params[2])
    [out4, cache_layer4] = conv_batchnorm_relu_pool_forward(out3, W4, b4, conv_param, pool_param, self.params['gamma4'], self.params['beta4'], self.bn_params[3])
    [out5, cache_layer5] = affine_batchnorm_relu_forward(out4, W5, b5, self.params['gamma5'], self.params['beta5'], self.bn_params[4])
    [scores, cache_layer6] = affine_forward(out5,W6,b6)
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
    loss, dscores = softmax_loss(scores,y)
    loss += 0.5*self.reg* (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) + np.sum(W4*W4) + np.sum(W5*W5) + np.sum(W6*W6))
    dout5, grads['W6'], grads['b6'] = affine_backward(dscores,cache_layer6)
    grads['W6'] += self.reg*W6
    dout4, grads['W5'], grads['b5'], grads['gamma5'], grads['beta5'] = affine_batchnorm_relu_backward(dout5, cache_layer5)
    grads['W5'] += self.reg*W5
    dout3, grads['W4'], grads['b4'], grads['gamma4'], grads['beta4'] = conv_batchnorm_relu_pool_backward(dout4, cache_layer4)
    grads['W4'] += self.reg*W4
    dout2, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = conv_batchnorm_relu_pool_backward(dout3, cache_layer3)
    grads['W3'] += self.reg*W3
    dout1, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_batchnorm_relu_pool_backward(dout2, cache_layer2)
    grads['W2'] += self.reg*W2
    dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dout1, cache_layer1)
    grads['W1'] += self.reg*W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

