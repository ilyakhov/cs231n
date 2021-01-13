from builtins import object
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
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros((num_filters))
        print('W1.shape', self.params['W1'].shape) 
        print('b1.shape', self.params['b1'].shape)
        
        pad = (filter_size - 1) // 2
        conv_stride = 1
        input_height = input_dim[1]
        conv_map_size = (input_height - filter_size + 2*pad)/conv_stride + 1
        
        pool_filter_size = 2
        pool_stride = 2
        pool_map_size = (conv_map_size - pool_filter_size) / pool_stride + 1
        
        assert int(pool_map_size) == pool_map_size, 'pool_map_size is not integer'
        hidden_dim_2 = int(num_filters * (pool_map_size ** 2))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim_2, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
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
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        conv_cache0, relu_cache0, pool_cache0 = conv_cache
        
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        
        densed = np.reshape(conv_out, newshape=(-1, W2.shape[0]))
        affine_out1, affine_cache1 = affine_relu_forward(densed, W2, b2)
        fc_cache1, relu_cache1 = affine_cache1
        
        affine_out2, affine_cache2 = affine_forward(affine_out1, W3, b3)
        fc_cache2 = affine_cache2
        
        scores = affine_out2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, loss_grad = softmax_loss(scores, y)
        
        N = 3  # 3 weight sets 
        W_params = [self.params['W{}'.format(i+1)] for i in range(N)]
        L2_reg = 0
        for w in W_params:
            L2_reg += np.sum(w*w)
            
        loss += 0.5 * self.reg * L2_reg
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        
        dx3, dw3, db3 = affine_backward(loss_grad, affine_cache2)
        grads['W{}'.format(N)] = dw3 + self.reg * self.params['W{}'.format(N)]
        grads['b{}'.format(N)] = db3
        
        dx2, dw2, db2 = affine_relu_backward(dx3, affine_cache1)
        grads['W{}'.format(N-1)] = dw2 + self.reg * self.params['W{}'.format(N-1)]
        grads['b{}'.format(N-1)] = db2
        
        dx2_reshaped = np.reshape(dx2, newshape=(conv_out.shape))
        
        dx1, dw1, db1 = conv_relu_pool_backward(dx2_reshaped, conv_cache)
        grads['W{}'.format(N-2)] = dw1 + self.reg * self.params['W{}'.format(N-2)]
        grads['b{}'.format(N-2)] = db1
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
