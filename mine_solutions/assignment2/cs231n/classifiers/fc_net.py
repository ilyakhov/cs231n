from builtins import range
from builtins import object
from functools import reduce
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        affine_1_output, affine_1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        relu_1_output, relu_1_cache = relu_forward(affine_1_output)
        affine_2_output, affine_2_cache = affine_forward(relu_1_output, self.params['W2'], self.params['b2'])
        scores = affine_2_output
        
        # softmax
        #softmax_base = np.sum(np.exp(scores), axis=1)
        #softmax_base = np.reshape(softmax_base, (softmax_base.shape[0], 1))
        #scores = np.exp(scores) / softmax_base
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, loss_grad = softmax_loss(scores, y)
        
        W1 = self.params['W1']
        W2 = self.params['W2']
        loss += 0.5 * self.reg * (np.sum(W2 * W2) + np.sum(W1 * W1))
        
        dx_2, dw2, db2 = affine_backward(loss_grad, affine_2_cache)
        dx_1 = relu_backward(dx_2, relu_1_cache)
        _, dw1, db1 = affine_backward(dx_1, affine_1_cache)
        
        grads['W1'] = dw1 + self.reg * W1  # plus loss
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.num_hidden = len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        for i in range(0, self.num_hidden):
            if i == 0:
                weights = np.random.randn(input_dim, hidden_dims[i])
            else:
                weights = np.random.randn(hidden_dims[i-1], hidden_dims[i])
            
            self.params['W{}'.format(i+1)] = weight_scale * weights
            self.params['b{}'.format(i+1)] = np.zeros(hidden_dims[i])
            
        weights = np.random.randn(hidden_dims[self.num_hidden-1], num_classes)
        self.params['W{}'.format(self.num_layers)] = weight_scale * weights
        self.params['b{}'.format(self.num_layers)] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            #self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for i in range(0, self.num_hidden):
                bn_param = dict()
                bn_param['running_mean'] = np.zeros(hidden_dims[i], dtype=dtype)
                bn_param['running_var'] = np.zeros(hidden_dims[i], dtype=dtype)
                self.bn_params.append(bn_param)
        
        if self.use_batchnorm:
            for i in range(0, self.num_hidden):
                self.params['gamma{}'.format(i+1)] = np.random.randn(hidden_dims[i]).astype(dtype)
                self.params['beta{}'.format(i+1)] = np.zeros(hidden_dims[i]).astype(dtype)
            #np.random.randn(hidden_dims[i]).astype(dtype)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
      
        
        affine_out_stack = []
        affine_cache_stack = []
        relu_out_stack = []
        relu_cache_stack = []
        batchnorm_out_stack = []
        batchnorm_cache_stack = []
        batchnorm_out_stack = []
        dropout_cache_stack = []
        dropout_out_stack = []
        
        steps = self.num_layers - 1
        inp = X
        for i in range(0, steps):
            
            assert not isinstance(inp, tuple), 'Affine input check!'
            affine_output, affine_cache = affine_forward(inp, self.params['W{}'.format(i+1)], 
                                                         self.params['b{}'.format(i+1)])
            assert not isinstance(affine_output, tuple), 'Affine output check!'
            affine_out_stack.append(affine_output)
            affine_cache_stack.append(affine_cache)
            
            # BN layer
            if self.use_batchnorm:
                gamma = self.params['gamma{}'.format(i+1)]
                beta = self.params['beta{}'.format(i+1)]
                bn_param = self.bn_params[i]
                bn_output, bn_cache = batchnorm_forward(affine_output, gamma, beta, bn_param)
                
                if bn_param['mode'] == 'train':
                    updated_bn_param = bn_cache[6]
                    assert isinstance(updated_bn_param, dict), "Uncorrect object is got from cache"
                    # inside forward procedure running_mean, _var are updated
                    self.bn_params[i].update(updated_bn_param)
                batchnorm_out_stack.append(bn_output)
                batchnorm_cache_stack.append(bn_cache)

                relu_output, relu_cache = relu_forward(bn_output)
            else:
                relu_output, relu_cache = relu_forward(affine_output)
                
            relu_out_stack.append(relu_output)
            relu_cache_stack.append(relu_cache)
            
            assert not isinstance(relu_output, tuple), 'Before dropout'
            
            # dropout layer
            if self.dropout_param.get('p') is not None:
                if self.dropout_param.get('p') > 0:
                    relu_output, d_cache = dropout_forward(relu_output, self.dropout_param)
                    dropout_cache_stack.append(d_cache)
                    dropout_out_stack.append(relu_output)
            
            assert not isinstance(relu_output, tuple), 'After dropout'
            inp = relu_output
            
            if i == steps - 1:
                last_relu = relu_output
        
        assert len(relu_out_stack) == len(affine_out_stack)
        assert len(relu_out_stack) == self.num_layers - 1
        
        output, out_cache = affine_forward(last_relu, self.params['W{}'.format(self.num_layers)], 
            self.params['b{}'.format(self.num_layers)])
        scores = output
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, loss_grad = softmax_loss(scores, y)
        
        W_params = [self.params['W{}'.format(i+1)] for i in range(self.num_layers)]
        L2_reg = 0.0
        for w in W_params:
            L2_reg += np.sum(w*w)
        #L2_reg = [np.sum(w*w) for w in W_params]
        #L2_reg = reduce((lambda x, y: x + y), L2_reg)
        loss += 0.5 * self.reg * L2_reg
        
        N = self.num_layers
        dx0, dw0, db0 = affine_backward(loss_grad, out_cache)
        grads['W{}'.format(N)] = dw0 + self.reg * self.params['W{}'.format(N)]
        grads['b{}'.format(N)] = db0
        
        # backward
        last_dx = dx0
        for i in range(0, N-1):  # 0, 1 (N==3)
            j = i + 1 # 1, 2
            if self.dropout_param.get('p') is not None:
                last_dx = dropout_backward(last_dx, dropout_cache_stack[-j])
            rdx = relu_backward(last_dx, relu_cache_stack[-j])
            if self.use_batchnorm:
                dx, dgamma, dbeta = batchnorm_backward(last_dx, batchnorm_cache_stack[-j])
                dx, dw, db = affine_backward(dx, affine_cache_stack[-j])
                grads['gamma{}'.format(N-j)] = dgamma
                grads['beta{}'.format(N-j)] = dbeta
            else:    
                dx, dw, db = affine_backward(rdx, affine_cache_stack[-j])
            last_dx = dx
                      
            grads['W{}'.format(N-j)] = dw + self.reg * self.params['W{}'.format(N-j)]
            grads['b{}'.format(N-j)] = db
            
        #print(grads.keys())
        #raise Exception('')
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
