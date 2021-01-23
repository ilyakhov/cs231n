import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i,:].dot(W)
    f_max = np.max(scores)
    scores -= f_max
    correct_class_score = scores[y[i]]
    scores_exp = np.exp(scores)
    norm_base = np.sum(scores_exp)
    loss += -1 * np.log(np.exp(correct_class_score)/norm_base)
    
    for num_class in range(scores.shape[0]):
        if num_class != y[i]:
            dW[:, num_class] += X[i] * scores_exp[num_class]/norm_base
        else:
            dW[:, num_class] += X[i] * (scores_exp[num_class]/norm_base - 1)
    
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_train = X.shape[0]
  
  scores = X.dot(W)
  f_max = np.max(scores, axis=1)
  scores -= np.broadcast_to(np.reshape(f_max, (f_max.shape[0], 1)), (X.shape[0], W.shape[1]))
  correct_class_score = scores[np.arange(y.shape[0]), y]
  scores_exp = np.exp(scores)
  norm_base = np.sum(scores_exp, axis=1)
  loss = np.mean(-1 * np.log(np.exp(correct_class_score)/norm_base))

  loss += reg * np.sum(W * W)
   
  norm_base = np.reshape(norm_base, (norm_base.shape[0], 1))
  vector_softmax = np.divide(scores_exp, norm_base)
  vector_softmax[np.arange(vector_softmax.shape[0]), y] -= 1
  x_sum = X.T.dot(vector_softmax)
  dW += x_sum
    
  dW /= num_train
  dW += reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

