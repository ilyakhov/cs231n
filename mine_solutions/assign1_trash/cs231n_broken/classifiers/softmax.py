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
<<<<<<< HEAD
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
=======
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
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        # classification scores for the current sample, shape (1, num_classes)
        scores = X[i].dot(W)
        # just a scalar value
        correct_class_score = scores[y[i]]
        # compute softmax loss for the current sample
        loss += -correct_class_score + np.log(np.exp(scores).sum())
        # compute gradients w.r.t. to weights for each class separately
        for j in range(num_classes):
            dW[:, j] += np.exp(scores[j]) / np.exp(scores).sum() * X[i, :]
        # the formula for gradients w.r.t. to te weights of the correct class
        # is slightly different http://cs231n.github.io/optimization-1/
        dW[:, y[i]] -= X[i, :]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W
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
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    # select in each row i the score at position y[i]
    # the formula is given here http://cs231n.github.io/linear-classify/#softmax
    correct_class_scores = scores[range(num_train), y]
    # compute softmax loss
    # TODO: sometimes overflows, need to apply the log-sum-exp trick
    # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
    loss = - correct_class_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # compute softmax derivatives w.r.t. to scores, shape (num_train, num_classes)
    softmax_deriv = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1,1))
    softmax_deriv[range(num_train), y] -= 1
    # compute softmax gradients w.r.t. to weights W, shape (num_features, num_classes)
    dW = X.T.dot(softmax_deriv)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
>>>>>>> 42d14625af2f094e8f3c7a27b94ad0148c8de1d4

