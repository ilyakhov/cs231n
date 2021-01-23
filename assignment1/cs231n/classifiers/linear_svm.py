import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # X[i].dot(W[:,y[i]])
    margin_counter = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        margin_counter += 1
        loss += margin
        dW[:,j] += X[i]
    dW[:,y[i]] += -margin_counter*X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * np.sum(2 * dW) # reg * np.sum(2 * dW) #np.sum(dW * dW)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # READ this: http://cs231n.github.io/optimization-1/#gradcompute

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  
    Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)  # N * C
  correct_classes = scores[:,y]
    
  correct_scores = scores[np.arange(scores.shape[0]), y]
  correct_scores = np.reshape(correct_scores, (correct_scores.shape[0],1))
  correct_scores_matrix = np.broadcast_to(correct_scores, correct_scores.shape)
  Margins = scores - correct_scores_matrix + 1.0
  Margins[np.arange(Margins.shape[0]), y] = 0.0
  Margins[Margins < 0] = 0.0
  loss = np.sum(Margins)
  loss /= num_train
  
  loss += reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  nonzero_margins = Margins
  nonzero_margins[nonzero_margins > 0.0] = 1.0
  nonzero_margins_count = nonzero_margins.sum(axis=1)
  nonzero_margins[np.arange(nonzero_margins.shape[0]), y] = -1 * nonzero_margins_count
    
  #for i in range(num_train):
  #  slice_X = np.reshape(X[i,:], (X[i,:].shape[0], 1))
  #  slice_margins = np.reshape(nonzero_margins[i,:], (1, nonzero_margins[i,:].shape[0]))
  #  dW += slice_X.dot(slice_margins)
    
  dW += X.T.dot(nonzero_margins)
  
  dW /= num_train
  dW += reg * np.sum(2*dW)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
