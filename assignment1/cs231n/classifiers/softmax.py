import numpy as np
from random import shuffle

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
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp = np.exp(scores)
    denom = np.sum(exp)
    output = exp / denom
    loss += -np.log(output[y[i]])
    target = np.zeros(num_classes)
    target[y[i]] = 1

    dW += np.matrix(X[i,:]).T.dot(np.matrix(output - target))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # Shift for numerical stability
  # instead: first shift the values of f so that the highest number is 0:
  scores -= np.max(scores)

  exp = np.exp(scores)
  denom = np.sum(exp, axis=1)

  output = (exp.T/denom).T
  losses = -np.log(output[range(num_train), y])
  loss = np.mean(losses)
  targets = np.zeros_like(output)
  targets[range(num_train), y] = 1
  # import pdb; pdb.set_trace()

  dW = X.T.dot(output - targets) / num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

