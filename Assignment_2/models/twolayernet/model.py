from abc import abstractmethod
from copy import deepcopy

import numpy as np

from utils.activation_funtions import relu, softmax


class TwoLayerNetv1(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dropout: The dropout probability between the two layers. If dropout is 0, then no dropout
                   is applied.
        """
        np.random.seed(0)
        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
        self.dropout = dropout

    def forward(self, X):
        """
        Compute the final outputs for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        A matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        """

        # Unpack variables from the params dictionary
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # Compute the forward pass
        softmax_scores = None
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        Z2 = X @ W1 + b1
        self.Z2 = Z2
        A2 = relu(Z2) #ReLU
        self.A2 = A2
        Z3 = A2 @ W2 + b2
        softmax_scores = softmax(Z3) #Softmax

        # scores shape: (N, C)
        self.scores = softmax_scores

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        return softmax_scores

    @abstractmethod
    def compute_loss(self, **kwargs):
        raise NotImplementedError


class TwoLayerNetv2(TwoLayerNetv1):

    def compute_loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # Compute the forward pass
        softmax_scores = None
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        softmax_scores = self.forward(X)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return softmax_scores

        # Compute the loss
        loss = 0.0

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss = np.average(-np.log(softmax_scores[np.arange(N), y])) + reg * (
            np.sum(W1 * W1) + np.sum(W2 * W2)
        )

        self.loss = loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss

    @abstractmethod
    def back_propagation(self, **kwargs):
        raise NotImplementedError  # No need to implement here!


class TwoLayerNetv3(TwoLayerNetv2):

    def back_propagation(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        # Compute the forward pass
        scores = 0.0

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****

        scores = self.forward(X)
      
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.0
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss = self.compute_loss(X, y, reg)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        A3 = self.scores
        C = A3.shape[1]

        pJ_pZ3 = (self.scores - np.eye(self.scores.shape[1])[y]) / N
        assert pJ_pZ3.shape == (N, C)

        pJ_pW2 = self.A2.T @ pJ_pZ3 + 2 * reg * W2
        assert pJ_pW2.shape == W2.shape
        grads["W2"] = pJ_pW2

        pJ_pb2 = pJ_pZ3.sum(axis=0)
        assert pJ_pb2.shape == b2.shape
        grads["b2"] = pJ_pb2

        pJ_pA2 = pJ_pZ3 @ W2.T
        pJ_pZ2 = []
        for n in range(pJ_pA2.shape[0]):
            pJ_pZ2.append(pJ_pA2[n] @ (np.diag(self.Z2[n]) > 0))
        pJ_pZ2 = np.array(pJ_pZ2)

        pJ_pW1 = X.T @ pJ_pZ2 + 2 * reg * W1
        assert pJ_pW1.shape == W1.shape
        grads["W1"] = pJ_pW1

        pJ_pb1 = pJ_pZ2.sum(axis=0)
        assert pJ_pb1.shape == b1.shape
        grads["b1"] = pJ_pb1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError


class TwoLayerNetv4(TwoLayerNetv3):

    def train(
        self,
        X,
        y,
        X_val,
        y_val,
        learning_rate=1e-3,
        learning_rate_decay=0.95,
        reg=5e-6,
        num_iters=100,
        batch_size=200,
        verbose=False,
        restore_best_weights=False,
        validation_metric="train_loss",
    ):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        - restore_best_weights: boolean; if true restore the weights corresponding to the best
          validation accuracy or loss (depending on `validation_metric`).
        - validation_metric: string; either "val_acc" or "train_loss". The metric used to determine
          the best weights.
        """
        assert validation_metric in ("train_loss", "val_acc")
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_loss = np.inf
        best_acc = 0
        best_at = ""

        for it in range(num_iters):
            X_batch = X
            y_batch = y

            #########################################################################
            # Create a random minibatch of training data and labels, storing        #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            if batch_size > num_train:
                rand_ind = np.random.choice(num_train, size=batch_size, replace=True)
            else:
                rand_ind = np.random.choice(num_train, size=batch_size, replace=False)
            X_batch = X[rand_ind]
            y_batch = y[rand_ind]
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.back_propagation(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            # Do not forget to apply the learning_rate                              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print(f"iteration {it} / {num_iters}: loss {loss}", end="\r")

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                if best_loss > loss and validation_metric == "train_loss":
                    best_loss = loss
                    best_weights = deepcopy(self.params)
                    best_at = f"{it} / {num_iters}"
                elif best_acc < val_acc and validation_metric == "val_acc":
                    best_acc = val_acc
                    best_weights = deepcopy(self.params)
                    best_at = f"{it} / {num_iters}"

                # Decay learning rate
                learning_rate *= learning_rate_decay

        if restore_best_weights:
            self.params = best_weights
            if verbose:
                print(
                    f"Restored best weights at {best_at} with {validation_metric} = "
                    + f"{best_loss if validation_metric == 'loss' else best_acc}"
                )

        return {
            "loss_history": loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        y_pred = np.argmax(self.forward(X), axis=1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
