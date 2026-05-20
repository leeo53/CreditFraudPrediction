import numpy as np
import numba
import math

class GradientBoosting:
    class Node:
        def __init__(self, feature_index, threshold, children, prediction=None):
            if prediction is not None:
                self.is_leaf = True
                self.prediction = prediction
            else:
                self.is_leaf = False
                self.left = children[0]
                self.right = children[1]
                self.feature_index = feature_index
                self.threshold = threshold

    def __init__(self,X,y,n_estimators, learning_rate, max_depth, min_samples):
        self.X = X
        self.y = y
        eps = 1e-15
        y_mean = np.clip(np.mean(y), eps, 1 - eps)
        self.model = np.full(X.shape[0], math.log(y_mean/(1-y_mean)))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.lmbda = 1
        self.trees = []

    #builds n_estimators trees each based on the residuals of the last tree.
    #after each tree is constructed the model is adjusted based on the built trees predictions.
    def fit(self):
        all_indices = np.arange(self.X.shape[0])
        self.trees = []
        for i in range(self.n_estimators):
            residuals = self.y - self._sigmoid(self.model)
            tree = self._build_tree(all_indices, residuals)
            self.trees.append(tree)
            self.model = self.model + self.learning_rate * self._predict_tree(tree, self.X)
            
    #we start out with the log odds of the mean of the training data and walk through the trees in order computing
    #the corrections each tree predicts and applying it to the output F_O. sigmoid turns F_0 from log odds into probabilities.
    #returned probabilities can be used for threshold tuning.
    def predict(self,X_input):
        eps = 1e-15  # small constant to avoid log(0)
        y_mean = np.clip(np.mean(self.y), eps, 1 - eps)
        F_0 = np.full(X_input.shape[0], math.log(y_mean/(1-y_mean)))
        for tree in self.trees:
            F_0 = F_0 + self.learning_rate * self._predict_tree(tree, X_input)
        probabilities = self._sigmoid(F_0)
        y_pred = (probabilities > 0.5).astype(int)
        return y_pred, probabilities

    #walk through a single tree finding the correction it predicts for each sample
    def _predict_tree(self, tree, X_input):
        predictions = np.zeros(X_input.shape[0])
        for idx, sample in enumerate(X_input):
            temp_node = tree
            while not temp_node.is_leaf:
                if sample[temp_node.feature_index] <= temp_node.threshold:
                    temp_node = temp_node.left
                else:
                    temp_node = temp_node.right
            predictions[idx] = temp_node.prediction
        return predictions

    #uses the feature index and threshold found using _best_split to create the branching decisions recursively.
    #if we are at the max_depth or the number of indices is too small to split we create a leaf and store the mean
    #prediction at those indices.
    def _build_tree(self, indices, residuals, depth=0):
        feature_index, threshold = self._best_split(indices, residuals)
        if depth == self.max_depth or feature_index is None or len(indices) < self.min_samples:
            p = self._get_probs(indices)
            #prediction = _get_pred(self.y,indices, p, self.lmbda)
            prediction = np.mean(self.y[indices] - p)
            #print(prediction)
            tree = self.Node(None, None, None, prediction = prediction)
        else:
            left_indices, right_indices = self._split_indices(indices, feature_index, threshold)
            children = [self._build_tree(left_indices, residuals, depth + 1),
                        self._build_tree(right_indices, residuals, depth + 1)]
            tree = self.Node(feature_index, threshold, children)
        return tree

    #finds the best feature_index and threshold that maximizes the gain
    def _best_split(self, indices, residuals):
        best_gain = -np.inf
        feature_index = -1
        best_threshold = 0
        for i in range(self.X.shape[1]):
            unique_vals = np.unique(self.X[indices, i])
            max_thresholds = 50
            if len(unique_vals) > max_thresholds:
                unique_vals = np.random.choice(unique_vals, size=max_thresholds, replace=False)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for threshold in thresholds:
                left_indices, right_indices = self._split_indices(indices, i, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # skip this threshold entirely
                else:
                    probs_l = self._get_probs(left_indices)
                    probs_r = self._get_probs(right_indices)
                    if np.sum(probs_l * (1 - probs_l)) == 0 or np.sum(probs_r * (1 - probs_r)) == 0:
                        continue  # skip this split entirely
                    gain = _get_gain(self.y, left_indices,right_indices,probs_l, probs_r)
                    if gain > best_gain:
                        best_gain = gain
                        feature_index = i
                        best_threshold = threshold
        if feature_index == -1:
            return None, None
        return feature_index, best_threshold

    #numerically stable version of sigmoid
    def _sigmoid(self, x):
        out = np.zeros_like(x, dtype=np.float64)
        mask = x >= 0
        out[mask] = 1 / (1 + np.exp(-x[mask]))
        out[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
        return out

    #helper for getting the probabilities easily
    def _get_probs(self, indices):
        probs = self._sigmoid(self.model[indices])
        return probs

    def _split_indices(self, indices, feature_index, threshold):
        left = []
        right = []
        for i in indices:
            if self.X[i, feature_index] <= threshold:
                left.append(i)
            else:
                right.append(i)
        return np.array(left), np.array(right)

@numba.njit
def _get_pred(y, indices, probs, lmbda):
    numerator = 0
    denominator = 0
    eps = 1e-10
    for idx, i in enumerate(indices):
        numerator += y[i] - probs[idx]
        denominator += probs[idx]*(1 - probs[idx])
    return numerator / (denominator + lmbda)

@numba.njit
def _get_gain(y, left_indices, right_indices, probs_l, probs_r):
    gradients_l = 0
    hessians_l = 0
    eps = 1e-10
    for idx, i in enumerate(left_indices):
        gradients_l += y[i] - probs_l[idx]
        hessians_l += probs_l[idx]*(1 - probs_l[idx])
    hessians_l += eps
    gradients_r = 0
    hessians_r = 0
    for idx, i in enumerate(right_indices):
        gradients_r += y[i] - probs_r[idx]
        hessians_r += probs_r[idx]*(1 - probs_r[idx])
    hessians_r += eps
    return (gradients_l ** 2 / hessians_l) + (gradients_r ** 2 / hessians_r) - ((gradients_l + gradients_r) ** 2 / (hessians_l + hessians_r))