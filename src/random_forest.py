import numpy as np
import random as rand

n_trees = 100
max_depth = 10
min_samples_split = 10
trees = []
bootstrap = True

class Node:
    def __init__(self,feature_index, threshold, children, prediction = None):
        if not children:
            self.is_leaf = True
            self.prediction = prediction
        else:
            self.is_leaf = False
            self.left = children[0]
            self.right = children[1]
            self.feature_index = feature_index
            self.threshold = threshold


def fit(X,y):
    for i in range(n_trees):
        n_samples = X.shape[0]
        sample_indices = np.random.choice(n_samples,n_samples,replace=True)
        sample_X = X[sample_indices]
        sample_y = y[sample_indices]
        trees.append(_build_tree(sample_X,sample_y))

def predict(X):
    predictions = []
    for tree in trees:
        predictions.append(_predict_tree(tree,X))
    predictions = np.array(predictions)
    final_predictions = []
    for i in range(X.shape[0]):
        sample_predictions = predictions[:,i]
        values, counts = np.unique(sample_predictions, return_counts=True)
        final_predictions.append(values[np.argmax(counts)])
    return np.array(final_predictions)

def _build_tree(X, y, depth=0):
    all_features = np.arange(X.shape[1])
    max_features = int(np.sqrt(all_features.size))
    feature_index, threshold = _best_split(X,y,rand.sample(all_features.tolist(),max_features))
    if len(np.unique(y)) == 1 or feature_index is None or len(y) < min_samples_split or depth >= max_depth:
        values, counts = np.unique(y, return_counts=True)
        tree = Node(None, None, None, prediction=values[np.argmax(counts)])
    else:
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        children = [_build_tree(X[left_mask], y[left_mask], depth + 1),
                    _build_tree(X[right_mask], y[right_mask], depth + 1)]
        tree = Node(feature_index, threshold, children)
    return tree

def _best_split(X, y, features):
    best_impurity = float('inf')
    feature_index = -1
    best_threshold = 0
    for feature in features:
        unique_vals = np.unique(X[:, feature])
        thresholds = np.linspace(unique_vals.min(), unique_vals.max(), num=20)
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue  # skip this threshold entirely
            y_left = y[left_mask]
            y_right = y[right_mask]
            if y_left.size > 0 and y_right.size > 0:
                counts_l = np.array([np.sum(y_left),y_left.size-np.sum(y_left)])
                counts_r = np.array([np.sum(y_right),y_right.size-np.sum(y_right)])
                gini_impurity_left = 1 - np.sum((counts_l / y_left.size) ** 2)
                gini_impurity_right = 1 - np.sum((counts_r / y_right.size) ** 2)
                gini_impurity = (y_left.size / y.size) * gini_impurity_left + (y_right.size / y.size) * gini_impurity_right
                if gini_impurity < best_impurity:
                    best_impurity = gini_impurity
                    best_threshold = threshold
                    feature_index = feature
    if feature_index == -1:
        return None, None
    return feature_index, best_threshold

def _predict_tree(tree,X):
    predictions = []
    for sample in X:
        temp_node = tree
        while not temp_node.is_leaf:
            if sample[temp_node.feature_index] <= temp_node.threshold:
                temp_node = temp_node.left
            else :
                temp_node = temp_node.right
        predictions.append(temp_node.prediction)
    return predictions