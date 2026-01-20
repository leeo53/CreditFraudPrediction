import numpy as np
import random as rand
import numba
from multiprocessing import Pool

def build_tree_helper(args):
    rf_instance, seed = args
    np.random.seed(seed)  # ensure reproducibility
    sample_indices = np.random.choice(rf_instance.numSamples, rf_instance.numSamples, replace=True)
    return rf_instance._build_tree(sample_indices)

class RandomForest:
    class Node:
        def __init__(self,feature_index, threshold, children, prediction = None):
            if prediction is not None:
                self.is_leaf = True
                self.prediction = prediction
            else:
                self.is_leaf = False
                self.left = children[0]
                self.right = children[1]
                self.feature_index = feature_index
                self.threshold = threshold

    def __init__(self,X,y, n_trees, max_depth, min_samples_split):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.X=X
        self.y=y
        self.numFeatures = X.shape[1]
        self.numSamples = X.shape[0]
        self.numClasses = np.unique(y).size

    # Train Forest on dataset
    # For each tree
    #   take a sample of X and y
    #   build a decision tree using the chosen features
    #   store the tree in trees
    def fit(self, num_cores=1):
        self.trees = []
        if num_cores == 1:
            for i in range(self.n_trees):
                sample_indices = np.random.choice(self.numSamples,self.numSamples,replace=True)
                self.trees.append(self._build_tree(sample_indices))
        else:
            with Pool(num_cores) as pool:
                # pass self along with unique seeds for reproducibility
                args = [(self, i) for i in range(self.n_trees)]
                self.trees = pool.map(build_tree_helper, args)


    # Make predictions using the trained forest
    # Output: 1D array of predicted class labels (majority vote across trees)
    # For each tree, get its prediction for X.
    # Aggregate the results (majority vote)
    def predict(self,X_input):
        predictions = []
        for tree in self.trees:
            predictions.append(self._predict_tree(tree, X_input))
        predictions = np.array(predictions)
        final_predictions = []
        for i in range(X_input.shape[0]):
            sample_predictions = predictions[:,i]
            counts = np.bincount(sample_predictions, minlength=self.numClasses)
            fraud_votes = counts[1]
            if fraud_votes / self.n_trees > .52:
                final_predictions.append(1)
            else:
                final_predictions.append(0)
        return np.array(final_predictions)

    # Recursively build a single decision tree
    # Output: Node containing tree
    # If stopping criteria are met return leaf node with prediction
    # Else:
    #   Split X and y into left and right subsets
    #   recursively call on left and right subsets
    #   return a node storing split infor and child nodes.
    def _build_tree(self, indices, depth=0):
        all_features = np.arange(self.numFeatures)
        max_features = int(np.sqrt(all_features.size))
        feature_index, threshold = self._best_split(indices,rand.sample(all_features.tolist(),max_features))
        counts = get_class_counts(self.y, indices, self.numClasses)
        length = np.count_nonzero(counts)
        if length == 1 or feature_index is None or len(indices) < self.min_samples_split or depth >= self.max_depth:
            tree = self.Node(None, None, None, prediction=np.argmax(counts))
        else:
            left_indices, right_indices = self.split_indices(indices, feature_index, threshold)
            children = [self._build_tree(left_indices, depth + 1),
                        self._build_tree(right_indices, depth + 1)]
            tree = self.Node(feature_index, threshold, children)
        return tree

    # Find the best feature and threshold to split the current node
    # For each feature, try possible thresholds
    # Evaluate split using Gini impurity metric
    # return the split that minimizes impurity
    def _best_split(self, indices, features):
        best_impurity = float('inf')
        feature_index = -1
        best_threshold = 0
        total_samples = len(indices)
        for feature in features:
            unique_vals = np.unique(self.X[indices, feature])
            max_thresholds = 50
            if len(unique_vals) > max_thresholds:
                unique_vals = np.random.choice(unique_vals, size=max_thresholds, replace=False)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for threshold in thresholds:
                left_indices, right_indices = self.split_indices(indices,feature, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # skip this threshold entirely
                else:
                    gini_impurity = (len(left_indices) / total_samples) * self._gini(left_indices) + (len(right_indices) / total_samples) * self._gini(right_indices)
                    if gini_impurity < best_impurity:
                        best_impurity = gini_impurity
                        best_threshold = threshold
                        feature_index = feature
        if feature_index == -1:
            return None, None
        return feature_index, best_threshold

    # Make predictions for a single tree
    # Traverse the tree from root to leaf for each sample
    # At leaf node, return prediction value or class label
    def _predict_tree(self, tree, X_input):
        predictions = []
        for sample in X_input:
            temp_node = tree
            while not temp_node.is_leaf:
                if sample[temp_node.feature_index] <= temp_node.threshold:
                    temp_node = temp_node.left
                else :
                    temp_node = temp_node.right
            predictions.append(temp_node.prediction)
        return predictions

    def _gini(self, indices):
        counts = get_class_counts(self.y, indices, self.numClasses)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def split_indices(self, indices, feature_index, threshold):
        left = []
        right = []
        for i in indices:
            if self.X[i, feature_index] <= threshold:
                left.append(i)
            else:
                right.append(i)
        return left, right

@numba.njit
def get_class_counts(y,indices,num_classes):
    counts = np.zeros(num_classes, dtype=np.int32)
    for i in indices:
        counts[y[i]] += 1
    return counts