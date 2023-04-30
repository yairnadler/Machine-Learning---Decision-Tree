import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    S = len(data)
    num_edible = np.sum(data[:,-1] == 'e')
    num_poisonous = S - num_edible
    gini = 1 - (num_edible/S)**2 - (num_poisonous/S)**2

    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    S = len(data)
    num_edible = np.sum(data[:,-1] == 'e')
    num_poisonous = S - num_edible
    if num_edible == 0 or num_poisonous == 0:
        return 0
    p_e = num_edible/S
    p_p = num_poisonous/S
    entropy = -p_e * np.log2(p_e) - p_p * np.log2(p_p)
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    S = len(data)

    for instance in data:
        if instance[feature] not in groups:
            groups[instance[feature]] = np.array([instance])
        else:
            groups[instance[feature]] = np.vstack((groups[instance[feature]], instance))

    goodness = (impurity_func(data) - sum([len(group)/S * impurity_func(group) for group in groups.values()]))


    if gain_ratio:
        # calculating the split info
        split_info = 0
        split_info = np.sum([-len(group)/ S * np.log2(len(group)/S) for group in groups.values()])
        goodness = goodness/split_info

    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        #calculating the prediction of the node
        if type(self.data) == DecisionNode:
            pred = 'e' if np.sum(self.data.data[:,-1] == 'e') > np.sum(self.data.data[:,-1] == 'p') else 'p'
        else:
            pred = 'e' if np.sum(self.data[:,-1] == 'e') > np.sum(self.data[:,-1] == 'p') else 'p'
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        # check if we reached the max depth or if all instances have the same label
        if self.depth >= self.max_depth or len(np.unique(self.data[:,-1])) == 1:
            self.terminal = True
            return
            
        
        # find best feature to split according to
        best_feature = None
        best_goodness = -np.inf
        best_groups = None
        for feature in range(self.data.shape[1]-1):
            goodness, groups = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)
            if goodness > best_goodness:
                best_feature = feature
                best_goodness = goodness
                best_groups = groups
        
        # check if we can't split anymore
        if best_goodness == 0:
            self.terminal = True
            return
        
        # calculate the node's chi squared value and the degree of freedom
        chi_calc, degree_of_freedom = self.chi_squared(groups)
        # if the node's chi field is less than 1, we need to check if the chi squared value is less than the
        # alpha risk value in the chi table. If it is, the node is a terminal node.
        if self.chi < 1 and degree_of_freedom > 0 and chi_calc < chi_table.get(degree_of_freedom).get(self.chi):
            self.terminal = True
            return
        
        self.feature = best_feature
        
        # create children nodes
        for feature_value in best_groups.keys():
            child_data = best_groups[feature_value]
            child_node = DecisionNode(child_data, feature=best_feature, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth)
            self.add_child(child_node, feature_value)

    def chi_squared(self, groups):
        """
        Calculate the chi squared value of the node, and the degree of freedom.

        Input:
        - data: the data of the node
        - groups: the groups divided by the feature values

        Returns:
        - chi_squared: the chi squared value of the node
        - degree_of_freedom: the degree of freedom of the node
        """
        chi = 0
        degree_of_freedom = 0
        S = len(self.data)
        num_edible = np.sum(self.data[:,-1] == 'e')
        num_poisonous = len(self.data) - num_edible
        
        for group in groups.values():
            group_size = len(group)
            group_num_edible = np.sum(group[:,-1] == 'e')
            group_num_poisonous = group_size - group_num_edible
            chi += (group_num_edible - (num_edible * group_size / S)) ** 2 / (num_edible * group_size / S)
            chi += (group_num_poisonous - (num_poisonous * group_size / S)) ** 2 / (num_poisonous * group_size / S)
        
        degree_of_freedom = len(groups) - 1

        return chi, degree_of_freedom

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """

    root = DecisionNode(data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    tree = recursive_build_tree(root, impurity)
    return tree

def recursive_build_tree(root, impurity):
    """
    Recursively build the tree.

    Input:
    - node: the current node.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root.split(impurity)
    for i in range(len(root.children)):
        recursive_build_tree(root.children[i], impurity)

    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    while not root.terminal:
        feature = root.feature
        feature_value = instance[feature]
        isPassed = False
        for i in range(len(root.children)):
            if root.children_values[i] == feature_value:
                root = root.children[i]
                isPassed = True
                break

        if not isPassed:
            return None
        
    pred = root.calc_node_pred()
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    for instance in dataset:
        if predict(node, instance) == instance[-1]:
            accuracy += 1
    accuracy /= len(dataset)
    return accuracy * 100

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, calc_entropy, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))

    return training, testing

def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    for value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train, calc_entropy, chi=value)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        depth.append(calc_tree_depth(tree))
        
    return chi_training_acc, chi_testing_acc, depth

def calc_tree_depth(root):
    """
    Calculate the depth of a given tree
 
    Input:
    - root: the root of the decision tree.
 
    Output: the depth of the decision tree.
    """
    depth = 0
    for child in root.children:
        depth = max(depth, calc_tree_depth(child))
    return depth + 1

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    if node.terminal:
        return 1
    else:
        count = 1
        for child in node.children:
            count += count_nodes(child)
        return count
