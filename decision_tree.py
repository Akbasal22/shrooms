import pandas as pd
import numpy as np
from data import get_x_y
from scipy.stats import entropy


splits = get_x_y('./processed.csv')
def entropy_function(nodes):
    e = 0
    p = 0
    for i in range(len(nodes)):
        if(nodes[i] == 'e'):
            e=e+1
        else:
            p=p+1
    
    prob = e/(e+p)
    base=2
    pk = np.array([prob, 1-prob])
    return entropy(pk, base=base)


def calculate_information_gain(x_train,y_train, index):
    left_x_train = []
    right_x_train = []
    left_y_train = []
    right_y_train = []
    right_count = 0
    left_count = 0
    for i in range(len(x_train)):
        if x_train[i][index]:
            right_x_train.append(x_train[i])
            right_y_train.append(y_train[i])
            right_count+=1
        else:
            left_x_train.append(x_train[i])
            left_y_train.append(y_train[i])
            left_count+=1
    right_weight = right_count/(right_count+left_count)
    left_weight = left_count/(right_count+left_count)
    if left_count == 0 or right_count == 0:
        return right_x_train , right_y_train , left_x_train , left_y_train , 0
    information_gain = entropy_function(np.array(y_train)) - ((right_weight*entropy_function(np.array(right_y_train)))+(left_weight*entropy_function(np.array(left_y_train))))
    return (
    np.array(right_x_train),
    np.array(right_y_train),
    np.array(left_x_train),
    np.array(left_y_train),
    information_gain
)


def calculate_max_information_gain(x_train,y_train):
    gain = 0
    max_gain = 0
    max_gain_index = 0
    right_x_train = np.array([])
    right_y_train = np.array([])
    left_x_train = np.array([])
    left_y_train = np.array([])

    for i in range(x_train.shape[1]):
        xright_x_train,xright_y_train,xleft_x_train,xleft_y_train,gain = calculate_information_gain(x_train, y_train, i)
        if (gain > max_gain):
            right_x_train,right_y_train,left_x_train,left_y_train =  xright_x_train,xright_y_train,xleft_x_train,xleft_y_train
            max_gain =gain
            max_gain_index=i
    return right_x_train,right_y_train,left_x_train,left_y_train,max_gain, max_gain_index
    


def construct_decision_tree(x_train,y_train,decision_tree,index=0, iterations=5):
    # tree will be an array
    # where root is i, left node is 2i+1 and right node is 2i+2
    right_x_train,right_y_train,left_x_train,left_y_train,max_gain, max_gain_index = calculate_max_information_gain(x_train, y_train)
    if(max_gain<0.001): return
    decision_tree[index] = max_gain_index
    if (iterations > 0):
        construct_decision_tree(left_x_train, left_y_train,decision_tree, 2*index+1, iterations-1)
        construct_decision_tree(right_x_train, right_y_train, decision_tree, 2*index+2, iterations-1)

    
 
    return 0

x_train,y_train,x_val,y_val = splits[0]

decision_tree = np.zeros(33)
construct_decision_tree(x_train,y_train,decision_tree)
print(decision_tree)

