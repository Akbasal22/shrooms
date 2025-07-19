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
        if x_train.iloc[i,index]:
            right_x_train.append(x_train.iloc[i])
            right_y_train.append(y_train[i])
            right_count+=1
        else:
            left_x_train.append(x_train.iloc[i])
            left_y_train.append(y_train[i])
            left_count+=1
    right_weight = right_count/(right_count+left_count)
    left_weight = left_count/(right_count+left_count)
    if left_count == 0 or right_count == 0:
        return 0,index
    information_gain = entropy_function(np.array(y_train)) - (right_weight*entropy_function(np.array(right_y_train)))+(left_weight*entropy_function(np.array(left_y_train)))
    
    return information_gain,index






x_train,y_train,x_val,y_val = splits[0]
print(x_train.shape)
print(y_train.shape)

print(calculate_information_gain(x_train,y_train,2))
