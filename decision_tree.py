import pandas as pd
import numpy as np
from data import get_x_y
from scipy.stats import entropy


splits, test_set = get_x_y('./processed.csv')
def entropy_function(nodes):
    e = np.sum(nodes=='e')
    p=np.sum(nodes=='p')
    
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
    if(max_gain<0.01 or iterations==0):     
        if np.sum(y_train == 'e') > np.sum(y_train == 'p'):
            decision_tree[index] = -1  # 'e'
        else:
            decision_tree[index] = -2  # 'p'
        return
    decision_tree[index] = max_gain_index
    if (iterations > 0):
        construct_decision_tree(left_x_train, left_y_train,decision_tree, 2*index+1, iterations-1)
        construct_decision_tree(right_x_train, right_y_train, decision_tree, 2*index+2, iterations-1)
    return 0

def test(decision_tree, x_val, y_val,):
    success =0
    fail = 0
    e = 0
    p = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    #e is negative(safe to eat)
    #p is positive(unsafe to eat)
    for i in range(len(x_val)):
        prediction = predict(decision_tree,x_val[i])
        if prediction=='e' and prediction==y_val[i]:
            e+=1
            success+=1
            true_neg+=1
        elif prediction=='e' and prediction!=y_val[i]:
            e+=1
            fail+=1
            false_neg+=1
        elif prediction=='p' and prediction==y_val[i]:
            p+=1
            success+=1
            true_pos+=1
        else:
            p+=1
            fail+=1
            false_pos+=1
    print(f"{success+fail} tests")
    print(f"Model predicted {e} times e")
    print(f"Model predicted {p} times p")
    print(f"true positive: {true_pos}")
    print(f"true negative: {true_neg}")
    print(f"false positive: {false_pos}")
    print(f"false negative: {false_neg}")
    print(f"Successful attempts: {success}")
    print(f"Failed attempts: {fail}")
    print(f"overall success: %{(success/(success+fail)*100):.2f}")

    return prediction

def predict(decision_tree,x_in,index=0):
    feature_index = int(decision_tree[index])
    if feature_index==0:
        print("feature index is 0, possible error")
    if feature_index == -1:
        return 'e'
    elif feature_index == -2: 
        return 'p'
    if x_in[feature_index]:
        index = index*2 + 2
    else: index = index*2+1
    if index >= len(decision_tree): 
        print(f"ERROR, PREDICTION OUT OF BOUNDS")
    return predict(decision_tree,x_in,index)
       
 
def cross_testing(splits):
    for i in range(9):
        decision_tree = np.zeros(60)
        x_train,y_train,x_val,y_val = splits[i]
        construct_decision_tree(x_train,y_train,decision_tree)
        print('\n')
        print(f"MODEL {i+1}'S SUCCES RATE:")
        test(decision_tree,x_val,y_val)
        #output results say that model 4 and 8 is the best wint %99.45 success rate

def final_test(splits):
    decision_tree=np.zeros(60)
    x_train,y_train,_,_ = splits[3]
    construct_decision_tree(x_train,y_train,decision_tree)
    print("FINAL TEST RESULTS")
    print(test_set[0].shape)
    test(decision_tree, test_set[0], test_set[1])
    return 0

def random_forest(splits):
    forest = []
    for i in range(9):
        x_train, y_train, _, _ = splits[i]
        n=2000
        for k in range(11):
            indices = np.random.choice(x_train.shape[0], size=n, replace=True)
            x_random = x_train[indices]
            y_random = y_train[indices]
            decision_tree = np.zeros(60)
            construct_decision_tree(x_random,y_random,decision_tree)
            forest.append(decision_tree)

    
    return forest

def final_test_forest(splits,test_set):
    forest = random_forest(splits)
    x_test = test_set[0]
    y_test = test_set[1]
    success = 0
    fail = 0
    for i in range(x_test.shape[0]):
        e_vote = 0
        p_vote = 0
        for j in range(len(forest)):
            prediction = predict(forest[j],x_test[i])
            if prediction == 'e': e_vote+=1
            else: p_vote +=1
        if (p_vote>e_vote): consensus='p'
        else: consensus='e'
        if consensus==y_test[i]: success+=1
        else: fail+=1
    total_tests = success + fail
    accuracy = (success / total_tests * 100) if total_tests > 0 else 0
    print("--- Random Forest Model Evaluation on Test Set ---")
    print(f"Number of trees in the forest: {len(forest)}")
    print(f"Total tests: {total_tests}")
    print(f"Successful predictions: {success}")
    print(f"Failed predictions: {fail}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    return 0



final_test_forest(splits,test_set)
