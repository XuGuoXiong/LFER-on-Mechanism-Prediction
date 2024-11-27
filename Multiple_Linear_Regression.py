import re
import os
import sys
import pandas as pd
import numpy as np
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import combinations


# Define various parameters
train_input_file = ''                   # Input file name for training dataset
validation_input_file = ''              # Input file name for validation dataset
normalize_method = 0                    # 0: No; 1: max,min normalization; 2: Mean-variance standardization
cross_limit = 2                         # cross_limit: Limit on the number of cross terms; 0: No cross terms; 1 or higher integer: Contains corresponding number of cross terms
cross_type = 1                          # cross_type: Method for constructing cross terms; 1: No normalization, construct cross terms directly; 2: Cross first then normalize; 3: Normalize first then construct cross terms
test_method = -1                        # test_method: Model testing method; -1: Leave-One-Out; Others: k-Fold, input the value of k, (k: 1), represents (training set: test set = k: 1)

feature_selection_method = 1            # feature_selection_method: Feature selection strategy is; 1: Full combination
feature_selection_number = 4            # feature_selection_number: Number of selected features (including cross terms)
constant_threshold = 0.05               # constant_threshold: Threshold for constant term pre-selection of features, [0,1], the larger the value, the stricter the selection condition
relevance_threshold = 0.95              # relevance_threshold: Threshold for relevance pre-selection of features, [0,1], the smaller the value, the stricter the selection condition

required_model_num = 1                  # required_model_num: The top-ranked models needed

start_time = time.time()

def csv_cleaner(file): 

    data = pd.read_csv(file)
    csv_column = data.shape[0]
    csv_row = data.shape[1]
    feature = data.columns

    #读取因变量
    Y_label = feature[0]
    Y = data.iloc[:,0]

    #读取自变量
    X_label = []
    X = []
    for i in range(1,csv_row):
        X_label.append(feature[i])
        X.append(data.iloc[:,i])
        
    return Y_label, Y, X_label, X, csv_column, csv_row


# Feature pre-processing module A, pre-select features based on their correlation and constancy
def feature_pre_selection(feature_num, feature_label, feature, constant_threshold, relevance_threshold):

    feature_num_select = 0
    feature_select_label = []
    feature_select = []

    coef_pre_list = []
    r2_score_pre_list = []

    delete_list_const = []
    delete_index_list_const = []
    delete_list_relav = []
    delete_index_list_relav = []

    if constant_threshold != -1:
        print('(constant_threshold = %.2f)' %constant_threshold)

    print('(relevance_threshold = %.2f)' %relevance_threshold)

    feature_select_label.extend(feature_label)
    feature_select.extend(feature)

    # Pre-selection of constant term features
    if constant_threshold == -1:
        constant_threshold = 0
        for p in range(0, feature_num):
            feature_diff = abs((np.max(feature[p]) - np.min(feature[p]))/(np.max(feature[p])))
            if feature_diff == 0:
                delete_list_const.append(feature_label[p])
                delete_index_list_const.append(p)
                print('Discard possible constant feature:', feature_label[p])
    else:
        for p in range(0, feature_num):
            feature_diff = abs((np.max(feature[p]) - np.min(feature[p]))/(np.max(feature[p])))
            if feature_diff <= constant_threshold:
                delete_list_const.append(feature_label[p])
                delete_index_list_const.append(p)
                print('Discard possible constant feature:', feature_label[p])        

    delete_index_list_const = sorted(set(delete_index_list_const))

    for m in range(0, len(delete_index_list_const)):
        del_no = delete_index_list_const[m]
        del_no_true = del_no - m
        del feature_select_label[del_no_true]
        feature_select_label.sort
        del feature_select[del_no_true]
        feature_select.sort
    
    # Pre-selection of correlated features
    feature_num = feature_num - len(delete_index_list_const)
    for i in range(0, feature_num):
        feature_select_i_ready = np.array(feature_select[i])
        feature_select_i_ready = feature_select_i_ready.reshape(-1, 1)
        for j in range(i + 1, feature_num):

            feature_select_j_ready = np.array(feature_select[j])
            feature_select_j_ready = feature_select_j_ready.reshape(-1, 1)

            model_pre = LinearRegression()
            model_pre.fit(feature_select_j_ready, feature_select_i_ready)          

            coef_pre_list.append(model_pre.coef_)
            feature_select_pre_pred = model_pre.predict(feature_select_j_ready)
            r2_score_pre = r2_score(feature_select_i_ready, feature_select_pre_pred)
            r2_score_pre_list.append(r2_score_pre)
            
            if r2_score_pre >= relevance_threshold:                
                delete_list_relav.append(feature_select_label[j])
                delete_index_list_relav.append(j)
                print('Feature:', feature_select_label[i], '& Feature:', feature_select_label[j], 'has significant revalence, discard the latter!')

    delete_index_list_relav = sorted(set(delete_index_list_relav))

    for n in range(0, len(delete_index_list_relav)):
        del_no = delete_index_list_relav[n]
        del_no_true = del_no - n 
        del feature_select_label[del_no_true]
        feature_select_label.sort
        del feature_select[del_no_true]
        feature_select.sort

    feature_num_select = feature_num - len(delete_index_list_relav)

    return feature_select_label, feature_select, feature_num_select


# Feature pre-processing module C, construct quadratic cross terms and output all pre-processed features
def normalizer(N_list, N_type):

    # If type is 0 or other input, turn off the feature
    
    if N_type == 1: # max,min normalization
        list_normalized = (N_list - np.min(N_list)) / (np.max(N_list) - np.min(N_list))
        
    elif N_type == 2: # Standard deviation, mean standardization
        N_mean = np.mean(N_list)
        N_var = np.std(N_list)
        list_normalized = (N_list - N_mean) / (N_var)

    elif N_type == 0: # Do not use normalization/standardization data processing feature
        list_normalized = N_list

    else:
        print("unknown nomralization parameter!")
        sys.exit(0)        

    return list_normalized


# Feature pre-processing module C, construct quadratic cross terms and output all pre-processed features
def feature_pretreatment(file, cross_type, normalizer_type, constant_threshold, relevance_threshold): # cross term includes square terms

    start_time = time.time()

    print("Training set file: ", file)
    Y_label, Y, X_label, X, col, row = csv_cleaner(file)

    if normalizer_type == 0:
        print('(normalize_method = 0)Normalization/Standardization skipped!')

    elif normalizer_type == 1:
        print('(normalize_method = 1)Data was processed using normalization!')

    elif normalizer_type == 2:
        print('(normalize_method = 2)Data was processed using standardization!')

     # If constructing cross terms directly, and if data selection is also normalized/standardized, then force to normalize/standardize data first, then construct cross terms
    if cross_type == 1:
        if normalizer_type != 0:
            cross_type = 3
            if normalizer_type == 1:
                print('(cross_type set to 3)Normalization was first applied prior to the construction of cross terms!')            
            elif normalizer_type == 2:
                print('(cross_type set to 3)Standardization was first applied prior to the construction of cross terms!')
        else:
            print('(cross_type = %d)Cross terms were constructed without Normalization/Standardization' %cross_type)

    # If "cross then normalize" or "normalize then cross" options are selected, and if the no cross/normalization option is also selected, then force to use direct construction of cross terms
    elif cross_type != 1:
        if normalizer_type != 0:
            constant_threshold = -1 
            print('(cross_type = %d)Construction of cross terms was first applied prior to Normalization/Standardization!' %cross_type)
            print('(constant_threshold set to %d)Constanst pre-selection was skipped!' %constant_threshold)
        elif normalizer_type == 0:
            cross_type = 1
            print('(cross_type set to 1)Constructing cross terms without Normalization/Standardization!!')           
    feature_num = row - 1
    X_label, X, feature_num = feature_pre_selection(feature_num, X_label, X, constant_threshold, relevance_threshold)

    
    cross = []
    cross_label = []
    Y_norm = []
    X_norm = []
    X_norm_limited = []
    c_num = int(math.factorial(feature_num) / (math.factorial(2) * math.factorial(feature_num - 2)))

    if cross_type == 1:
        for p in range(0, feature_num): 
            for q in range(0,p + 1):
                cross.append(X[p] * X[q])
                cross_label.append(X_label[q] + '_X_' + X_label[p])
        X.extend(cross)
        X_label.extend(cross_label)
        X_norm.extend(X)
        Y_norm.extend(Y)
            
    elif cross_type == 2:
        for p in range(0, feature_num): 
            for q in range(0,p + 1):
                cross.append(X[p] * X[q])
                cross_label.append(X_label[q] + '_X_' + X_label[p])
        X.extend(cross)
        X_label.extend(cross_label)
        Y_norm = normalizer(Y, normalizer_type)
        for i in range(0 , len(X_label)):
            X_norm.append(normalizer(X[i], normalizer_type))

    elif cross_type == 3:
        for i in range(0, feature_num):
            X_norm.append(normalizer(X[i], normalizer_type))
        for p in range(0, feature_num): 
            for q in range(0, p + 1):
                cross.append(X_norm[p] * X_norm[q])
                cross_label.append(X_label[q] + '_X_' + X_label[p])
        Y_norm = normalizer(Y, normalizer_type)
        X_norm.extend(cross)
        X_label.extend(cross_label)

    else:
        print("unknown cross_type parameter!")
        sys.exit(0)

    row_new = len(X_label)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Fearure Pre-treatment Time Elapsed: ", '%.2f'%execution_time, "Seconds", "\n")

    return Y_label, Y_norm, X_label, X_norm, col, row_new


# Feature pre-processing module D, constructs the feature space for multiple linear regression
def feature_engine(feature_num, feature_label, feature, feature_selection_method, feature_selection_num, cross_limit):

    # If feature_selection_method = 0 or other input, do not use this feature engine
    # If feature_selection_method = 1, use all feature combinations
    # If feature_selection_method = 2, use recursive feature combinations (this feature is not supported at the moment)

    c_num = int(math.factorial(feature_num) / (math.factorial(feature_selection_num) * math.factorial(feature_num - feature_selection_num)))

    cross_feature_search = '_X_'
    cross_num = len([i for i, string in enumerate(feature_label) if cross_feature_search in string])
    cross_indices = feature_num - cross_num

    feature_original_selection_num = feature_selection_num - cross_limit
    
    # Split original terms and cross terms
    feature_original_label =  feature_label[:cross_indices]
    feature_original = feature[:cross_indices]
    feature_cross_label = feature_label[cross_indices:]
    feature_cross = feature[cross_indices:]

    # Initialize the list of new features after original terms and feature engineering
    feature_original_extend_label = []
    feature_original_extend_label_Tuple = []
    feature_original_extend = []
    feature_original_extend_Tuple = []
    feature_cross_extend_label = []
    feature_cross_extend_label_Tuple = []
    feature_cross_extend = []
    feature_cross_extend_Tuple = []

    feature_extend_label = []
    feature_extend_label_Tuple = []
    feature_extend = []
    feature_extend_Tuple = []
    
    if feature_selection_num > feature_num:
        print('not enough features! check input parameter!')
        sys.exit(0)

    if cross_limit > feature_selection_num:
        print('unreasonable cross_limit!')
        sys.exit(0)

    if cross_limit == 0:
        if feature_num >= feature_selection_num:
            feature_extend_label_Tuple = list(combinations(feature_original_label, feature_selection_num))
            feature_extend_Tuple = list(combinations(feature_original, feature_selection_num))
            feature_extend_label = ['_&_'.join(s) for s in feature_extend_label_Tuple]
            feature_extend = [list(t) for t in feature_extend_Tuple]
        else:
            print('not enough features! check input parameter!')
            sys.exit(0)

    elif cross_limit != feature_selection_num:
        feature_original_extend_label_Tuple = list(combinations(feature_original_label, feature_original_selection_num))
        feature_original_extend_Tuple = list(combinations(feature_original, feature_original_selection_num))
        feature_cross_extend_label_Tuple = list(combinations(feature_cross_label, cross_limit))
        feature_cross_extend_Tuple = list(combinations(feature_cross, cross_limit))

        feature_extend_label_Tuple = [p + q for p in feature_original_extend_label_Tuple for q in feature_cross_extend_label_Tuple]
        feature_extend_Tuple = [np.concatenate((j, k)) for j in feature_original_extend_Tuple for k in feature_cross_extend_Tuple]           
        
        feature_extend_label = ['_&_'.join(s) for s in feature_extend_label_Tuple]
        feature_extend = [list(t) for t in feature_extend_Tuple]

    elif cross_limit == feature_selection_num:
        feature_cross_extend_label_Tuple = list(combinations(feature_cross_label, cross_limit))
        feature_cross_extend_Tuple = list(combinations(feature_cross, cross_limit))
        
        feature_extend_label = ['_&_'.join(s) for s in feature_cross_extend_label_Tuple]
        feature_extend = [list(t) for t in feature_cross_extend_Tuple]       

    return feature_extend_label, feature_extend
    

# Multiple Linear Regression Module
def linear_regressor(Y_label, Y_data, X_label, X_data, col, row, feature_selection_method, feature_selection_number, cross_limit, required_model_num, test_method):

    start_time = time.time()

    Y = np.array(Y_data)
    X = np.array(X_data)

    X_label_update = []
    X_update = []

    X_label_update, X_update = feature_engine(row, X_label, X, feature_selection_method, feature_selection_number, cross_limit)

    estimator_list = []
    coefficients_list = []
    intercept_list = []
    r_square_list = []

    r_square_list_sorted = []
    r_square_list_top = []

    r2_score_list = []
    mae_list = []

    print('All Combination Number: ', len(X_label_update))
    print('Model will select %d features, including up to %d cross terms!'%(feature_selection_number, cross_limit))
    
    for i in range(0, len(X_label_update)):
        X_ready = []
        X_ready = np.array(X_update[i]).T
        cur_feature_list = X_label_update[i].split('_&_')
        model =  LinearRegression()
        model.fit(X_ready, Y)
        coefficients = model.coef_
        coefficients_list.append(coefficients)        
        intercept = model.intercept_
        intercept_list.append(intercept)

        equation = f'{Y_label} = '
        for j in range(len(cur_feature_list)):
            equation += f'{coefficients[j]:.4f} * {cur_feature_list[j]}'
            if j < len(cur_feature_list) - 1:
                equation += ' + '

        equation += f' + {intercept:.4f}'
        estimator_list.append(equation)

        # Output the model's coefficient of determination, which is the mathematical expression of R^2_Score
        Y_pred = model.predict(X_ready)
        SSR = np.sum((Y_pred - np.mean(Y))**2)  # Regression sum of squares
        SST = np.sum((Y - np.mean(Y))**2)  # Total sum
        r_square = SSR / SST
        r_square_list.append(r_square)
        r2_score_pred = r2_score(Y, Y_pred)
        r2_score_list.append(r2_score_pred)
        mae = mean_absolute_error(Y, Y_pred)
        mae_list.append(mae)

    # Sort the correlation coefficients of all models and output the top N models
    top_model = []
    top_feature = []
    
    r_square_list_sorted = sorted(list(enumerate(r_square_list)), key = lambda x: x[1], reverse=True)
    r_square_list_top = [index for index, _ in r_square_list_sorted[:required_model_num]]
    for top_num in range(0, len(r_square_list_top)):
        print('Top', top_num + 1, 'model:')
        print(estimator_list[r_square_list_top[top_num]])
        print('R2_Score = %.2f' %r_square_list[r_square_list_top[top_num]])
        print('MAE:', '%.2f'%mae_list[r_square_list_top[top_num]] + '\n')

        test_engine(coefficients_list[r_square_list_top[top_num]], X_label_update[r_square_list_top[top_num]].split('_&_'), intercept_list[r_square_list_top[top_num]], X_update[r_square_list_top[top_num]], Y_label, Y_data, col, test_method)

    # Module that only outputs the TOP 1
    max_r_sq = max(r_square_list)
    optimal_index = r_square_list.index(max_r_sq)
    optimal_model = estimator_list[optimal_index]
    optimal_mae = mae_list[optimal_index]

    end_time = time.time()
    execution_time = end_time - start_time
    print("Linear_Regressor Time Elapsed：", '%.2f'%execution_time, "Seconds", "\n")


# Regression Model Testing/Validation Module
def test_engine(feature_index, feature_label, intercept, X_data, Y_label, Y_data, Y_num, test_method):

    Y = np.array(Y_data)
    X = np.array(X_data).T

    loo_estimator_list = []
    coefficients_list = []
    intercept_list = []
    r_square_list = []
    Y_pred_list = []

    if test_method == -1:
        for i in range(0, Y_num):
            Y_update = []
            X_update = []
            Y_update = [element for index, element in enumerate(Y) if index != i]
            X_update = [element for index, element in enumerate(X) if index != i]
            X_update_ready = np.array(X_update)

            model =  LinearRegression()
            model.fit(X_update_ready, Y_update)
            coefficients = model.coef_
            coefficients_list.append(coefficients)        
            intercept = model.intercept_
            intercept_list.append(intercept)

            equation = f'{Y_label} = '
            for j in range(len(feature_label)):
                equation += f'{coefficients[j]:.4f} * {feature_label[j]}'
                if j < len(feature_label) - 1:
                    equation += ' + '

            equation += f' + {intercept:.4f}'
            loo_estimator_list.append(equation)

            # Output the model's coefficient of determination, which is the mathematical expression of R^2_Score
            Y_pred = model.predict(X_update_ready)
            Y_pred_list.append(Y_pred)
            SSR = np.sum((Y_pred - np.mean(Y_update))**2)  # Regression sum of squares
            SST = np.sum((Y_update - np.mean(Y_update))**2)  # Total sum
            r_square = SSR / SST
            r_square_list.append(r_square)
            print('Each LOO R2_score:', '%.2f'%r_square, 'Leave No.%.i'%(i+1))

        r2_score_pred = r2_score(Y_update, Y_pred)
        print('LOO R2_score:', '%.2f'%r2_score_pred, '\n')
    else:
        print("k-fold method currently not supported!")
        sys.exit(0)


dep_var_label, dep_var, ind_var_label, ind_var, col, row = feature_pretreatment(train_input_file, cross_type, normalize_method, constant_threshold, relevance_threshold)

linear_regressor(dep_var_label, dep_var, ind_var_label, ind_var, col, row, feature_selection_method, feature_selection_number, cross_limit, required_model_num, test_method)

end_time = time.time()
execution_time = end_time - start_time
print("\nTotal Time Elapsed：", '%.2f'%execution_time, "Seconds")