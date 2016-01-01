#!/usr/bin/python

import numpy as np
import pandas


def get_numpy_data(dataset, features, output):
    feature_matrix = np.ones((dataset.shape[0], len(features)+1))
    for f, i in zip(features, range(len(features))):
        feature_matrix[:, i+1] = dataset[f]
    output_array = np.array(dataset[output]).reshape(-1, 1)
    return feature_matrix, output_array


def predict_output(feature_matrix, weights):
    return np.dot(feature_matrix, weights.reshape(-1, 1))


def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms)


def lasso_coordinate_descent_step(i, feature_matrix, output,
                                  weights, l1_penalty):
    # compute prediction
    # prediction = predict_output(feature_matrix, weights).reshape(1, -1)
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction +
    # weight[i]*[feature_i]) ]
    v = (weights[i]*feature_matrix[:, i]).reshape(-1, 1)
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + v))

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.0:
        new_weight_i = ro_i + l1_penalty/2.0
    elif ro_i > l1_penalty/2.0:
        new_weight_i = ro_i - l1_penalty/2.0
    else:
        new_weight_i = 0.0

    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights,
                                      l1_penalty, tolerance):
    # make a copy of initial weights
    weights = initial_weights.copy()
    # converged condition variable
    converged = False
    while not converged:
        max_change = 0
        for i in range(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix,
                                                       output, weights,
                                                       l1_penalty)
            change_i = np.abs(old_weights_i - weights[i])
            if change_i > max_change:
                max_change = change_i
        if max_change < tolerance:
            converged = True
    return weights


if __name__ == "__main__":
    dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
                  'sqft_living15': float, 'grade': int, 'yr_renovated': int,
                  'price': float, 'bedrooms': float, 'zipcode': str,
                  'long': float, 'sqft_lot15': float, 'sqft_living': float,
                  'floors': str, 'condition': int, 'lat': float, 'date': str,
                  'sqft_basement': int, 'yr_built': int, 'id': str,
                  'sqft_lot': int, 'view': int}

    sales = pandas.read_csv("Data/kc_house_data.csv", dtype=dtype_dict)
    test = pandas.read_csv("Data/kc_house_test_data.csv", dtype=dtype_dict)
    train = pandas.read_csv("Data/kc_house_train_data.csv", dtype=dtype_dict)

    feature_list = ['sqft_living', 'bedrooms']
    feature_matrix, output = get_numpy_data(sales, feature_list, 'price')

    normd_feat, norms = normalize_features(feature_matrix)
    weights = np.array([1, 4, 1]).reshape(-1, 1)
    prediction = predict_output(normd_feat, weights)

    # ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
    # ro = np.dot(normd_feat.T, (output - prediction +
    #                            np.dot(normd_feat, weights)))
    ro = np.zeros(weights.shape)
    for i in range(len(weights)):
        v = (weights[i]*normd_feat[:, i]).reshape(-1, 1)
        ro[i] = np.dot(normd_feat[:, i], (output - prediction + v))
    diff = abs((ro[1]*2) - (ro[2]*2))
    print('lamda = (%e, %e)' % ((ro[2]-diff/2+1)*2, (ro[2]+diff/2-1)*2))

    print(ro)
    print("Quiz Question: What value of l2_penalty would set w[3] to 0?")
    print("Quiz Question: What value of l2_penalty would set w[2] & w[3] to 0?")

    w = lasso_cyclical_coordinate_descent(normd_feat, output,
                                          np.array([0, 0, 0]), 1e7, 1.0)
    prediction = predict_output(normd_feat, w)
    rss = np.sum((prediction - output)**2)
    print("Quiz Question: What is RSS?")
    print(rss)
    print("Quiz Question: Zero weight features?")
    print(feature_list)
    print(w)

    # More Complex
    feature_list = np.array(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                             'floors', 'waterfront', 'view', 'condition',
                             'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                             'yr_renovated'])
    f_list = np.array(['C', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                       'floors', 'waterfront', 'view', 'condition',
                       'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                       'yr_renovated'])
    feature_matrix, output = get_numpy_data(train, feature_list, 'price')
    normd_feat, norms = normalize_features(feature_matrix)

    initial_weights = np.zeros((normd_feat.shape[1], 1))
    weights1e7 = lasso_cyclical_coordinate_descent(normd_feat, output,
                                                   initial_weights, 1e7, 1.0)
    print("Quiz Question: 1e7 Non-zero weights?")
    print(f_list[np.nonzero(weights1e7)[0]])

    weights1e8 = lasso_cyclical_coordinate_descent(normd_feat, output,
                                                   initial_weights, 1e8, 1.0)
    print("Quiz Question: 1e8 Non-zero weights?")
    print(f_list[np.nonzero(weights1e8)[0]])

    weights1e4 = lasso_cyclical_coordinate_descent(normd_feat, output,
                                                   initial_weights, 1e4, 1.0)
    print("Quiz Question: 1e4 Non-zero weights?")
    print(f_list[np.nonzero(weights1e4)[0]])

    normalized_weights1e7 = weights1e7.reshape(1, -1) / norms
    normalized_weights1e8 = weights1e8.reshape(1, -1) / norms
    normalized_weights1e4 = weights1e4.reshape(1, -1) / norms

    print("Check: Should be 161.31745624837794")
    print(normalized_weights1e7[0, 3])

    # Test Set
    test_mat, test_out = get_numpy_data(test, feature_list, 'price')
    rss = {}
    rss['1e7'] = np.sum((predict_output(test_mat, normalized_weights1e7.T) -
                         test_out)**2)
    rss['1e8'] = np.sum((predict_output(test_mat, normalized_weights1e8.T) -
                         test_out)**2)
    rss['1e4'] = np.sum((predict_output(test_mat, normalized_weights1e4.T) -
                         test_out)**2)
    print("Best RSS")
    print(min(rss, key=rss.get))
