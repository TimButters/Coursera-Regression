#!/usr/bin/python

import numpy as np
import pandas


def get_numpy_data(dataset, features, output):
    feature_matrix = np.ones((dataset.shape[0], len(features)+1))
    for f, i in zip(features, range(len(features))):
        feature_matrix[:, i+1] = dataset[f]
    output_array = np.array(dataset[output])
    return feature_matrix, output_array


def predict_outcome(feature_matrix, weights):
    return np.dot(feature_matrix, weights.reshape(-1, 1))


def feature_derivative_ridge(errors, feature, weight, l2_penalty,
                             feature_is_constant):
    if feature_is_constant:
        return 2*np.dot(feature.reshape(-1, 1).T, errors.reshape(-1, 1))
    else:
        return (2*np.dot(feature.reshape(-1, 1).T, errors.reshape(-1, 1)) +
                2*l2_penalty*weight)


def ridge_regression_gradient_descent(feature_matrix, output,
                                      initial_weights, step_size,
                                      l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)
    iterations = 0
    while iterations < max_iterations:
        predictions = predict_outcome(feature_matrix, weights)
        errors = predictions - output.reshape(-1, 1)
        for i in range(len(weights)):  # loop over each weight
            if i == 0:
                is_const = True
            else:
                is_const = False
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i],
                                                  weights[i], l2_penalty,
                                                  is_const)
            weights[i] = weights[i] - step_size*derivative
        iterations += 1
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

    simple_features = ['sqft_living']
    my_output = 'price'
    (simple_feature_matrix, output) = get_numpy_data(train, simple_features,
                                                     my_output)
    (simple_test_feature_matrix, test_output) = get_numpy_data(test,
                                                               simple_features,
                                                               my_output)

    simple_weights_0_penalty = (
        ridge_regression_gradient_descent(simple_feature_matrix, output,
                                          np.zeros((2, 1)),
                                          1e-12, 0.0, 1000))

    simple_weights_high_penalty = (
        ridge_regression_gradient_descent(simple_feature_matrix, output,
                                          np.zeros((2, 1)),
                                          1e-12, 1e11, 1000))

    print("Simple 0", simple_weights_0_penalty)
    print("Simple High", simple_weights_high_penalty)

    pred1 = predict_outcome(simple_test_feature_matrix, np.zeros((2, 1)))
    pred2 = predict_outcome(simple_test_feature_matrix,
                            simple_weights_0_penalty)
    pred3 = predict_outcome(simple_test_feature_matrix,
                            simple_weights_high_penalty)
    print("RSS Init", np.sum((pred1 - test_output.reshape(-1, 1))**2))
    print("RSS No", np.sum((pred2 - test_output.reshape(-1, 1))**2))
    print("RSS High", np.sum((pred3 - test_output.reshape(-1, 1))**2))

    import matplotlib.pyplot as plt
    plt.plot(simple_feature_matrix, output, 'k.', label='data')
    plt.plot(simple_feature_matrix,
             predict_outcome(simple_feature_matrix,
                             simple_weights_0_penalty), 'b', label='zero')
    plt.plot(simple_feature_matrix,
             predict_outcome(simple_feature_matrix,
                             simple_weights_high_penalty), 'r', label='high')
    plt.legend()
    plt.show()

    model_features = ['sqft_living', 'sqft_living15']
    my_output = 'price'
    (feature_matrix, output) = get_numpy_data(train, model_features,
                                              my_output)
    (test_feature_matrix, test_output) = get_numpy_data(test,
                                                        model_features,
                                                        my_output)

    multiple_weights_0_penalty = (
        ridge_regression_gradient_descent(feature_matrix, output,
                                          np.zeros((3, 1)),
                                          1e-12, 0.0, 1000))
    multiple_weights_high_penalty = (
        ridge_regression_gradient_descent(feature_matrix, output,
                                          np.zeros((3, 1)),
                                          1e-12, 1e11, 1000))
    print("Multi 0", multiple_weights_0_penalty)
    print("Multi High", multiple_weights_high_penalty)

    pred1 = predict_outcome(test_feature_matrix, np.zeros((3, 1)))
    pred2 = predict_outcome(test_feature_matrix,
                            multiple_weights_0_penalty)
    pred3 = predict_outcome(test_feature_matrix,
                            multiple_weights_high_penalty)

    print("RSS Init", np.sum((pred1 - test_output.reshape(-1, 1))**2))
    print("RSS No", np.sum((pred2 - test_output.reshape(-1, 1))**2))
    print("RSS High", np.sum((pred3 - test_output.reshape(-1, 1))**2))

    print(pred2[0])
    print(pred3[0])
