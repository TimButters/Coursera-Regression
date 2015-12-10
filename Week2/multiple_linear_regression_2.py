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
    return np.dot(feature_matrix, weights)


def feature_derivative(errors, feature):
    return 2*np.dot(feature, errors)


def regression_gradient_descent(feature_matrix, output, initial_weights,
                                step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = predict_outcome(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += derivative**2
            weights[i] = weights[i] - step_size*derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
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

    training_data = pandas.read_csv("Data/kc_house_train_data.csv",
                                    dtype=dtype_dict)
    test_data = pandas.read_csv("Data/kc_house_test_data.csv", dtype=dtype_dict)

    # Simple Model
    simple_features = ['sqft_living']
    my_output = 'price'
    simple_feature_matrix, output = get_numpy_data(training_data,
                                                   simple_features, my_output)
    initial_weights = np.array([-47000.0, 1.0])
    step_size = 7e-12
    tolerance = 2.5e7

    simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                                 initial_weights, step_size,
                                                 tolerance)
    print("Quiz Question: Value of weight for sqft_living")
    print(simple_weights[1], "\n")

    test_simple_feature_matrix, test_output = get_numpy_data(test_data,
                                                             simple_features,
                                                             my_output)
    test_predictions = predict_outcome(test_simple_feature_matrix,
                                       simple_weights)
    simple_prediction = test_predictions[0]
    print("Quiz Question: What is the predicted price of the 1st House?")
    print(simple_prediction, "\n")

    rss_simple = np.sum((test_predictions - test_output)**2)

    # More Complex Model
    model_features = ['sqft_living', 'sqft_living15']
    my_output = 'price'
    initial_weights = [-100000.0, 1.0, 1.0]
    step_size = 4e-12
    tolerance = 1e9

    feature_matrix, output = get_numpy_data(training_data, model_features,
                                            my_output)
    weights = regression_gradient_descent(feature_matrix, output,
                                          initial_weights, step_size,
                                          tolerance)
    test_feature_matrix, test_output = get_numpy_data(test_data, model_features,
                                                      my_output)
    predictions = predict_outcome(test_feature_matrix, weights)

    print("Quiz Question: Price of 1st house in test data")
    print(predictions[0], "\n")

    best = 'Model 1' if (np.abs(simple_prediction - test_output[0]) <
                         np.abs(predictions[0] - test_output[0])) else 'Model 2'
    print("Quiz Question: Which model was closer to the real price?")
    print(best, "")

    rss = np.sum((predictions - test_output)**2)
    best = 'Model 1' if rss_simple < rss else 'Model 2'

    print("Quiz Question: Which model has the lowest RSS?")
    print(best, "")
