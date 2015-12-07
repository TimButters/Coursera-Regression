#!/usr/bin/python

import numpy as np
import pandas


def simple_linear_regression(input_feature, output):
    X = np.array([np.ones(len(input_feature)), input_feature]).transpose()
    XTX = np.dot(X.transpose(), X)
    intercept, slope = np.dot(np.dot(np.linalg.inv(XTX), X.transpose()), output)
    return intercept, slope


def get_regression_predictions(input_feature, intercept, slope):
    return intercept + input_feature*slope


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    predictions = get_regression_predictions(input_feature, intercept, slope)
    return np.sum((output - predictions)**2)


def inverse_regression_predictions(output, intercept, slope):
    return (output - intercept)/slope


if __name__ == "__main__":
    training_data = pandas.read_csv("Data/kc_house_train_data.csv")
    test_data = pandas.read_csv("Data/kc_house_test_data.csv")

    intercept, slope = simple_linear_regression(training_data['sqft_living'],
                                                training_data['price'])

    print("Quiz Question: Predicted price for house with 2650 sqft")
    print(get_regression_predictions(2650, intercept, slope))
    print("")

    print("Quiz Question: RSS for training data")
    print(get_residual_sum_of_squares(training_data['sqft_living'],
                                      training_data['price'],
                                      intercept, slope))
    print("")

    print("Quiz Question: Estimated sqft of house costing $800,000")
    print(inverse_regression_predictions(800000, intercept, slope))
    print("")

    print("Quiz Question: Which model (sqft or bedrooms) has the lowest RSS")
    bedroom_intercept, bedroom_slope = (
        simple_linear_regression(training_data['bedrooms'],
                                 training_data['price']))
    sqft_rss = get_residual_sum_of_squares(test_data['sqft_living'],
                                           test_data['price'], intercept, slope)
    bedroom_rss = get_residual_sum_of_squares(test_data['bedrooms'],
                                              test_data['price'],
                                              intercept, slope)
    best = "sqft" if sqft_rss < bedroom_rss else "bedrooms"
    print(best)
