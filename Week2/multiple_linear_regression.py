#!/usr/bin/python

import numpy as np
import pandas


if __name__ == "__main__":
    training_data = pandas.read_csv("Data/kc_house_train_data.csv")
    test_data = pandas.read_csv("Data/kc_house_test_data.csv")

    training_data['bedrooms_squared'] = training_data['bedrooms']**2
    test_data['bedrooms_squared'] = test_data['bedrooms']**2

    training_data['bed_bath_rooms'] = (training_data['bedrooms'] *
                                       training_data['bathrooms'])
    test_data['bed_bath_rooms'] = (test_data['bedrooms'] *
                                   test_data['bathrooms'])

    training_data['log_sqft_living'] = np.log(training_data['sqft_living'])
    test_data['log_sqft_living'] = np.log(test_data['sqft_living'])

    training_data['lat_plus_long'] = (training_data['lat'] +
                                      training_data['long'])
    test_data['lat_plus_long'] = (test_data['lat'] +
                                  test_data['long'])
