#!/usr/bin/python

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas


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

    # New Feature Means
    print("Quiz Question: Mean for new features in test data")
    print("bedrooms_squared = ", np.mean(test_data['bedrooms_squared']))
    print("bed_bath_rooms = ", np.mean(test_data['bed_bath_rooms']))
    print("log_sqft_living = ", np.mean(test_data['log_sqft_living']))
    print("lat_plus_long = ", np.mean(test_data['lat_plus_long']))
    print("")

    # Regression Coefficients
    regressor1 = LinearRegression(fit_intercept=True)
    regressor2 = LinearRegression(fit_intercept=True)
    regressor3 = LinearRegression(fit_intercept=True)

    rss = {}

    y = np.array(training_data['price'])

    X_m1 = np.array([np.array(training_data['sqft_living']),
                     np.array(training_data['bedrooms']),
                     np.array(training_data['bathrooms']),
                     np.array(training_data['lat']),
                     np.array(training_data['long'])]).transpose()
    model1 = regressor1.fit(X_m1, y)
    print("Quiz Question: What is the sign for the coeff bathrooms in model 1")
    print("Value = ", model1.coef_[2])
    print("")
    rss['model1'] = np.mean((model1.predict(X_m1) - y)**2)

    X_m2 = np.array([np.array(training_data['sqft_living']),
                     np.array(training_data['bedrooms']),
                     np.array(training_data['bathrooms']),
                     np.array(training_data['lat']),
                     np.array(training_data['long']),
                     np.array(training_data['bed_bath_rooms'])]).transpose()
    model2 = regressor2.fit(X_m2, y)
    print("Quiz Question: What is the sign for the coeff bathrooms in model 2")
    print("Value = ", model2.coef_[2])
    print("")
    rss['model2'] = np.mean((model2.predict(X_m2) - y)**2)

    X_m3 = np.array([np.array(training_data['sqft_living']),
                     np.array(training_data['bedrooms']),
                     np.array(training_data['bathrooms']),
                     np.array(training_data['lat']),
                     np.array(training_data['long']),
                     np.array(training_data['bed_bath_rooms']),
                     np.array(training_data['bedrooms_squared']),
                     np.array(training_data['log_sqft_living']),
                     np.array(training_data['lat_plus_long'])]).transpose()
    model3 = regressor3.fit(X_m3, y)
    rss['model3'] = np.mean((model3.predict(X_m3) - y)**2)

    print("Quiz Question: Which model has the smallest RSS for training")
    print(min(rss, key=rss.get))
    print("")

    y_test = np.array(test_data['price'])

    X_t1 = np.array([np.array(test_data['sqft_living']),
                     np.array(test_data['bedrooms']),
                     np.array(test_data['bathrooms']),
                     np.array(test_data['lat']),
                     np.array(test_data['long'])]).transpose()
    X_t2 = np.array([np.array(test_data['sqft_living']),
                     np.array(test_data['bedrooms']),
                     np.array(test_data['bathrooms']),
                     np.array(test_data['lat']),
                     np.array(test_data['long']),
                     np.array(test_data['bed_bath_rooms'])]).transpose()
    X_t3 = np.array([np.array(test_data['sqft_living']),
                     np.array(test_data['bedrooms']),
                     np.array(test_data['bathrooms']),
                     np.array(test_data['lat']),
                     np.array(test_data['long']),
                     np.array(test_data['bed_bath_rooms']),
                     np.array(test_data['bedrooms_squared']),
                     np.array(test_data['log_sqft_living']),
                     np.array(test_data['lat_plus_long'])]).transpose()
    rss_test = {}
    rss_test['model1'] = np.mean((model1.predict(X_t1) - y_test)**2)
    rss_test['model2'] = np.mean((model2.predict(X_t2) - y_test)**2)
    rss_test['model3'] = np.mean((model3.predict(X_t3) - y_test)**2)

    print("Quiz Question: Which model has the smallest RSS for test")
    print(min(rss_test, key=rss_test.get))
