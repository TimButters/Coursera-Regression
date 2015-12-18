#!/usr/bin/python

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas


def polynomial_dframe(feature, degree):
    new_df = pandas.DataFrame()
    new_df['power_1'] = np.array(feature)
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            new_df[name] = np.array(feature)**power
    return new_df


if __name__ == "__main__":
    dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
                  'sqft_living15': float, 'grade': int, 'yr_renovated': int,
                  'price': float, 'bedrooms': float, 'zipcode': str,
                  'long': float, 'sqft_lot15': float, 'sqft_living': float,
                  'floors': str, 'condition': int, 'lat': float, 'date': str,
                  'sqft_basement': int, 'yr_built': int, 'id': str,
                  'sqft_lot': int, 'view': int}

    sales = pandas.read_csv("Data/kc_house_data.csv", dtype=dtype_dict)
    sales = sales.sort(['sqft_living', 'price'])

    poly1_data = polynomial_dframe(sales['sqft_living'], 1)
    poly1_data['price'] = sales['price']

    # Degree 1
    regressor1 = LinearRegression(fit_intercept=True)
    # X = np.array([np.array(poly1_data['sqft_living'])]).transpose()
    X1 = np.array(poly1_data['power_1']).reshape(-1, 1)
    y = np.array(sales['price']).reshape(-1, 1)
    model1 = regressor1.fit(X1, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly1_data['power_1'], poly1_data['price'], '.')
    ax.plot(poly1_data['power_1'], model1.predict(X1), '-')

    # Degree 2
    poly2_data = polynomial_dframe(sales['sqft_living'], 2)
    poly2_data['price'] = sales['price']

    regressor2 = LinearRegression(fit_intercept=True)
    X2 = np.array([np.array(poly2_data['power_1']),
                   np.array(poly2_data['power_2'])]).transpose()
    y = np.array(sales['price']).reshape(-1, 1)
    model2 = regressor2.fit(X2, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly2_data['power_1'], poly2_data['price'], '.')
    ax.plot(poly2_data['power_1'], model2.predict(X2), '-')

    # Degree 3
    poly3_data = polynomial_dframe(sales['sqft_living'], 3)
    poly3_data['price'] = sales['price']

    regressor3 = LinearRegression(fit_intercept=True)
    X3 = np.array(poly3_data)[:, 0:-1]
    y = np.array(sales['price']).reshape(-1, 1)
    model3 = regressor3.fit(X3, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly3_data['power_1'], poly3_data['price'], '.')
    ax.plot(poly3_data['power_1'], model3.predict(X3), '-')

    # Degree 15
    poly15_data = polynomial_dframe(sales['sqft_living'], 15)
    poly15_data['price'] = sales['price']

    regressor15 = LinearRegression(fit_intercept=True)
    X15 = np.array(poly15_data)[:, 0:-1]
    y = np.array(sales['price']).reshape(-1, 1)
    model15 = regressor15.fit(X15, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly15_data['power_1'], poly15_data['price'], '.')
    ax.plot(poly15_data['power_1'], model15.predict(X15), '-')

    # New Datasets
    set_1 = pandas.read_csv("Data/wk3_kc_house_set_1_data.csv",
                            dtype=dtype_dict)
    set_1 = set_1.sort(['sqft_living', 'price'])
    set_2 = pandas.read_csv("Data/wk3_kc_house_set_2_data.csv",
                            dtype=dtype_dict)
    set_2 = set_2.sort(['sqft_living', 'price'])
    set_3 = pandas.read_csv("Data/wk3_kc_house_set_3_data.csv",
                            dtype=dtype_dict)
    set_3 = set_3.sort(['sqft_living', 'price'])
    set_4 = pandas.read_csv("Data/wk3_kc_house_set_4_data.csv",
                            dtype=dtype_dict)
    set_4 = set_4.sort(['sqft_living', 'price'])

    set_1_poly_data = polynomial_dframe(set_1['sqft_living'], 15)
    set_1_poly_data['price'] = set_1['price']
    set_2_poly_data = polynomial_dframe(set_2['sqft_living'], 15)
    set_2_poly_data['price'] = set_2['price']
    set_3_poly_data = polynomial_dframe(set_3['sqft_living'], 15)
    set_3_poly_data['price'] = set_3['price']
    set_4_poly_data = polynomial_dframe(set_4['sqft_living'], 15)
    set_4_poly_data['price'] = set_4['price']

    regressor_1 = LinearRegression(fit_intercept=True)
    X_1 = np.array(set_1_poly_data)[:, 0:-1]
    y_1 = np.array(set_1['price']).reshape(-1, 1)
    model_1 = regressor_1.fit(X_1, y_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(set_1_poly_data['power_1'], set_1_poly_data['price'], '.')
    ax.plot(set_1_poly_data['power_1'], model_1.predict(X_1), '-')

    regressor_2 = LinearRegression(fit_intercept=True)
    X_2 = np.array(set_2_poly_data)[:, 0:-1]
    y_2 = np.array(set_2['price']).reshape(-1, 1)
    model_2 = regressor_2.fit(X_2, y_2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(set_2_poly_data['power_1'], set_2_poly_data['price'], '.')
    ax.plot(set_2_poly_data['power_1'], model_2.predict(X_2), '-')

    regressor_3 = LinearRegression(fit_intercept=True)
    X_3 = np.array(set_3_poly_data)[:, 0:-1]
    y_3 = np.array(set_3['price']).reshape(-1, 1)
    model_3 = regressor_3.fit(X_3, y_3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(set_3_poly_data['power_1'], set_3_poly_data['price'], '.')
    ax.plot(set_3_poly_data['power_1'], model_3.predict(X_3), '-')

    regressor_4 = LinearRegression(fit_intercept=True)
    X_4 = np.array(set_4_poly_data)[:, 0:-1]
    y_4 = np.array(set_4['price']).reshape(-1, 1)
    model_4 = regressor_4.fit(X_4, y_4)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(set_4_poly_data['power_1'], set_4_poly_data['price'], '.')
    ax.plot(set_4_poly_data['power_1'], model_4.predict(X_4), '-')

    print("Quiz Question: Is the sign for ^15 the same in all models?")
    print("Model 1: ", model_1.coef_[0][14])
    print("Model 2: ", model_2.coef_[0][14])
    print("Model 3: ", model_3.coef_[0][14])
    print("Model 4: ", model_4.coef_[0][14])

    train = pandas.read_csv("Data/wk3_kc_house_train_data.csv",
                            dtype=dtype_dict)
    train = train.sort(['sqft_living', 'price'])
    test = pandas.read_csv("Data/wk3_kc_house_test_data.csv",
                           dtype=dtype_dict)
    test = test.sort(['sqft_living', 'price'])
    valid = pandas.read_csv("Data/wk3_kc_house_valid_data.csv",
                            dtype=dtype_dict)
    valid = valid.sort(['sqft_living', 'price'])

    rss = {}
    rss_test = {}

    for i in range(1, 15+1):
        poly_data = polynomial_dframe(train['sqft_living'], i)

        regressor = LinearRegression(fit_intercept=True)
        X = np.array(poly_data)
        y = np.array(train['price'])
        model = regressor.fit(X, y)

        validation_x = np.array(polynomial_dframe(valid['sqft_living'], i))
        validation_y = np.array(valid['price'])
        rss[str(i)] = np.sum((model.predict(validation_x) - validation_y)**2)

        test_x = np.array(polynomial_dframe(test['sqft_living'], i))
        test_y = np.array(test['price'])
        rss_test[str(i)] = np.sum((model.predict(test_x) - test_y)**2)

    print("Quiz Question: Which degree has the lowerst RSS?")
    print("Degree", min(rss, key=rss.get), "has the loweest RSS")

    print("Quiz Question: What is the RSS on the test data for this degree?")
    print(rss_test[str(min(rss, key=rss.get))])

    # plt.show()
