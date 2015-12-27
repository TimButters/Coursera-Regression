#!/usr/bin/python

from sklearn import linear_model
# import matplotlib.pyplot as plt
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


def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    k = 10  # 10-fold cross-validation

    rss = 0
    for i in range(k):
        start = int((n*i)/k)
        end = int((n*(i+1))/k-1)
        # print(i, (start, end))
        validation = data[start:end+1]
        validation_output = output[start:end+1]
        training = data[0:start].append(data[end+1:n])
        train_output = output[0:start].append(output[end+1:n])

        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(training, train_output)

        predictions = model.predict(validation)
        rss += np.sum((predictions - validation_output)**2)

    return rss/k


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

    poly15_data = polynomial_dframe(sales['sqft_living'], 15)

    l2_small_penalty = 1.5e-5

    model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model.fit(poly15_data, sales['price'])

    print("Quiz Question: What's the value of the coeff for feature power_1")
    print(model.coef_[0])

    set_1 = pandas.read_csv('Data/wk3_kc_house_set_1_data.csv',
                            dtype=dtype_dict)
    set_1 = set_1.sort(['sqft_living', 'price'])
    poly_set_1 = polynomial_dframe(set_1['sqft_living'], 15)

    set_2 = pandas.read_csv('Data/wk3_kc_house_set_2_data.csv',
                            dtype=dtype_dict)
    set_2 = set_2.sort(['sqft_living', 'price'])
    poly_set_2 = polynomial_dframe(set_2['sqft_living'], 15)

    set_3 = pandas.read_csv('Data/wk3_kc_house_set_3_data.csv',
                            dtype=dtype_dict)
    set_3 = set_3.sort(['sqft_living', 'price'])
    poly_set_3 = polynomial_dframe(set_3['sqft_living'], 15)

    set_4 = pandas.read_csv('Data/wk3_kc_house_set_4_data.csv',
                            dtype=dtype_dict)
    set_4 = set_4.sort(['sqft_living', 'price'])
    poly_set_4 = polynomial_dframe(set_4['sqft_living'], 15)

    l2_small_penalty = 1e-9
    model1 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model1.fit(poly_set_1, set_1['price'])
    print()
    print("Small L2 Penalty")
    print("Model 1:", model1.coef_[0])
    print()

    model2 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model2.fit(poly_set_2, set_2['price'])
    print("Model 2:", model2.coef_[0])
    print()

    model3 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model3.fit(poly_set_3, set_3['price'])
    print("Model 3:", model3.coef_[0])
    print()

    model4 = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model4.fit(poly_set_4, set_4['price'])
    print("Model 4:", model4.coef_[0])
    print()

    l2_large_penalty = 1.23e2
    model_large_1 = linear_model.Ridge(alpha=l2_large_penalty,
                                       normalize=True)
    model_large_1.fit(poly_set_1, set_1['price'])
    print()
    print("Large L2 Penalty")
    print("Model 1:", model_large_1.coef_[0])
    print()

    model_large_2 = linear_model.Ridge(alpha=l2_large_penalty,
                                       normalize=True)
    model_large_2.fit(poly_set_2, set_2['price'])
    print("Model 2:", model_large_2.coef_[0])
    print()

    model_large_3 = linear_model.Ridge(alpha=l2_large_penalty,
                                       normalize=True)
    model_large_3.fit(poly_set_3, set_3['price'])
    print("Model 3:", model_large_3.coef_[0])
    print()

    model_large_4 = linear_model.Ridge(alpha=l2_large_penalty,
                                       normalize=True)
    model_large_4.fit(poly_set_4, set_4['price'])
    print("Model 4:", model_large_4.coef_[0])
    print()

    # Cross Validation
    train_valid_shuffled = (
        pandas.read_csv('Data/wk3_kc_house_train_valid_shuffled.csv',
                        dtype=dtype_dict))
    test = pandas.read_csv('Data/wk3_kc_house_test_data.csv',
                           dtype=dtype_dict)

    dataf = polynomial_dframe(train_valid_shuffled['sqft_living'], 15)
    out = train_valid_shuffled['price']

    ave_err = {}
    for l2p in np.logspace(3, 9, num=13):
        ave_err[l2p] = k_fold_cross_validation(10, l2p, dataf, out)
    best_l2p = min(ave_err, key=ave_err.get)
    print(best_l2p)

    best_model = linear_model.Ridge(alpha=best_l2p, normalize=True)
    best_model.fit(dataf, out)

    poly_test = polynomial_dframe(test['sqft_living'], 15)
    test_out = test['price']

    rss = np.sum((best_model.predict(poly_test) - test_out)**2)
    print(rss)
