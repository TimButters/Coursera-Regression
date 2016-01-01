#!/usr/bin/python

from sklearn import linear_model
import numpy as np
import pandas


if __name__ == "__main__":
    dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int,
                  'sqft_living15': float, 'grade': int, 'yr_renovated': int,
                  'price': float, 'bedrooms': float, 'zipcode': str,
                  'long': float, 'sqft_lot15': float, 'sqft_living': float,
                  'floors': float, 'condition': int, 'lat': float, 'date': str,
                  'sqft_basement': int, 'yr_built': int, 'id': str,
                  'sqft_lot': int, 'view': int}

    sales = pandas.read_csv("Data/kc_house_data.csv", dtype=dtype_dict)

    sales['sqft_living_sqrt'] = sales['sqft_living'].apply(np.sqrt)
    sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(np.sqrt)
    sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
    sales['floors_square'] = sales['floors']*sales['floors']

    all_features = np.array(['bedrooms', 'bedrooms_square', 'bathrooms',
                             'sqft_living', 'sqft_living_sqrt', 'sqft_lot',
                             'sqft_lot_sqrt', 'floors', 'floors_square',
                             'waterfront', 'view', 'condition', 'grade',
                             'sqft_above', 'sqft_basement', 'yr_built',
                             'yr_renovated'])

    model_all = linear_model.Lasso(alpha=5e2, normalize=True)
    model_all.fit(sales[all_features], sales['price'])

    print("Quiz Question: Which features have been chosen by LASSO?")
    print(all_features[np.nonzero(model_all.coef_)])

    testing = pandas.read_csv('Data/wk3_kc_house_test_data.csv',
                              dtype=dtype_dict)
    training = pandas.read_csv('Data/wk3_kc_house_train_data.csv',
                               dtype=dtype_dict)
    validation = pandas.read_csv('Data/wk3_kc_house_valid_data.csv',
                                 dtype=dtype_dict)

    testing['sqft_living_sqrt'] = testing['sqft_living'].apply(np.sqrt)
    testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(np.sqrt)
    testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
    testing['floors_square'] = testing['floors']*testing['floors']

    training['sqft_living_sqrt'] = training['sqft_living'].apply(np.sqrt)
    training['sqft_lot_sqrt'] = training['sqft_lot'].apply(np.sqrt)
    training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
    training['floors_square'] = training['floors']*training['floors']

    validation['sqft_living_sqrt'] = validation['sqft_living'].apply(np.sqrt)
    validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(np.sqrt)
    validation['bedrooms_square'] = (validation['bedrooms'] *
                                     validation['bedrooms'])
    validation['floors_square'] = validation['floors']*validation['floors']

    rss = {}
    for l1_penalty in np.logspace(1, 7, num=13):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(training[all_features], training['price'])

        rss[l1_penalty] = np.sum((model.predict(validation[all_features]) -
                                  validation['price'])**2)
    print("Quiz Question: Which was the best value for the l1_penalty?")
    print(min(rss, key=rss.get))

    model = linear_model.Lasso(alpha=min(rss, key=rss.get), normalize=True)
    model.fit(training[all_features], training['price'])
    rss_test = np.sum((model.predict(testing[all_features]) -
                       testing['price'])**2)

    print("Quiz Question: Using the best L1 penalty, how many nonzero weights?")
    print(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))

    max_nnz = 7
    l1_penalty_min = 0
    l1_penalty_max = 0
    d = {}
    flag = True
    for l1_penalty in np.logspace(1, 4, num=20):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(training[all_features], training['price'])
        nnz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
        d[l1_penalty] = nnz
        if l1_penalty > l1_penalty_min and nnz > max_nnz:
            l1_penalty_min = l1_penalty
        if l1_penalty > l1_penalty_max and nnz < max_nnz and flag:
            l1_penalty_max = l1_penalty
            flag = False
    print("Quiz Question: What values did you find for l1_penalty_min and "
          + "l1_penalty_max?")
    print("l1_penalty_min", l1_penalty_min)
    print("l1_penalty_max", l1_penalty_max)

    rss_7 = {}
    for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, 20):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(training[all_features], training['price'])

        nnz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
        if nnz == max_nnz:
            rss_7[l1_penalty] = np.sum((model.predict(validation[all_features])
                                        - validation['price'])**2)
    best_l1p = min(rss_7, key=rss_7.get)
    model = linear_model.Lasso(alpha=best_l1p, normalize=True)
    model.fit(training[all_features], training['price'])

    print("Quiz Question: What value of l1_penalty has the lowest RSS?")
    print(best_l1p)

    print("Quiz Question: What features in this model have non-zero coeffs?")
    print(all_features[np.nonzero(model.coef_)])
