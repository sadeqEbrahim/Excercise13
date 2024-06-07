from flask import Flask, render_template
import pandas as pd
import xgboost as xgb
#import time
#import os

app = Flask(__name__)

@app.route('/')
def index():
    # Load labeled data
    transactions_train = pd.read_csv('dataset/transactions_train.csv')
    train_target = pd.read_csv('dataset/train_target.csv')

    # Load unlabeled test data
    transactions_test = pd.read_csv('dataset/transactions_test.csv')
    test_id = pd.read_csv('dataset/test.csv')

    # Aggregating features
    agg_features_train = transactions_train.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    agg_features_test = transactions_test.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

    # Counting transactions for small groups (for test data)
    counter_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_train = counter_df_train.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur')
    cat_counts_train = cat_counts_train.fillna(0)
    cat_counts_train.columns = ['small_group_' + str(i) for i in cat_counts_train.columns]

    counter_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_test = counter_df_test.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur')
    cat_counts_test = cat_counts_test.fillna(0)
    cat_counts_test.columns = ['small_group_' + str(i) for i in cat_counts_test.columns]

    # Merging features for train data
    train = pd.merge(train_target, agg_features_train, on='client_id')
    train = pd.merge(train, cat_counts_train.reset_index(), on='client_id')

    # Creating pseudo-labels for test data
    test = pd.merge(test_id, agg_features_test, on='client_id')
    test = pd.merge(test, cat_counts_test.reset_index(), on='client_id')

    common_features = list(set(train.columns).intersection(set(test.columns)))

    y_train = train['bins']
    X_train = train[common_features]
    X_test = test[common_features]

    # Train the initial model
    initial_model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_jobs=4, seed=42, n_estimators=300)
    initial_model.fit(X_train, y_train)

    pseudo_labels = initial_model.predict(X_test)

    # Combine labeled and pseudo-labeled data
    combined_X = pd.concat([X_train, X_test])
    combined_y = pd.concat([y_train, pd.Series(pseudo_labels, index=X_test.index)])

    # Train a new model using the combined dataset
    new_model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_jobs=4, seed=42, n_estimators=300)
    new_model.fit(combined_X, combined_y)

    pred = new_model.predict(X_test)

    submission = pd.DataFrame({'bins': pred}, index=X_test.client_id)
    return submission.to_html()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
