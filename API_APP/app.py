from flask import Flask, render_template, send_file, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from matplotlib import pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Define the data path
data_path = 'dataset/'

# Load the datasets
transactions_train = pd.read_csv(data_path + 'transactions_train.csv')
train_target = pd.read_csv(data_path + 'train_target.csv')
transactions_test = pd.read_csv(data_path + 'transactions_test.csv')
test_id = pd.read_csv(data_path + 'test.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data_shapes')
def data_shapes():
    shapes = {
        'transactions_train': transactions_train.shape,
        'train_target': train_target.shape,
        'transactions_test': transactions_test.shape,
        'test_id': test_id.shape
    }
    return jsonify(shapes)

@app.route('/data_heads')
def data_heads():
    heads = {
        'transactions_train': transactions_train.head().to_html(),
        'train_target': train_target.head().to_html(),
        'transactions_test': transactions_test.head().to_html(),
        'test_id': test_id.head().to_html()
    }
    return jsonify(heads)

@app.route('/run_model')
def run_model():
    # Calculate the simplest aggregation signs for each client
    agg_features_train = transactions_train.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    agg_features_test = transactions_test.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

    # Calculate the number of transactions for each category for each client
    counter_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_train = counter_df_train.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur').fillna(0)
    cat_counts_train.columns = ['small_group_' + str(i) for i in cat_counts_train.columns]

    counter_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_test = counter_df_test.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur').fillna(0)
    cat_counts_test.columns = ['small_group_' + str(i) for i in cat_counts_test.columns]

    # Merge all the files into a single dataframe
    train = pd.merge(agg_features_train, cat_counts_train.reset_index(), on='client_id')
    test = pd.merge(agg_features_test, cat_counts_test.reset_index(), on='client_id')

    # Ensure consistency in feature space
    common_features = list(set(train.columns).intersection(set(test.columns)))
    X_train = train[common_features]
    X_test = test[common_features]

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=['client_id']))
    X_test_scaled = scaler.transform(X_test.drop(columns=['client_id']))

    # Semi-supervised clustering using Label Spreading with batch processing
    batch_size = 1000
    num_batches = len(X_train_scaled) // batch_size

    label_prop_model = LabelSpreading(kernel='rbf', gamma=0.25)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train_scaled))
        batch_X = X_train_scaled[start_idx:end_idx]
        batch_y = train_target['bins'].iloc[start_idx:end_idx]
        label_prop_model.fit(batch_X, batch_y)

    test_clusters = label_prop_model.predict(X_test_scaled)
    
    # Prepare submission file
    submission = pd.DataFrame({'client_id': test['client_id'], 'cluster': test_clusters})
    submission['cluster'].plot(kind='hist', bins=20, title='Cluster Distribution')
    plt.gca().spines[['top', 'right']].set_visible(False)

    # Save the plot as an image
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/plot.png')
    plt.close()

    return send_file('static/plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
