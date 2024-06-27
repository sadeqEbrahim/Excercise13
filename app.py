from flask import Flask, request, send_file, redirect, url_for, jsonify, render_template_string
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
import joblib
import os
from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model_file = 'label_prop_model.pkl'

def load_uploaded_files():
    transactions_train = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'transactions_train.csv'))
    train_target = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'train_target.csv'))
    transactions_test = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'transactions_test.csv'))
    test_id = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'test.csv'))
    return transactions_train, train_target, transactions_test, test_id

@app.route('/')
def home():
    return '''
    <h1>Upload Files</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <label for="transactions_train">Transactions Train:</label>
        <input type="file" id="transactions_train" name="transactions_train"><br><br>
        <label for="train_target">Train Target:</label>
        <input type="file" id="train_target" name="train_target"><br><br>
        <label for="transactions_test">Transactions Test:</label>
        <input type="file" id="transactions_test" name="transactions_test"><br><br>
        <label for="test_id">Test ID:</label>
        <input type="file" id="test_id" name="test_id"><br><br>
        <button type="submit">Upload</button>
    </form>

    <h2>Using curl</h2>
    <p>You can also upload files using curl:</p>
    <pre>
    <code>
curl -F "transactions_train=@/path/to/transactions_train.csv" \
     -F "train_target=@/path/to/train_target.csv" \
     -F "transactions_test=@/path/to/transactions_test.csv" \
     -F "test_id=@/path/to/test.csv" \
     http://localhost:8080/upload
    </code>
    </pre>
    <p>Run the model using curl:</p>
    <pre>
    <code>
curl -X POST http://localhost:8080/run_model
    </code>
    </pre>
    '''

@app.route('/upload', methods=['POST'])
def upload_files():
    for key in request.files:
        file = request.files[key]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return redirect(url_for('files_uploaded'))

@app.route('/files_uploaded')
def files_uploaded():
    return '''
    <h1>Files successfully uploaded</h1>
    <form action="/run_model" method="POST">
        <button type="submit">Run Model</button>
    </form>
    '''

@app.route('/run_model', methods=['POST'])
def run_model():
    transactions_train, train_target, transactions_test, test_id = load_uploaded_files()

    # Calculate aggregation features for each client
    agg_features_train = transactions_train.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    agg_features_test = transactions_test.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

    # Calculate transaction counts for each category for each client
    counter_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_train = counter_df_train.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur').fillna(0)
    cat_counts_train.columns = ['small_group_' + str(i) for i in cat_counts_train.columns]

    counter_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_test = counter_df_test.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur').fillna(0)
    cat_counts_test.columns = ['small_group_' + str(i) for i in cat_counts_test.columns]

    # Merge train data
    train = pd.merge(train_target, agg_features_train, on='client_id')
    train = pd.merge(train, cat_counts_train.reset_index(), on='client_id')

    # Merge test data
    test = pd.merge(test_id, agg_features_test, on='client_id')
    test = pd.merge(test, cat_counts_test.reset_index(), on='client_id')
    common_features = list(set(train.columns).intersection(set(test.columns)))
    y_train = train['bins']
    X_train = train[common_features]
    X_test = test[common_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=['client_id']))
    X_test_scaled = scaler.transform(X_test.drop(columns=['client_id']))

    # Check if the model file exists
    if os.path.exists(model_file):
        # Load the model from the file
        label_prop_model = joblib.load(model_file)
    else:
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

        # Save the model to the file
        joblib.dump(label_prop_model, model_file)

    # Predict test clusters
    test_clusters = label_prop_model.predict(X_test_scaled)

    # Prepare submission file
    submission = pd.DataFrame({'client_id': test['client_id'], 'cluster': test_clusters})
    submission['cluster'].plot(kind='hist', bins=20, title='Cluster Distribution')
    plt.gca().spines[['top', 'right']].set_visible(False)

    # Save the plot as an image
    plot_path = os.path.join('static', 'plot.png')
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig(plot_path)
    plt.close()

    return render_template_string('''
    <h1>Model Result</h1>
    <img src="{{ url_for('static', filename='plot.png') }}" alt="Cluster Distribution Plot">
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
