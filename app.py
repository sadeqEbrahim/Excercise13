from flask import Flask, render_template, send_file
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Function to calculate the number of transactions for each category for each client
def get_cat_counts(df):
    counter_df = df.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts = counter_df.unstack(fill_value=0)
    cat_counts.columns = ['small_group_' + str(i) for i in cat_counts.columns]
    return cat_counts

@app.route('/')
def index():
    # Load the datasets
    transactions_train = pd.read_csv('dataset/transactions_train.csv')
    transactions_test = pd.read_csv('dataset/transactions_test.csv')
    test_id = pd.read_csv('dataset/test.csv')

    # Calculate aggregation features for each client
    agg_features_train = transactions_train.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    agg_features_test = transactions_test.groupby('client_id')['amount_rur'].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()

    # Get categorical counts
    cat_counts_train = get_cat_counts(transactions_train)
    cat_counts_test = get_cat_counts(transactions_test)

    # Merge all the files into a single dataframe
    train = pd.merge(agg_features_train, cat_counts_train.reset_index(), on='client_id')
    test = pd.merge(agg_features_test, cat_counts_test.reset_index(), on='client_id')

    # Ensure consistency in feature space
    common_features = list(set(train.columns).intersection(set(test.columns)))
    X_train = train[common_features].drop(columns=['client_id'])
    X_test = test[common_features].drop(columns=['client_id'])

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform KMEAN clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    train['cluster'] = kmeans.fit_predict(X_train_scaled)
    test['cluster'] = kmeans.predict(X_test_scaled)

    # Prepare the file to be sent to the system
    submission = pd.DataFrame({'client_id': test['client_id'], 'cluster': test['cluster']})
    submission['cluster'].plot(kind='hist', bins=20, title='Cluster Distribution')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.savefig('static/plot.png')
    plt.close()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
