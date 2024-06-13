from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def home():
    # Define the data path
    data_path = 'dataset/'

    # Load the datasets
    transactions_train = pd.read_csv(data_path + 'transactions_train.csv')
    train_target = pd.read_csv(data_path + 'train_target.csv')
    transactions_test = pd.read_csv(data_path + 'transactions_test.csv')
    test_id = pd.read_csv(data_path + 'test.csv')

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

    # Perform KMEAN clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    train['cluster'] = kmeans.fit_predict(X_train_scaled)
    test['cluster'] = kmeans.predict(X_test_scaled)

    # Prepare the file to be sent to the system
    submission = pd.DataFrame({'client_id': test['client_id'], 'cluster': test['cluster']})
    submission['cluster'].plot(kind='hist', bins=20, title='Cluster Distribution')
    plt.gca().spines[['top', 'right']].set_visible(False)

    # Save the plot as an image
    plt.savefig('static/plot.png')

    # Render the HTML file
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
