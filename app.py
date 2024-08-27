from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import scipy.cluster.hierarchy as sch
from sklearn. preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)

# Load the trained model
clf = joblib.load('random_forest_model.pkl')

# Define your preprocessing function
def pre_process(dataframe, scale=False, Custlocation=False):
    df = dataframe.copy()
    df.dropna(inplace=True)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'])
    df['CustomerAge'] = df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year
    df.drop(columns=['TransactionTime'], inplace=True)
    df.drop(df[df['CustGender'] == 'T'].index, inplace=True)
    df['TransactionDate1'] = df['TransactionDate']  # First transaction
    df['TransactionDate2'] = df['TransactionDate']  # Last transaction
    DF = df.groupby("CustomerID").agg({
        "TransactionID": "count",
        "CustGender": "first",
        "CustLocation": "first",
        "CustAccountBalance": "mean",
        "TransactionAmount (INR)": "mean",
        "CustomerAge": "median",
        "TransactionDate2": "max",
        "TransactionDate1": "min",
        "TransactionDate": "median"
    })
    DF = DF.reset_index()
    DF.drop(columns=["CustomerID"], inplace=True)
    DF.rename(columns={"TransactionID": "Frequency"}, inplace=True)
    DF['Recency'] = DF['TransactionDate2'] - DF['TransactionDate1']
    DF['Recency'] = DF['Recency'].astype(str)
    DF['Recency'] = DF['Recency'].apply(lambda x: int(re.search('\d+', x).group()))
    DF['Recency'] = DF['Recency'].apply(lambda i: 1 if i == 0 else i)
    DF.drop(columns=["TransactionDate1", "TransactionDate2"], inplace=True)
    DF['CustGender'] = DF['CustGender'].map({'M': 1, 'F': 0})
    DF = DF.drop(columns=['TransactionDate'])

    if Custlocation == False:
        DF = DF.drop(columns=['CustLocation'])
    DF = DF.reset_index(drop=True)
    
    if len(DF) > 1:
        DF_scaled = StandardScaler().fit_transform(DF)
        DF_scaled = pd.DataFrame(DF_scaled, columns=DF.columns)
    elif scale == True:
        DF_scaled = StandardScaler().fit_transform(DF)
        DF_scaled = pd.DataFrame(DF_scaled, columns=DF.columns)
        
    else:
        DF_scaled = DF  # If only one row, skip scaling

    return DF_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input from the form
    if request.method == 'POST':
        data = request.form
        df_test = pd.DataFrame({
            "CustomerID": [data['CustomerID']],
            "TransactionID": [data['TransactionID']],
            "CustGender": [data['CustGender']],
            "CustLocation": [data['CustLocation']],
            "CustAccountBalance": [int(data['CustAccountBalance'])],
            "TransactionAmount (INR)": [int(data['TransactionAmount (INR)'])],
            "CustomerDOB": [data['CustomerDOB']],
            "TransactionDate": [data['TransactionDate']],
            "TransactionTime": [int(data['TransactionTime'])]
        })

        # Preprocess the input data
        df_test = pre_process(df_test,scale=True)

        # Making predictions
        prediction = clf.predict(df_test)
        cluster_labels = {0: ' Is likely a financially stable, engaged, and valuable customer. Tailoring your approach to reinforce their loyalty and maximize their lifetime value would be a strategic move.',
                            1: 'They are  inactive. This cluster is primarily differentiated by gender. This group could benefit from targeted re-engagement strategies, such as promotional offers or personalized recommendations to increase their activity and spending.',
                            2: 'This group may be price-sensitive or have a limited budget, but they are actively engaged with the business. Potential strategies for this cluster could include loyalty programs or small rewards for frequent purchases to maintain their engagement and potentially increase their spending.',
                            3: 'It suggests that age is the dominant characteristic distinguishing this group from others. Consider tailoring your strategies or offers based on age-specific preferences and needs to effectively engage this cluster.',
                            4: 'It is one of the loyal customers who interact with your business often and recently. Focus on personalized marketing, loyalty rewards, and exclusive offers to enhance their engagement and retention.'}

        custom_prediction =[cluster_labels[pred] for pred in prediction]

        return render_template('predict.html', prediction_text=f'{custom_prediction}')
if __name__ == "__main__":
    app.run(debug=True)
