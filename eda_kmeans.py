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

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


df=pd.read_csv("bank_transactions.csv")
df=df.sample(n=10000,random_state=42).reset_index(drop=True)
df.shape

def pre_process(dataframe,scale=False,Custlocation=False):
  df=dataframe.copy()
  df.dropna(inplace=True)
  df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
  df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'])
  df['CustomerAge'] =df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year
  df.drop(columns=['TransactionTime'],inplace=True)
  df.drop(df[df['CustGender']=='T'].index,inplace=True)
  df['TransactionDate1']=df['TransactionDate'] # ==> to calculate the minimum (first transaction)
  df['TransactionDate2']=df['TransactionDate'] # ==> to calculate the maximum (last transaction)
  DF = df.groupby("CustomerID").agg({
                                        "TransactionID" : "count",
                                        "CustGender" : "first",
                                        "CustLocation":"first",
                                        "CustAccountBalance"  : "mean",
                                        "TransactionAmount (INR)" : "mean",
                                        "CustomerAge" : "median",
                                        "TransactionDate2":"max",
                                        "TransactionDate1":"min",
                                        "TransactionDate":"median"
                        })
  DF = DF.reset_index()
  DF.drop(columns=["CustomerID"],inplace=True)
  DF.rename(columns={"TransactionID":"Frequency"},inplace=True)
  DF['Recency']=DF['TransactionDate2']-DF['TransactionDate1']
  DF['Recency']=DF['Recency'].astype(str)
  DF['Recency']=DF['Recency'].apply(lambda x :re.search('\d+',x).group())
  DF['Recency']=DF['Recency'].astype(int)
  DF['Recency']=DF['Recency'].apply(rep_0)
  DF.drop(columns=["TransactionDate1","TransactionDate2"],inplace=True)
  DF['CustGender']=DF['CustGender'].map({'M':1,'F':0})
  DF=DF.drop(columns=['TransactionDate'])

  if Custlocation==False:
      DF=DF.drop(columns=['CustLocation'])
  DF=DF.reset_index(drop=True)
  # Check if the DataFrame has more than one row before scaling
  if len(DF) > 1 and scale==True:
      DF_scaled = StandardScaler().fit_transform(DF)
      DF_scaled = pd.DataFrame(DF_scaled, columns=DF.columns)
  else:
      DF_scaled = DF  # If only one row, skip scaling

  return DF_scaled
  #Frequency	CustGender	CustAccountBalance	TransactionAmount (INR)	CustomerAge	Recency

def rep_0(i):
    if i==0:
        return 1
    else:
        return i

def calculate_metrics(labels, data):
    silhouette = silhouette_score(data, labels)
    ch_index = calinski_harabasz_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    return silhouette, ch_index, db_index

def get_user_input():
    customer_id = str(input("Enter CustomerID: "))
    transaction_id = str(input("Enter TransactionID: "))
    cust_gender = str(input("Enter CustGender (M/F): "))
    cust_location = str(input("Enter CustLocation: "))
    cust_account_balance = int(input("Enter CustAccountBalance (INR): "))
    transaction_amount = int(input("Enter TransactionAmount (INR): "))
    customer_dob = str(input("Enter CustomerDOB (MM/DD/YYYY): "))
    transaction_date = str(input("Enter TransactionDate (MM/DD/YYYY): "))
    transaction_time = int(input("Enter TransactionTime (HHMMSS): "))

    # Creating DataFrame
    df = pd.DataFrame({
        "CustomerID": [customer_id],
        "TransactionID": [transaction_id],
        "CustGender": [cust_gender],
        "CustLocation": [cust_location],
        "CustAccountBalance": [cust_account_balance],
        "TransactionAmount (INR)": [transaction_amount],
        "CustomerDOB": [customer_dob],
        "TransactionDate": [transaction_date],
        "TransactionTime": [transaction_time]
    })
    return df

df_scaled=pre_process(df,scale=True)
df_EDA=pre_process(df,scale=False,Custlocation=True)
# df_h=df_scaled.sample(n=100000,random_state=42).reset_index(drop=True)
# df_d=df_h.copy()

num_col = df_EDA.select_dtypes(include=np.number)
cat_col = df_EDA.select_dtypes(exclude=np.number)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(30,30))
for index,column in enumerate(num_col):
    plt.subplot(7,4,index+1)
    sns.boxplot(data=num_col,x=column)

plt.tight_layout(pad = 1.0)
plt.show()

# correlation between features
plt.figure(figsize=(7,5))
correlation=df_EDA.drop(columns=["CustLocation"]).corr()
sns.heatmap(correlation,vmin=None,
    vmax=0.8,
    cmap='rocket_r',
    annot=True,
    fmt='.1f',
    linecolor='white',
    cbar=True)
plt.show()

#Count Plot
plt.style.use("fivethirtyeight")
chart=sns.countplot(x='Frequency',data=df_EDA,palette='rocket', order = df_EDA['Frequency'].value_counts().index)
plt.title("Frequency",
          fontsize='20',
          backgroundcolor='AliceBlue',
          color='magenta')
plt.show()

#color_palette
plt.style.use("fivethirtyeight")
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(15,5))
palette_color = sns.color_palette('rocket')
ax[0].hist(x=df_EDA['CustomerAge'],color='purple')
ax[0].set_title("Distribution of Customer Age")
ax[1].pie(df_EDA['CustGender'].value_counts(),autopct='%1.f%%',colors=palette_color,labels=['Male','Female'])
ax[1].set_title("Customer Gender")
plt.tight_layout()
plt.show()

# CountPlot
plt.style.use("fivethirtyeight")
plt.figure(figsize=(15,7))
chart=sns.countplot(y='CustLocation',data=df_EDA,palette='rocket', order =df_EDA['CustLocation'].value_counts()[:20].index)
plt.title("Most 20 Location of Customer ",
          fontsize='20',
          backgroundcolor='AliceBlue',
          color='magenta')
plt.show() 

#ScatterPlot
plt.style.use("fivethirtyeight")
sns.scatterplot(x='TransactionAmount (INR)',y='CustAccountBalance',data=df_EDA,palette='rocket',hue='Frequency',size='Recency' )
plt.title("TransactionAmount (INR) and CustAccountBalance",
          fontsize='20',
          backgroundcolor='AliceBlue',
          color='magenta')
plt.show() 

# # Kmeans algorithm settings
kmeans_set = {"init":"random", "max_iter":300, "random_state":42}

inertias = []
silhouette_coefficients = []

for k in range(2, 9):
    kmeans = MiniBatchKMeans(n_clusters=k, **kmeans_set, batch_size=10000)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
fig,ax=plt.subplots(1,2,figsize=(20,5))
kl = KneeLocator(range(2,9), inertias, curve='convex', direction="decreasing")
ax[0].plot(range(2,9), inertias,color='purple',marker='o',label='Elbow Method')
ax[0].set_xticks(range(2,9))
ax[0].set_xlabel("Number of Clusters", labelpad=20)
ax[0].set_ylabel("Inertia", labelpad=20)
ax[0].set_title("Elbow Method")
ax[0].axvline(x=kl.elbow, color='black', label='axvline-fullheight', ls='--', linewidth=3)

ax[1].plot(range(2,9), silhouette_coefficients,color='purple',marker='o',label='silhouette coefficients')
ax[1].set_xticks(range(2,9))
ax[1].set_title("silhouette values")
ax[1].set_xlabel("Number of clusters")
ax[1].set_ylabel("silhouette_coefficients")
plt.tight_layout();

plt.figure(figsize=(30,10))
z = sch.linkage(df_scaled.iloc[:500,:], 'ward')
sch.dendrogram(z);

#the three charts insure that the number of clustering is 5 so I will use it
kmeans_set = {"init":"random", "max_iter":300, "random_state":42}
kmeans = KMeans(n_clusters=5 , **kmeans_set)
kmeans.fit(df_scaled)
df_scaled['Label']=kmeans.labels_
kmeans_silhouette, kmeans_ch, kmeans_dbi = calculate_metrics(df_scaled['Label'], df_scaled.iloc[:,:-1])
print(f"KMeans - Silhouette Score: {kmeans_silhouette}, CH Index: {kmeans_ch}, DBI: {kmeans_dbi}")

attributes = df_scaled.columns[:5]
colors = ['purple', 'yellow', 'red', 'blue', 'orange']
cluster = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']


fig = go.Figure()

for index in range(0, 5):
    # Calculate the mean of each attribute for the current cluster
    means = df_scaled.loc[df_scaled.Label == index, attributes].mean(axis=0).values

    fig.add_trace(go.Scatterpolar(
        r=means,  # This should now be an array of means
        theta=attributes,
        fill='toself',
        fillcolor=colors[index],
        opacity=0.4,
        name=cluster[index]
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    title="Radar plot - Cluster attributes (Normalized Values)"
)
fig.show()

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(df_h)

hierarchical_silhouette, hierarchical_ch, hierarchical_dbi = calculate_metrics(hierarchical_labels, df_h)
print(f"Hierarchical - Silhouette Score: {hierarchical_silhouette}, CH Index: {hierarchical_ch}, DBI: {hierarchical_dbi}")

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(df_d)

# Filter out noise points (labeled as -1)
dbscan_labels_filtered = dbscan_labels[dbscan_labels != -1]
df_scaled_filtered_d = df_d[dbscan_labels != -1]

if len(set(dbscan_labels_filtered)) == 5:  # Check if 5 clusters were found
    dbscan_silhouette, dbscan_ch, dbscan_dbi = calculate_metrics(dbscan_labels_filtered, df_scaled_filtered_d)
    print(f"DBSCAN - Silhouette Score: {dbscan_silhouette}, CH Index: {dbscan_ch}, DBI: {dbscan_dbi}")
else:
    print(f"DBSCAN resulted in {len(set(dbscan_labels_filtered))} clusters, unable to enforce exactly 5 clusters.")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = df_scaled.drop(columns=['Label'])
y = df_scaled['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Training the model
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
#Add Labels
cluster_labels = {0: ' Is likely a financially stable, engaged, and valuable customer. Tailoring your approach to reinforce their loyalty and maximize their lifetime value would be a strategic move.',
                  1: 'They are  inactive. This cluster is primarily differentiated by gender. This group could benefit from targeted re-engagement strategies, such as promotional offers or personalized recommendations to increase their activity and spending.',
                  2: 'This group may be price-sensitive or have a limited budget, but they are actively engaged with the business. Potential strategies for this cluster could include loyalty programs or small rewards for frequent purchases to maintain their engagement and potentially increase their spending.',
                  3: 'It suggests that age is the dominant characteristic distinguishing this group from others. Consider tailoring your strategies or offers based on age-specific preferences and needs to effectively engage this cluster.',
                  4: 'It is one of the loyal customers who interact with your business often and recently. Focus on personalized marketing, loyalty rewards, and exclusive offers to enhance their engagement and retention.'}

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

