# Data Science Project: Customer Segmentation
# Used dataset: Mall_Customers.csv from https://www.kaggle.com/code/kushal1996/customer-segmentation-k-means-analysis/input
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")
df = df.drop("CustomerID", axis=1)  # Drop CustomerID column

# Descriptive Statistics
descriptive_stats = df.describe()
print("Descriptive Statistics:\n", descriptive_stats)

# Missing Data Analysis
missing_values = df.isnull().sum()
print("Missing Data Analysis:\n", missing_values)

# Select features to use
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train K-Means model
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize clustering results
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(optimal_k):
    plt.scatter(X[df['Cluster'] == i]['Annual Income (k$)'], 
                X[df['Cluster'] == i]['Spending Score (1-100)'], 
                s=100, c=colors[i], label=f'Cluster {i}')

# Show cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='black', marker='X', label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation - K-Means')
plt.legend()
plt.show()
