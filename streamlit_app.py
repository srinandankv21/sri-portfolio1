import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Title and Description
st.title('Energy Consumption Clustering App')
st.write('This app performs K-means clustering on energy consumption data across different cities in the state.')

# Load the dataset
st.write("### Sample Energy Consumption Data")
df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=365, freq='D'),
    'City': np.random.choice(['CityA', 'CityB', 'CityC', 'CityD'], 365),
    'Energy Consumption (kWh)': np.random.randint(1000, 5000, size=365)
})
st.write(df.head())

# User input for the number of clusters
num_clusters = st.slider('Select number of clusters', min_value=2, max_value=6, value=3)

# Prepare data for clustering
grouped_df = df.groupby('City')['Energy Consumption (kWh)'].sum().reset_index()

# Apply K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
grouped_df['Cluster'] = kmeans.fit_predict(grouped_df[['Energy Consumption (kWh)']])

# Display the clustered data
st.write("### Clustered Data")
st.write(grouped_df)

# Plot the clusters
fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i in range(num_clusters):
    cluster_data = grouped_df[grouped_df['Cluster'] == i]
    ax.scatter(cluster_data['City'], cluster_data['Energy Consumption (kWh)'], color=colors[i], label=f'Cluster {i}')

ax.set_xlabel('City')
ax.set_ylabel('Energy Consumption (kWh)')
ax.set_title('K-means Clustering of Energy Consumption')
ax.legend()
st.pyplot(fig)
