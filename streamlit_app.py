import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the style for plots
#plt.style.use('seaborn')

# Title and Description
st.title('Energy Consumption Clustering App')
st.write("""
This app performs K-means clustering on energy consumption data across different cities in a state over a year.
You can visualize the data before clustering, view the clusters, and see the cluster centroids.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# User input for the number of cities
num_cities = st.sidebar.slider('Number of Cities', min_value=10, max_value=100, value=50, step=10)

# User input for the number of clusters
num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4)

# Function to generate sample data
@st.cache
def generate_data(num_cities):
    np.random.seed(42)  # For reproducibility
    
    # Generate city names
    cities = [f'City_{i}' for i in range(1, num_cities + 1)]
    
    # Generate a date range for a year
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    
    # Generate energy consumption data
    data = {
        'Date': np.repeat(dates, num_cities),
        'City': cities * len(dates),
        'Energy Consumption (kWh)': np.random.randint(1000, 5000, size=len(dates) * num_cities)
    }
    
    df = pd.DataFrame(data)
    return df

# Load the dataset
df = generate_data(num_cities)

st.subheader("Sample Energy Consumption Data")
st.write(df.head())

# Data Preprocessing
st.subheader("Data Preprocessing")

# Aggregate data to get total and average energy consumption per city
agg_df = df.groupby('City').agg({
    'Energy Consumption (kWh)': ['sum', 'mean']
}).reset_index()

# Flatten the multi-level columns
agg_df.columns = ['City', 'Total Energy Consumption (kWh)', 'Average Daily Consumption (kWh)']

st.write("### Aggregated Data per City")
st.write(agg_df.head())

# Feature Selection for Clustering
features = agg_df[['Total Energy Consumption (kWh)', 'Average Daily Consumption (kWh)']]

# K-means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
agg_df['Cluster'] = kmeans.fit_predict(features)

# Assign colors to clusters
colors = plt.cm.get_cmap('tab10', num_clusters)

# --- 1. Before Clustering: Display All Data Points ---
st.subheader("1. Energy Consumption Before Clustering")

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(
    agg_df['Total Energy Consumption (kWh)'],
    agg_df['Average Daily Consumption (kWh)'],
    color='grey',
    alpha=0.6
)
ax1.set_xlabel('Total Energy Consumption (kWh)')
ax1.set_ylabel('Average Daily Consumption (kWh)')
ax1.set_title('Energy Consumption Across Cities (Before Clustering)')
st.pyplot(fig1)

# --- 2. After Clustering: Display Clusters in Different Colors ---
st.subheader("2. Energy Consumption After K-means Clustering")

fig2, ax2 = plt.subplots(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = agg_df[agg_df['Cluster'] == cluster]
    ax2.scatter(
        cluster_data['Total Energy Consumption (kWh)'],
        cluster_data['Average Daily Consumption (kWh)'],
        color=colors(cluster),
        label=f'Cluster {cluster + 1}',
        alpha=0.6
    )
ax2.set_xlabel('Total Energy Consumption (kWh)')
ax2.set_ylabel('Average Daily Consumption (kWh)')
ax2.set_title('K-means Clustering of Energy Consumption')
ax2.legend()
st.pyplot(fig2)

# --- 3. Centroids Visualization ---
st.subheader("3. Cluster Centroids with Data Points")

fig3, ax3 = plt.subplots(figsize=(8, 6))
# Plot data points
for cluster in range(num_clusters):
    cluster_data = agg_df[agg_df['Cluster'] == cluster]
    ax3.scatter(
        cluster_data['Total Energy Consumption (kWh)'],
        cluster_data['Average Daily Consumption (kWh)'],
        color=colors(cluster),
        label=f'Cluster {cluster + 1}',
        alpha=0.6
    )

# Plot centroids
centroids = kmeans.cluster_centers_
ax3.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

ax3.set_xlabel('Total Energy Consumption (kWh)')
ax3.set_ylabel('Average Daily Consumption (kWh)')
ax3.set_title('K-means Clustering with Centroids')
ax3.legend()
st.pyplot(fig3)

# Display centroids data
st.subheader("Cluster Centroids")
centroids_df = pd.DataFrame(centroids, columns=['Total Energy Consumption (kWh)', 'Average Daily Consumption (kWh)'])
centroids_df.index = [f'Cluster {i + 1}' for i in range(num_clusters)]
st.write(centroids_df)

# Optional: Display clustered data
st.subheader("Clustered Data")
st.write(agg_df)

# Optional: Download clustered data
@st.cache
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(agg_df)

st.download_button(
    label="Download Clustered Data as CSV",
    data=csv,
    file_name='clustered_energy_consumption.csv',
    mime='text/csv',
)
