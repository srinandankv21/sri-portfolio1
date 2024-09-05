import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set the style for plots
#plt.style.use('seaborn')

# Title and Description
st.title('Energy Consumption Clustering App')
st.write("""
This app performs K-means clustering based on the **Average Daily Energy Consumption** across different users in a state over a year.
You can visualize the data before clustering, view the clusters, and see the cluster centroids.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# User input for the number of users
num_users = 300  # Fixed at 30,000 users

# User input for the number of clusters
num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=4)

# Function to generate sample data
@st.cache
def generate_data(num_users):
    np.random.seed(42)  # For reproducibility
    
    # Generate user IDs
    users = [f'User_{i}' for i in range(1, num_users + 1)]
    
    # Generate a date range for a year
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    
    # Generate energy consumption data
    data = {
        'Date': np.repeat(dates, num_users),
        'User': users * len(dates),
        'Energy Consumption (kWh)': np.random.randint(100, 4000, size=len(dates) * num_users)
    }
    
    df = pd.DataFrame(data)
    return df

# Load the dataset
df = generate_data(num_users)

st.subheader("Sample Energy Consumption Data")
st.write(df.head())

# Data Preprocessing
st.subheader("Data Preprocessing")

# Aggregate data to get total and average energy consumption per user
agg_df = df.groupby('User').agg({
    'Energy Consumption (kWh)': ['mean']
}).reset_index()

# Flatten the multi-level columns
agg_df.columns = ['User', 'Average Daily Consumption (kWh)']

st.write("### Aggregated Data per User")
st.write(agg_df.head())

# Select only 'Average Daily Consumption (kWh)' for clustering
average_consumption = agg_df[['Average Daily Consumption (kWh)']]

# Scaling the data between 0 and 10 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 10))
scaled_average_consumption = scaler.fit_transform(average_consumption)

# K-means Clustering on scaled data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
agg_df['Cluster'] = kmeans.fit_predict(scaled_average_consumption)

# Assign colors to clusters
colors = plt.cm.get_cmap('tab10', num_clusters)

# --- 1. Before Clustering: Display All Data Points ---
st.subheader("1. Average Daily Consumption Before Clustering (Scaled Data)")

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(
    np.arange(len(scaled_average_consumption)),
    scaled_average_consumption[:, 0],
    color='grey',
    alpha=0.6
)
ax1.set_xlabel('Users')
ax1.set_ylabel('Average Daily Consumption (kWh) (Scaled)')
ax1.set_title('Average Daily Consumption Across Users (Before Clustering, Scaled)')
st.pyplot(fig1)

# --- 2. After Clustering: Display Clusters in Different Colors ---
st.subheader("2. Average Daily Consumption After K-means Clustering (Scaled Data)")

fig2, ax2 = plt.subplots(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = scaled_average_consumption[agg_df['Cluster'] == cluster]
    ax2.scatter(
        np.where(agg_df['Cluster'] == cluster),
        cluster_data[:, 0],
        color=colors(cluster),
        label=f'Cluster {cluster + 1}',
        alpha=0.6
    )
ax2.set_xlabel('Users')
ax2.set_ylabel('Average Daily Consumption (kWh) (Scaled)')
ax2.set_title('K-means Clustering of Average Daily Consumption (Scaled)')
ax2.legend()
st.pyplot(fig2)

# --- 3. Centroids Visualization ---
st.subheader("3. Cluster Centroids with Data Points (Scaled Data)")

fig3, ax3 = plt.subplots(figsize=(8, 6))
# Plot data points
for cluster in range(num_clusters):
    cluster_data = scaled_average_consumption[agg_df['Cluster'] == cluster]
    ax3.scatter(
        np.where(agg_df['Cluster'] == cluster),
        cluster_data[:, 0],
        color=colors(cluster),
        label=f'Cluster {cluster + 1}',
        alpha=0.6
    )

# Plot centroids
centroids = kmeans.cluster_centers_
ax3.scatter(
    np.arange(num_clusters),
    centroids[:, 0],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

ax3.set_xlabel('Users')
ax3.set_ylabel('Average Daily Consumption (kWh) (Scaled)')
ax3.set_title('K-means Clustering with Centroids (Scaled)')
ax3.legend()
st.pyplot(fig3)

# Display centroids data (scaled)
st.subheader("Cluster Centroids (Scaled)")
centroids_df = pd.DataFrame(centroids, columns=['Average Daily Consumption (kWh) (Scaled)'])
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
    file_name='clustered_average_consumption.csv',
    mime='text/csv',
)
