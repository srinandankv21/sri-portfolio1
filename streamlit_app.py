import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Set the style for plots
#plt.style.use('seaborn')

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data for 300 houses
num_houses = 300

# Max and Min Energy Consumed per Day (kWh)
max_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Max energy between 50 to 500 kWh
min_energy_consumed = np.random.randint(50, 500, size=num_houses)  # Min energy between 50 to 500 kWh

# Total Energy Consumed for the Month (assuming 30 days in a month)
total_energy_consumed = np.random.randint(3000, 15000, size=num_houses)  # Total monthly energy

# Create a DataFrame
df = pd.DataFrame({
    'Max Energy Consumed per Day (kWh)': max_energy_consumed,
    'Min Energy Consumed per Day (kWh)': min_energy_consumed,
    'Total Energy Consumed for the Month (kWh)': total_energy_consumed
})

# Display the first few rows of the data
print("Sample Data:")
print(df.head())

# --- Data Scaling ---
# Scaling the 'Max' and 'Min' energy consumed between 0 and 10
scaler = MinMaxScaler(feature_range=(0, 10))
scaled_data = scaler.fit_transform(df[['Max Energy Consumed per Day (kWh)', 'Min Energy Consumed per Day (kWh)']])

# Convert the scaled data back into a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['Max Energy Consumed per Day (Scaled)', 'Min Energy Consumed per Day (Scaled)'])

# --- K-means Clustering ---
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Get the centroids (cluster centers)
centroids = kmeans.cluster_centers_

# --- Visualization ---

# 1. Scatter Plot of Max and Min Energy Consumption Before Clustering
plt.figure(figsize=(8, 6))
plt.scatter(scaled_df['Max Energy Consumed per Day (Scaled)'], 
            scaled_df['Min Energy Consumed per Day (Scaled)'], 
            color='grey', alpha=0.6)
plt.title('Max and Min Energy Consumption (Before Clustering, Scaled)')
plt.xlabel('Max Energy Consumed per Day (Scaled)')
plt.ylabel('Min Energy Consumed per Day (Scaled)')
plt.show()

# 2. Scatter Plot of Clusters with Different Colors
plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    clustered_data = scaled_df[df['Cluster'] == cluster]
    plt.scatter(clustered_data['Max Energy Consumed per Day (Scaled)'], 
                clustered_data['Min Energy Consumed per Day (Scaled)'], 
                label=f'Cluster {cluster + 1}', alpha=0.6)
    
# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

plt.title('K-means Clustering of Max and Min Energy Consumption')
plt.xlabel('Max Energy Consumed per Day (Scaled)')
plt.ylabel('Min Energy Consumed per Day (Scaled)')
plt.legend()
plt.show()

# Display final data with clusters
print("Data with Cluster Labels:")
print(df.head())

