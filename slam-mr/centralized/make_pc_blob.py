import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Total number of points
total_points = 1000
rectangle_points = int(total_points * 0.4)
circle_points = int(total_points * 0.3)

# Rectangle dimensions
rect_width = 4
rect_height = 2

# Random points along rectangle edges
def random_rectangle_points(n, width, height):
    sides = ['top', 'right', 'bottom', 'left']
    points_per_side = np.random.multinomial(n, [0.25]*4)
    xs, ys = [], []

    # Top
    xs += list(np.random.uniform(-width/2, width/2, points_per_side[0]))
    ys += [height/2] * points_per_side[0]
    # Right
    xs += [width/2] * points_per_side[1]
    ys += list(np.random.uniform(-height/2, height/2, points_per_side[1]))
    # Bottom
    xs += list(np.random.uniform(width/2, -width/2, points_per_side[2]))
    ys += [-height/2] * points_per_side[2]
    # Left
    xs += [-width/2] * points_per_side[3]
    ys += list(np.random.uniform(height/2, -height/2, points_per_side[3]))

    return np.array(xs), np.array(ys)

# Random circle point generation
def random_circle(center_x, center_y, radius, n_points):
    angles = np.random.uniform(0, 2*np.pi, n_points)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return x, y

# Generate all shape points
x_rect, y_rect = random_rectangle_points(rectangle_points, rect_width, rect_height)
x_c1, y_c1 = random_circle(-1, 0, 0.5, circle_points)
x_c2, y_c2 = random_circle(1, 0, 0.5, circle_points)

# Combine and shuffle
x_all = np.concatenate([x_rect, x_c1, x_c2])
y_all = np.concatenate([y_rect, y_c1, y_c2])

# Randomly jitter points
jitter_x = np.random.normal(0, 0.01, total_points)
jitter_y = np.random.normal(0, 0.01, total_points)
x_all += jitter_x
y_all += jitter_y

# Split into 4 quadrants
regions = []
for x, y in zip(x_all, y_all):
    if x < 0 and y >= 0:
        regions.append("top-left")
    elif x < 0 and y < 0:
        regions.append("bottom-left")
    elif x >= 0 and y >= 0:
        regions.append("top-right")
    else:
        regions.append("bottom-right")



# Create DataFrame
df = pd.DataFrame({'x': x_all, 'y': y_all, 'region': regions})

from sklearn.cluster import KMeans

# Number of blobs/clusters
k = 10  # You can change this to any integer

# Prepare data for clustering
points = np.column_stack((x_all, y_all))

# Perform k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(points)

# Add cluster labels to DataFrame
df['cluster'] = labels

# Plot the clusters
plt.figure(figsize=(6, 4))
for label in np.unique(labels):
    cluster_data = df[df['cluster'] == label]
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {label}', s=10)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', marker='x', label='Centers')
plt.title(f"K-Means Clustering with k={k}")
plt.legend()
plt.grid(True)
plt.show()

# Prepare data to save
data_to_save = {
    'points': points,                          # shape (N, 2)
    'labels': labels,                          # shape (N,)
    'cluster_centers': kmeans.cluster_centers_ # shape (k, 2)
}

# Save to file
np.savez('point_cloud_with_clusters_many.npz', **data_to_save)
print("Saved point cloud and clusters to 'point_cloud_with_clusters.npz'")


# # Plot
# plt.figure(figsize=(6, 4))
# colors = {'top-left': 'red', 'bottom-left': 'blue', 'top-right': 'green', 'bottom-right': 'orange'}
# for region in df['region'].unique():
#     plt.scatter(df[df['region'] == region]['x'], df[df['region'] == region]['y'],
#                 label=region, color=colors[region], s=10)
# plt.axhline(0, color='gray', linestyle='--')
# plt.axvline(0, color='gray', linestyle='--')
# plt.title("Randomized 2D Point Cloud with Quadrants")
# plt.legend()
# plt.grid(True)
# plt.show()
