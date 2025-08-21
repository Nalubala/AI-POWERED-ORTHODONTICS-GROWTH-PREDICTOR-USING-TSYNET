# cluster.py

import numpy as np
from sklearn.cluster import KMeans
import joblib

# Load features
features = np.load("features.npy")

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# Save cluster labels
np.save("clusters.npy", clusters)
joblib.dump(kmeans, "kmeans_model.joblib")

print("[INFO] Clustering complete. Saved clusters.npy and kmeans_model.joblib.")
