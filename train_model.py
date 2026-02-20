"""
Customer Segmentation - K-Means Model Training (Advanced)
Loads Mall_Customers.csv, applies StandardScaler, trains K-Means on scaled data,
extracts centroids, generates cluster visualization, and saves model, scaler, and centroids.
"""

import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_FILE = "Mall_Customers.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
CENTROIDS_FILE = "centroids.json"
VISUALIZATION_FILE = "static/cluster.png"
N_CLUSTERS = 5
RANDOM_STATE = 42
FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

# Cluster mapping for visualization colors
CLUSTER_COLORS = {
    0: "#6b7280",  # Gray
    1: "#059669",  # Green
    2: "#ea580c",  # Orange
    3: "#dc2626",  # Red
    4: "#2563eb",  # Blue
}


def main():
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    # Extract features
    X = df[FEATURE_COLUMNS].values
    print(f"Dataset loaded: {len(X)} samples")

    # Initialize and fit StandardScaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means on scaled data
    print(f"Training K-Means with {N_CLUSTERS} clusters...")
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    model.fit(X_scaled)

    # Get cluster labels for visualization
    labels = model.labels_

    # Extract centroids (in scaled space)
    centroids_scaled = model.cluster_centers_

    # Inverse transform centroids back to original scale
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Prepare centroids data for JSON
    centroids_data = {}
    for i in range(N_CLUSTERS):
        centroids_data[str(i)] = {
            "income": float(centroids_original[i][0]),
            "score": float(centroids_original[i][1]),
        }

    # Save model
    print(f"Saving model to {MODEL_FILE}...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    print(f"Saving scaler to {SCALER_FILE}...")
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    # Save centroids to JSON
    print(f"Saving centroids to {CENTROIDS_FILE}...")
    with open(CENTROIDS_FILE, "w") as f:
        json.dump(centroids_data, f, indent=2)

    # Generate cluster visualization
    print("Generating cluster visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot data points colored by cluster
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = labels == cluster_id
        plt.scatter(
            X[cluster_mask, 0],
            X[cluster_mask, 1],
            c=CLUSTER_COLORS[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    # Plot centroids (in original scale)
    plt.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        c="black",
        marker="X",
        s=300,
        label="Centroids",
        edgecolors="white",
        linewidth=2,
        zorder=10,
    )

    # Customize plot
    plt.xlabel("Annual Income (k$)", fontsize=12, fontweight="bold")
    plt.ylabel("Spending Score (1-100)", fontsize=12, fontweight="bold")
    plt.title("K-Means Customer Segmentation", fontsize=16, fontweight="bold", pad=20)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save visualization
    plt.savefig(VISUALIZATION_FILE, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {VISUALIZATION_FILE}")
    plt.close()

    print("\n✅ Training completed successfully!")
    print(f"   - Model: {MODEL_FILE}")
    print(f"   - Scaler: {SCALER_FILE}")
    print(f"   - Centroids: {CENTROIDS_FILE}")
    print(f"   - Visualization: {VISUALIZATION_FILE}")


if __name__ == "__main__":
    main()
