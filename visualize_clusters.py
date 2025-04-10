import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIRECTORY = "output"

def visualize_clusters(file_name, output_path, show_plot=True):
    file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)

    # Filter points and centroids
    points = df[df["is_centroid"] == False]
    centroids = df[df["is_centroid"] == True]

    # Check required features
    required_features = ["valence", "danceability", "energy"]
    for feature in required_features:
        if feature not in points.columns:
            raise ValueError(f"Missing feature: {feature}")

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    scatter = ax.scatter(
        points["valence"],
        points["danceability"],
        points["energy"],
        c=points["cluster_id"],
        cmap="tab10",
        s=5,
        alpha=0.6
    )

    # Plot centroids
    ax.scatter(
        centroids["valence"],
        centroids["danceability"],
        centroids["energy"],
        c="black",
        s=100,
        marker="X",
        label="Centroid"
    )

    # Adjust view angle
    ax.view_init(elev=30, azim=135)

    # Labels and legend
    ax.set_xlabel("Valence")
    ax.set_ylabel("Danceability")
    ax.set_zlabel("Energy")
    ax.set_title("3D Clustering Visualization")

    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(loc="lower left")  # for centroid label

    plt.tight_layout()
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to: {output_path}")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_clusters.py <results_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate default output filename
    if "_results.csv" in input_file:
        base_name = input_file.split("_results.csv")[0]
        output_file = f"{OUTPUT_DIRECTORY}/{base_name}_visualization.png"
    else:
        output_file = f"{OUTPUT_DIRECTORY}/cluster_plot.png"

    visualize_clusters(input_file, output_file)