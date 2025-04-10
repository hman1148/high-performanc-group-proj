import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIRECTORY = "output"

def visualize_clusters(file_name, output_path, show_plot=True):
    file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

    df = pd.read_csv(file_path)

    # Filter out only the points (not centroids)
    points = df[df["is_centroid"] == False]

    # Check for required features
    required_features = ["valence", "danceability", "energy"]
    for feature in required_features:
        if feature not in points.columns:
            raise ValueError(f"Missing feature: {feature}")

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        points["valence"],
        points["danceability"],
        points["energy"],
        c=points["cluster_id"],
        cmap="tab10",
        s=20,
        alpha=0.8
    )

    ax.set_xlabel("Valence")
    ax.set_ylabel("Danceability")
    ax.set_zlabel("Energy")
    ax.set_title("3D Clustering Visualization")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster ID", loc="upper right")
    ax.add_artist(legend)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_clusters.py <results_file.csv>")
        sys.exit(1)



    input_file = sys.argv[1]

    # Generate default output file name
    if "_results.csv" in input_file:
        base_name = input_file.split("_results.csv")[0]
        output_file = f"./output/{base_name}_visualization.png"
    else:
        output_file = "./output/cluster_plot.png"  # fallback

    visualize_clusters(input_file, output_file)