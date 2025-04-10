import sys
import os
import pandas as pd

OUTPUT_DIRECTORY = "output"

def read_csv_results(file_path):
    """Reads clustering results from a CSV, separating points and centroids."""
    df = pd.read_csv(file_path)

    # Split into points and centroids
    points = df[df["is_centroid"] == False].copy()
    centroids = df[df["is_centroid"] == True].copy()

    # Drop index and is_centroid from points for clean comparison
    points_sorted = points.drop(columns=["index", "is_centroid"]).sort_values(by=points.columns.tolist()).reset_index(drop=True)
    centroids_sorted = centroids.drop(columns=["index", "is_centroid", "cluster_id"]).sort_values(by=centroids.columns.tolist()).reset_index(drop=True)

    return points_sorted, centroids_sorted

def compare_files(filenames):
    if len(filenames) < 2:
        print("Need at least two files to compare!")
        return

    file_paths = [os.path.join(OUTPUT_DIRECTORY, f) for f in filenames]
    ref_points, ref_centroids = read_csv_results(file_paths[0])

    for file in file_paths[1:]:
        curr_points, curr_centroids = read_csv_results(file)

        if not ref_points.equals(curr_points):
            print(f"Mismatch in point assignments between {file_paths[0]} and {file}")
            return

        if not ref_centroids.equals(curr_centroids):
            print(f"Mismatch in centroids between {file_paths[0]} and {file}")
            return

    print("All implementations produce identical clustering results!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_consistency.py file1.csv file2.csv [file3.csv ...]")
        sys.exit(1)

    compare_files(sys.argv[1:])