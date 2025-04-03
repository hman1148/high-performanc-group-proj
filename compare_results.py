import sys

def read_results(file_path):
    """Reads clustering results from a file, separating points and centroids."""
    points = {}  # {point_repr: cluster_id}
    centroids = []  # List of centroid feature vectors

    with open(file_path, "r") as file:
        section = None  # Track if reading Points or Centroids
        for line in file:
            line = line.strip()
            if line.startswith("# Points"):
                section = "points"
                continue
            elif line.startswith("# Centroids"):
                section = "centroids"
                continue

            if section == "points":
                features, cluster_id = line.rsplit("| Cluster: ", 1)
                points[features.strip()] = int(cluster_id)
            elif section == "centroids":
                features = line.rsplit("| Centroid", 1)[0].strip()
                centroids.append(features)

    return points, centroids

def compare_files(files):
    """Compares multiple clustering results for consistency."""
    if len(files) < 2:
        print("Need at least two files to compare!")
        return

    # Read reference file
    ref_points, ref_centroids = read_results(files[0])

    for file in files[1:]:
        curr_points, curr_centroids = read_results(file)

        # Compare points-to-cluster assignments
        if ref_points != curr_points:
            print(f"Mismatch in point cluster assignments between {files[0]} and {file}")
            return

        # Compare centroids (order matters)
        if ref_centroids != curr_centroids:
            print(f"Mismatch in centroids between {files[0]} and {file}")
            return

    print("✅ All implementations produce identical clustering results!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_consistency.py file1.txt file2.txt [file3.txt ...]")
        sys.exit(1)

    compare_files(sys.argv[1:])