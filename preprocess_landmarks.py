import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import argparse

def normalize_skeleton(row):
    # center skeleton using mid-hip (landmark 23 & 24)
    mid_hip_x = (row["x23"] + row["x24"]) / 2
    mid_hip_y = (row["y23"] + row["y24"]) / 2
    mid_hip_z = (row["z23"] + row["z24"]) / 2

    xs, ys, zs = [], [], []
    for i in range(33):
        xs.append(row[f"x{i}"] - mid_hip_x)
        ys.append(row[f"y{i}"] - mid_hip_y)
        zs.append(row[f"z{i}"] - mid_hip_z)

    return xs + ys + zs


def compute_angles(row):
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    # indices for angle triplets (common joints)
    angle_triplets = [
        (11, 13, 15),  # left arm
        (12, 14, 16),  # right arm
        (23, 25, 27),  # left leg
        (24, 26, 28),  # right leg
        (11, 23, 25),  # left torso/leg
        (12, 24, 26)   # right torso/leg
    ]

    angles = []
    for a, b, c in angle_triplets:
        A = np.array([row[f"x{a}"], row[f"y{a}"], row[f"z{a}"]])
        B = np.array([row[f"x{b}"], row[f"y{b}"], row[f"z{b}"]])
        C = np.array([row[f"x{c}"], row[f"y{c}"], row[f"z{c}"]])
        angles.append(angle(A, B, C))

    return angles


def compute_pairwise_distances(row):
    distances = []
    important_pairs = [
        (11, 12),   # shoulders
        (23, 24),   # hips
        (15, 16),   # wrists
        (27, 28),   # ankles
        (11, 23),   # left shoulder to hip
        (12, 24)    # right shoulder to hip
    ]

    for a, b in important_pairs:
        ax, ay, az = row[f"x{a}"], row[f"y{a}"], row[f"z{a}"]
        bx, by, bz = row[f"x{b}"], row[f"y{b}"], row[f"z{b}"]
        dist = np.linalg.norm([ax - bx, ay - by, az - bz])
        distances.append(dist)

    return distances


def process_file(df):
    normalized = df.apply(normalize_skeleton, axis=1, result_type="expand")
    angles = df.apply(compute_angles, axis=1, result_type="expand")
    distances = df.apply(compute_pairwise_distances, axis=1, result_type="expand")

    features = pd.concat([normalized, angles, distances], axis=1)

    pca = PCA(n_components=0.95)
    compressed = pca.fit_transform(features)

    result = pd.DataFrame(compressed)
    result["label"] = df["label"].values
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = process_file(df)
    result.to_csv(args.output, index=False)

    print("Saved:", args.output)
