import pandas as pd
import numpy as np
import argparse
import os

SEQ_LEN = 30
STRIDE = 15

EXPECTED_COLS = 134   # 33 joints * (x,y,z,vis) + label

def normalize_landmarks(frame):
    coords = frame.reshape(-1, 2)

    hip_center = (coords[11] + coords[12]) / 2
    coords -= hip_center

    shoulder_center = (coords[5] + coords[6]) / 2
    torso_len = np.linalg.norm(shoulder_center)
    if torso_len > 0:
        coords /= torso_len

    return coords.flatten()

def clean_dataframe(df):
    # drop rows with incomplete landmark data
    df = df.dropna()

    # keep only rows with exact expected number of columns
    df = df[df.shape[1] == EXPECTED_COLS]

    df = df.reset_index(drop=True)
    return df

def build_sequences(df):
    df = clean_dataframe(df)

    if len(df) < SEQ_LEN:
        return np.array([]), np.array([])

    label = df["label"].iloc[0]
    X = df.drop(columns=["label"]).values

    xy_frames = []
    for row in X:
        if len(row) != EXPECTED_COLS - 1:
            continue
        filtered = []
        for i in range(0, len(row), 4):
            filtered.extend([row[i], row[i+1]])
        xy_frames.append(filtered)

    X = np.array(xy_frames)

    X_norm = np.array([normalize_landmarks(f) for f in X])

    sequences = []
    labels = []

    for i in range(0, len(X_norm) - SEQ_LEN, STRIDE):
        sequences.append(X_norm[i:i+SEQ_LEN])
        labels.append(label)

    return np.array(sequences), np.array(labels)

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    return build_sequences(df)

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    all_X = []
    all_y = []

    for file in os.listdir(args.input_folder):
        if file.endswith(".csv"):
            path = os.path.join(args.input_folder, file)
            X, y = process_csv(path)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

    if not all_X:
        print("No valid sequences found.")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    np.save(os.path.join(args.output_folder, "X_seq.npy"), X_all)
    np.save(os.path.join(args.output_folder, "y_seq.npy"), y_all)

    print("Done. Shapes:", X_all.shape, y_all.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()

    main(args)
