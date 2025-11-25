import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

NUM_LANDMARKS = 33

def normalize_row_flat(flat):
    # flat: [x0,y0,z0,vis0, x1,y1,z1,vis1, ...] length = NUM_LANDMARKS*4
    arr = np.array(flat).reshape(NUM_LANDMARKS, 4)
    xs = arr[:,0].astype(float)
    ys = arr[:,1].astype(float)
    vis = arr[:,3].astype(float)

    # find hip center (left_hip=23, right_hip=24)
    # if indices out of range, fallback to mean
    try:
        hip = np.nanmean(arr[[23,24], :2], axis=0)
    except:
        hip = np.nanmean(arr[:, :2], axis=0)
    xs = xs - hip[0]
    ys = ys - hip[1]

    # torso scale: distance between avg shoulders and hip center
    try:
        shoulders = np.nanmean(arr[[11,12], :2], axis=0)
        torso_dist = np.linalg.norm(shoulders - hip)
        if torso_dist == 0 or np.isnan(torso_dist):
            torso_dist = 1.0
    except:
        torso_dist = 1.0

    xs = xs / torso_dist
    ys = ys / torso_dist

    feat = np.concatenate([xs, ys])  # drop z & vis for simplicity
    return feat

def load_dataset(folder='dataset'):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file CSV di folder {folder}. Jalankan collector dulu.")
    X_list, y_list = [], []
    for f in files:
        df = pd.read_csv(f)
        # expect columns: x0,y0,z0,vis0, ... , label,timestamp
        # get only the landmark columns (first NUM_LANDMARKS*4 cols)
        landmark_cols = df.columns[:NUM_LANDMARKS*4]
        labels = df['label'].values
        for idx, row in df.iterrows():
            flat = row[landmark_cols].values
            feat = normalize_row_flat(flat)
            X_list.append(feat)
            y_list.append(labels[idx])
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

def main():
    print("Memuat dataset dari folder 'dataset/' ...")
    X, y = load_dataset('dataset')
    print("Dataset shape:", X.shape, y.shape)

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split subject-wise would be better but here we'll use random split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # pilih MLP sederhana
    clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42)
    print("Melatih MLPClassifier ...")
    clf.fit(X_train_s, y_train)

    # evaluasi
    y_pred = clf.predict(X_test_s)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # simpan model + preprocessor
    ensure_dir = lambda d: os.makedirs(d, exist_ok=True)
    ensure_dir('models')
    joblib.dump({'model': clf, 'scaler': scaler, 'label_encoder': le}, 'models/model_pose.pkl')
    print("Model tersimpan di models/model_pose.pkl")

if __name__ == "__main__":
    main()
