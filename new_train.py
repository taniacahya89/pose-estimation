import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("dataset/features_all.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(clf, "model_pose_classifier.pkl")
print("Saved model_pose_classifier.pkl")
