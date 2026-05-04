import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# LOAD DATA
# ----------------------------
def load_files(folder, label):
    data, labels = [], []
    for file in os.listdir(folder):
        if file.endswith(".php"):
            try:
                with open(os.path.join(folder, file), "r", errors="ignore") as f:
                    data.append(f.read())
                    labels.append(label)
            except:
                continue
    return data, labels

malicious_data, malicious_labels = load_files("/home/dr_client/Desktop/EUE/project/web-shell-detection/php-webshells-master/webshell-project/dataset/malicious", 1)
benign_data, benign_labels = load_files("/home/dr_client/Desktop/EUE/project/web-shell-detection/php-webshells-master/webshell-project/dataset/benign", 0)

X = malicious_data + benign_data
y = malicious_labels + benign_labels

print("Total samples:", len(X))

# ----------------------------
# VECTORIZE
# ----------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# ----------------------------
# TRAIN
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ----------------------------
# CONFUSION MATRIX IMAGE
# ----------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# ----------------------------
# HTML REPORT GENERATOR
# ----------------------------
html = f"""
<html>
<head>
    <title>Web Shell Detection Report</title>
    <style>
        body {{
            font-family: Arial;
            margin: 40px;
            background-color: #f4f4f4;
        }}
        .box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0px 0px 10px #ccc;
        }}
        h1 {{
            color: #222;
        }}
    </style>
</head>

<body>

<h1>Web Shell Detection Report</h1>

<div class="box">
    <h2>Dataset Info</h2>
    <p>Total Samples: {len(X)}</p>
    <p>Malicious Samples: {len(malicious_data)}</p>
    <p>Benign Samples: {len(benign_data)}</p>
</div>

<div class="box">
    <h2>Model Performance</h2>
    <p>Accuracy: {report['accuracy']:.2f}</p>
    <p>Precision (Malicious): {report['1']['precision']:.2f}</p>
    <p>Recall (Malicious): {report['1']['recall']:.2f}</p>
    <p>F1 Score (Malicious): {report['1']['f1-score']:.2f}</p>
</div>

<div class="box">
    <h2>Confusion Matrix</h2>
    <img src="confusion_matrix.png" width="400">
</div>
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(html)

print("REPORT GENERATED: report.html")
