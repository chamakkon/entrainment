import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# データセットの読み込み
f = open("entrainment_score.json", "r")
entrainment_score = json.load(f)
features= entrainment_score["0"].keys()
metrics = entrainment_score["0"]["F0_MAX"].keys()
X = np.empty((0,58))
i = 0
for feature in features:
    for metric in metrics:
        try:
            data = [metric_dict[metric] for metric_dict in [feature_dict[feature] for feature_dict in entrainment_score.values()]]
        except TypeError:
            print(f"skip {feature} {metric}")
            continue
        X = np.vstack((X, np.array(data)))
X = np.delete(X, 6, axis=1)
X = np.delete(X, 48, axis=1)
X = X.T
print(X.shape)
f = open("../labeled_corpus_with_time.json", "r")
corpus = json.load(f)
print(corpus.keys())
y = [value["y_score"] for value in corpus.values()]
print(len(y))
# トレーニングデータとテストデータに分割 (80%: トレーニング, 20%: テスト)

new_X = np.delete(X, i, axis=1)
results=[]
for a in [1,13,24,23,41,25,5,73, 49,73]:
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=a)

    # 特徴量のスケーリング（標準化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ロジスティック回帰モデルの作成とトレーニング
    model = SVC(kernel='linear', random_state=42)  # 線形カーネルを使用
    model.fit(X_train, y_train)

    # テストデータで予測
    y_pred = model.predict(X_test)

    # 結果の評価
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    coefficients = model.coef_
    results.append(accuracy)
    #print("モデルの係数:")
    #print(coefficients)

print(str(i)+ " : " +str(sum(results)/10))