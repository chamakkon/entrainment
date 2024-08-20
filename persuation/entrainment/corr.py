import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

f = open("entrainment_score.json", "r")
entrainment_score = json.load(f)
features= entrainment_score["0"].keys()
metrics = entrainment_score["0"]["F0_MAX"].keys()
X = np.empty((0,58))
i = 0
data_dict = {}
for feature in features:
    proximity = []
    convergence = []
    synchrony = []
    for metric in metrics:
        for feature_dict in entrainment_score.values():
            try:
                find_nan = np.isnan(feature_dict[feature])
            except TypeError:
                find_nan = False
            if find_nan==True:
                proximity.append(np.nan)
                convergence.append(np.nan)
                synchrony.append(np.nan)
            else:
                for metric_dict in feature_dict[feature]:
                    if feature == "proximity":
                        proximity.append(metric_dict[metric])
                    if feature == "convergence":
                        convergence.append(metric_dict[metric])
                    if feature == "synchrony":
                        synchrony.append(metric_dict[metric])
    data_dict[f"{feature}:p"] = proximity
    data_dict[f"{feature}:c"] = convergence
    data_dict[f"{feature}:s"] = synchrony

print(data_dict)