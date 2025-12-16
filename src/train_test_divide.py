import pickle
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

Y = []
with open("src/data/tcga/tcga_patient_to_cancer_type.csv", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        Y.append(row[1])
Y = Y[1:]

with open("src/data/tcga/tcga_titan_embeddings_reports.pkl", "rb") as f:
    Xdata = pickle.load(f)

X = []

patients = list(Xdata.values())
for patient in patients:
    for emb in patient['embeddings']:
        X.append(emb)
 
X = X[:(len(Y))]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

train_dict = {
    "description": "X = embeddings, Y = cancer type label",
    "X": X_train, 
    "Y": Y_train}

test_dict  = {
    "description": "X = embeddings, Y = cancer type label",
    "X": X_test, 
    "Y": Y_test}

with open("src/data/tcga/traindata.pkl", "wb") as f:
    pickle.dump(train_dict, f)

with open("src/data/tcga/testdata.pkl", "wb") as f:
    pickle.dump(test_dict, f)