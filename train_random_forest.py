import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

DATA_PATH = 'Training.csv'  # old dataset will be replaced; keeping variable for now
dataset = pd.read_csv(DATA_PATH)

X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
Y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

rng = np.random.default_rng(123)

X_train_noisy = X_train.copy()
ones_idx = np.argwhere(X_train_noisy.values == 1)
if len(ones_idx) > 0:
    n_drop = max(1, int(0.15 * len(ones_idx)))
    drop_rows = rng.choice(len(ones_idx), size=n_drop, replace=False)
    for i in drop_rows:
        r, c = ones_idx[i]
        X_train_noisy.iat[r, c] = 0

NOISE_PCT = 0.03
if NOISE_PCT > 0:
    n_flip = max(1, int(NOISE_PCT * len(y_train)))
    flip_idx = rng.choice(len(y_train), size=n_flip, replace=False)
    y_train_noisy = y_train.copy()
    for idx in flip_idx:
        current = y_train_noisy[idx]
        choices = [c for c in np.unique(Y) if c != current]
        y_train_noisy[idx] = rng.choice(choices)
else:
    y_train_noisy = y_train

rf = RandomForestClassifier(
    n_estimators=180,
    max_depth=12,
    max_features=0.45,
    min_samples_leaf=3,
    min_samples_split=8,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=123
)

rf.fit(X_train_noisy, y_train_noisy)

pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred, target_names=le.classes_, digits=4)
from sklearn.metrics import precision_recall_fscore_support
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, pred, average='macro')
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, pred, average='weighted')

print('Model: RandomForestClassifier')
print('Accuracy:', acc)
print('Precision (macro):', prec_macro)
print('Recall (macro):', rec_macro)
print('F1-score (macro):', f1_macro)
print('Precision (weighted):', prec_weighted)
print('Recall (weighted):', rec_weighted)
print('F1-score (weighted):', f1_weighted)
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print('OOB Score (approx gen. accuracy):', getattr(rf, 'oob_score_', None))

cv_f1 = cross_val_score(rf, X, Y, cv=5, scoring='f1_macro', n_jobs=-1)
cv_acc = cross_val_score(rf, X, Y, cv=5, scoring='accuracy', n_jobs=-1)
print('5-fold CV F1-macro: mean=', cv_f1.mean(), 'std=', cv_f1.std())
print('5-fold CV Accuracy: mean=', cv_acc.mean(), 'std=', cv_acc.std())

with open('svc.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Saved model to 'svc.pkl' and label encoder to 'label_encoder.pkl'.")
