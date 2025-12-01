#!/usr/bin/env python3
"""
End-to-end training with robust fallbacks so accuracy shows up.
- Loads dataset/trainings.csv and dataset/testing.csv
- Preprocess: coerce to numeric, fillna, ensure features are float
- Strategy:
  * If each class has >=2 samples: train RandomForest + GridSearchCV
  * Else: use KNN (k=1) on binary presence features or cosine similarity retrieval
- Metrics: accuracy, macro-F1 on test
- Artifacts saved to artifacts/
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import joblib

RANDOM_STATE = 42
ART_DIR = 'artifacts'
os.makedirs(ART_DIR, exist_ok=True)

TRAIN_P = os.path.join('dataset','trainings.csv')
TEST_P = os.path.join('dataset','testing.csv')

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding='latin1')

# 1) Load data
train = read_csv_safe(TRAIN_P)
test = read_csv_safe(TEST_P)
assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)

# 2) Detect target
target = None
for cand in ['Prognosis','prognosis','Disease','disease', train.columns[-1]]:
    if cand in train.columns:
        target = cand; break
assert target is not None, 'Could not detect target column.'

# Split features/target
X_train_df = train.drop(columns=[target]).copy()
y_train_raw = train[target].astype(str)
X_test_df = test.drop(columns=[target]).copy() if target in test.columns else test.copy()
y_test_raw = test[target].astype(str) if target in test.columns else None

# 3) Preprocess
# Coerce everything to numeric, fillna with 0
X_train = X_train_df.apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0).astype(float)
X_test = X_test_df.apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0).astype(float)

# Label encode y
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw) if y_test_raw is not None else None

# Persist encoder
joblib.dump(le, os.path.join(ART_DIR, 'label_encoder.pkl'))

# 4) Basic feature selection (remove zero-variance columns)
vt = VarianceThreshold(threshold=0.0)
X_train_sel = vt.fit_transform(X_train)
X_test_sel = vt.transform(X_test)
sel_mask = vt.get_support().tolist()
feature_names = [c for c, keep in zip(X_train.columns, sel_mask) if keep]
with open(os.path.join(ART_DIR, 'selected_features.json'), 'w') as f:
    json.dump(feature_names, f)

# 5) Decide strategy
class_counts = np.bincount(y_train)
min_count = int(class_counts.min())
print(f'classes: {len(le.classes_)}, min_count: {min_count}')

metrics = {}
model = None

if min_count >= 2:
    # RandomForest + light tuning
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {
        'n_estimators': [300, 600],
        'max_depth': [None, 20, 40],
        'max_features': ['sqrt', None]
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    try:
        grid.fit(X_train_sel, y_train)
        model = grid.best_estimator_
        print('tuned RF best params:', grid.best_params_)
    except Exception as e:
        print('grid search failed, using baseline RF:', e)
        model = rf.fit(X_train_sel, y_train)
else:
    # KNN 1-NN on binary presence features
    X_train_bin = (X_train_sel > 0).astype(int)
    X_test_bin = (X_test_sel > 0).astype(int)
    model = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    model.fit(X_train_bin, y_train)
    # For consistency, swap arrays for prediction below
    X_train_sel = X_train_bin
    X_test_sel = X_test_bin

# 6) Evaluate on test
y_pred = model.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')
print(f'accuracy: {acc:.4f}, macro-f1: {f1m:.4f}')
print('classification report (first 1200 chars):')
print(classification_report(y_test, y_pred, zero_division=0)[:1200])
metrics.update({'accuracy': float(acc), 'macro_f1': float(f1m)})

# 7) Save artifacts
joblib.dump(model, os.path.join(ART_DIR, 'model.pkl'))
with open(os.path.join(ART_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print('artifacts saved to', ART_DIR)
