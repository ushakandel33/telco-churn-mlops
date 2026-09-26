"""Train, compare, and register three IBM Telco churn models."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
FEATURES = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges']
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f'Download IBM Telco churn CSV to {DATA}')
    df = pd.read_csv(DATA)
    missing = set(FEATURES + ['Churn']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df
def pipeline(estimator):
    pre = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]),
         [c for c in FEATURES if c not in NUMERIC])])
    return Pipeline([('preprocess', pre), ('classifier', estimator)])
def main():
    df = load_data()
    x, y = df[FEATURES], df['Churn'].eq('Yes').astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y)
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('telco-churn-week17')
    candidates = [
        ('logistic_C_0.1', LogisticRegression(C=0.1, max_iter=1500, class_weight='balanced')),
        ('logistic_C_1', LogisticRegression(C=1, max_iter=1500, class_weight='balanced')),
        ('forest_depth_8', RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   class_weight='balanced', random_state=42))]
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    rows = []
    for name, estimator in candidates:
        model = pipeline(estimator)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({'model': estimator.__class__.__name__, **estimator.get_params(),
                               'split_seed': 42, 'test_fraction': 0.25})
            model.fit(x_train, y_train)
            pred, prob = model.predict(x_test), model.predict_proba(x_test)[:, 1]
            scores = dict(accuracy=accuracy_score(y_test, pred),
                          precision=precision_score(y_test, pred, zero_division=0),
                          recall=recall_score(y_test, pred, zero_division=0),
                          f1=f1_score(y_test, pred, zero_division=0),
                          roc_auc=roc_auc_score(y_test, prob))
            mlflow.log_metrics(scores)
            mlflow.sklearn.log_model(model, artifact_path='model', input_example=x_test.head(2))
            for kind in ('confusion', 'roc'):
                fig, ax = plt.subplots()
                if kind == 'confusion':
                    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
                else:
                    RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
                path = out / f'{name}_{kind}.png'
                fig.savefig(path, bbox_inches='tight'); plt.close(fig)
                mlflow.log_artifact(str(path))
            rows.append({'run': name, 'run_id': run.info.run_id, **scores})
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    winner = max(rows, key=lambda r: (r['f1'], r['roc_auc']))
    client = MlflowClient()
    registered = mlflow.register_model(f"runs:/{winner['run_id']}/model", 'telco-churn-week17')
    client.transition_model_version_stage(registered.name, registered.version, 'Staging')
    client.transition_model_version_stage(registered.name, registered.version, 'Production')
    (out / 'registry.json').write_text(json.dumps({'winner': winner, 'name': registered.name,
        'version': registered.version, 'transitions': ['Staging', 'Production']}, indent=2))
    print('Winner:', winner, 'registry version:', registered.version)
if __name__ == '__main__':
    main()
