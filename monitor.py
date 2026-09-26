"""Generate reproducible synthetic feature and target drift reports."""
import json
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from train import ROOT, load_data, FEATURES

def main():
    df = load_data()[FEATURES + ['Churn']]
    reference = df.sample(frac=.7, random_state=17)
    current = df.drop(reference.index).copy()
    rng = np.random.default_rng(17)
    current['MonthlyCharges'] += rng.normal(35, 5, len(current))
    current['Contract'] = rng.choice(['Month-to-month', 'One year', 'Two year'],
                                      len(current), p=[.9, .07, .03])
    flip = rng.choice(current.index.to_numpy(), size=int(len(current) * .12), replace=False)
    current.loc[flip, 'Churn'] = current.loc[flip, 'Churn'].map({'Yes': 'No', 'No': 'Yes'})
    custom = {'mean_monthly_charges_delta': float(current.MonthlyCharges.mean() - reference.MonthlyCharges.mean()),
              'reference_churn_rate': float(reference.Churn.eq('Yes').mean()),
              'current_churn_rate': float(current.Churn.eq('Yes').mean()),
              'reference_month_to_month_churn_rate': float(reference.loc[reference.Contract.eq('Month-to-month'), 'Churn'].eq('Yes').mean()),
              'current_month_to_month_churn_rate': float(current.loc[current.Contract.eq('Month-to-month'), 'Churn'].eq('Yes').mean())}
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    report.save_html(str(out / 'drift.html'))
    (out / 'custom_metrics.json').write_text(json.dumps(custom, indent=2))
    (out / 'drift.json').write_text(json.dumps(report.as_dict(), indent=2, default=str))
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('telco-churn-week17')
    with mlflow.start_run(run_name='synthetic-drift-monitor'):
        mlflow.log_metric('mean_monthly_charges_delta', custom['mean_monthly_charges_delta'])
        mlflow.log_metric('churn_rate_delta', custom['current_churn_rate'] - custom['reference_churn_rate'])
        for name in ('drift.html', 'drift.json', 'custom_metrics.json'):
            mlflow.log_artifact(str(out / name))
    print(json.dumps(custom, indent=2))
if __name__ == '__main__':
    main()
