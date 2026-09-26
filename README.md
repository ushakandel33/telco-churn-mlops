# Telco Churn Prediction · MLOps Track A


An end-to-end churn prediction project using the IBM Telco Customer Churn dataset. It compares models with MLflow, registers the selected model, serves predictions through FastAPI, and produces an Evidently drift report.

## What is included

- **Training:** preprocessing and comparison of three model configurations
- **Experiment tracking:** metrics and model artifacts logged to MLflow
- **Model registry:** selected model registered and transitioned through Staging to Production
- **Serving:** FastAPI endpoints at `/health` and `/predict`
- **Monitoring:** Evidently HTML report and custom drift metrics

## Setup and run

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Download the IBM Telco Customer Churn CSV and save it as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is excluded from Git; the code expects that exact filename.

```bash
uv sync --locked
uv run python train.py
uv run python monitor.py
```

View the generated reports:

```bash
open results/drift.html
uv run mlflow ui --backend-store-uri ./mlruns
```

MLflow opens at `http://127.0.0.1:5000`.

To serve predictions, run:

```bash
uv run uvicorn serve:app --port 8000
```

Open the API documentation at `http://127.0.0.1:8000/docs`. A prediction request must contain a `features` object with the customer's input fields.

## Model comparison

| Model configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression, C=0.1 | 0.7535 | 0.5232 | **0.7966** | 0.6316 | 0.8453 |
| Logistic regression, C=1 | 0.7501 | 0.5188 | **0.7966** | 0.6284 | **0.8461** |
| Random forest, max depth 8 | **0.7558** | **0.5264** | 0.7901 | **0.6318** | 0.8447 |

The selection rule uses F1, with ROC-AUC as a tie-breaker. The random forest won by a **small F1 margin** and was registered as `telco-churn-week17`, version 1, with Staging and Production transitions. Logistic regression with C=1 achieved the best ROC-AUC. See [`results/comparison.csv`](results/comparison.csv) and [`results/registry.json`](results/registry.json) for the exported records.

## Drift monitoring

The monitoring script creates a reference/current split and deliberately changes monthly charges, contract distribution, and some churn labels to demonstrate monitoring. The [Evidently HTML report](results/drift.html) flagged **`MonthlyCharges` and `Contract`** as drifted.

| Custom measure | Observed value |
| --- | ---: |
| Mean monthly charges change | +33.71 |
| Reference churn rate | 26.63% |
| Current churn rate | 32.42% |

Although the measured churn rate increased, Evidently did not flag `Churn` as drifted. A rate change alone does not guarantee that its statistical drift test crosses the detection threshold. These changes were deliberately simulated and should not be interpreted as real production drift. See [`results/custom_metrics.json`](results/custom_metrics.json) for the complete custom measures.

## Notes

The same 25% holdout is used for model selection and reported performance. It is therefore a validation set, not an untouched final test set. The local MLflow database and the dataset are excluded from Git; exported comparisons, registry details, figures, and the HTML report are included so results can be reviewed without running the project.

---

**Submitted by:** Ushakiran Kandel
