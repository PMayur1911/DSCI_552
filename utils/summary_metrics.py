import pandas as pd

def get_summary_metrics(models, set_name):
    comp_metrics = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

    df = {}
    for metric in comp_metrics:
        if metric == "Model":
            vals = [x.model_name for x in models]
        elif metric == "Accuracy":
            vals = [
                getattr(x, f"{set_name}_metrics", None).accuracy
                for x in models
            ]
        elif metric == "Precision":
            vals = [
                getattr(x, f"{set_name}_metrics", None).weighted_metrics.precision
                for x in models
            ]
        elif metric == "Recall":
            vals = [
                getattr(x, f"{set_name}_metrics", None).weighted_metrics.recall
                for x in models
            ]
        elif metric == "F1-Score":
            vals = [
                getattr(x, f"{set_name}_metrics", None).weighted_metrics.f1_score
                for x in models
            ]
        elif metric == "AUC":
            vals = [
                getattr(x, f"{set_name}_metrics", None).weighted_metrics.auc
                for x in models
            ]
        else:
            vals = []

        df[metric] = vals
    return pd.DataFrame(df)