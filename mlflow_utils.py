import mlflow
import os
import matplotlib.figure as mpl_fig

def set_mlflow_tracking(cfg):
    if cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(cfg["tracking_uri"])
    if cfg.get("registry_uri"):
        mlflow.set_registry_uri(cfg["registry_uri"])

def set_experiment_and_tags(name, tags):
    mlflow.set_experiment(name)
    if tags:
        mlflow.set_tags(tags)

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_params_flat(cfg):
    flat = flatten_dict(cfg)
    for k, v in flat.items():
        if not isinstance(v, (dict, list)):
            mlflow.log_param(k, v)

def log_figures_dict(figures):
    if not figures:
        return
    for name, fig in figures.items():
        if isinstance(fig, mpl_fig.Figure):
            mlflow.log_figure(fig, f"{name}.png")

def log_artifacts_dir(data, artifact_path):
    if not data:
        return
    if isinstance(data, str) and os.path.isdir(data):
        mlflow.log_artifacts(data, artifact_path=artifact_path)
