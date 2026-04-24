#!/usr/bin/env python3
"""
Main training entrypoint for MLflow-tracked experiments.
Trains pixel-level PCA+SVM and evaluates on pixel, line, and cube levels.
"""

import argparse
import yaml
import mlflow
import mlflow.sklearn

from data.loader import stream_dataset
from data.loader import *
from models.incremental_pca import *
from models.factory import build_model
from eval.eval_pixel import run as eval_pixel
from eval.eval_line import run as eval_line
from eval.eval_cube import run as eval_cube
from sklearn.decomposition import PCA
import numpy as np
import os
from joblib import dump, load
from models.autoencoder import *
import torch
from mlflow_utils import (
    set_mlflow_tracking,
    set_experiment_and_tags,
    log_params_flat,
    log_figures_dict,
    log_artifacts_dir,
)
from sklearn.model_selection import train_test_split
#import dump and load


def get_pca_model(ipca):
    
    pca = PCA(n_components=ipca.n_components_)

    # Copy learned attributes
    for attr in [
        'components_',
        'explained_variance_',
        'explained_variance_ratio_',
        'singular_values_',
        'mean_',
        'n_samples_seen_'
    ]:
        setattr(pca, attr, getattr(ipca, attr))
    return pca
def derive_pipeline(ipca, svm_model):
    """
    Create a sklearn Pipeline that chains IncrementalPCA and OneVsRestClassifier.
    """
    #check if it is an incremental or standard pca
    if(type(ipca)==PCA):
        pca = ipca
    else:
        pca = get_pca_model(ipca)


    pipeline = Pipeline([
        ('pca', pca),
        ('model', svm_model)
    ])
    return pipeline
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--experiment", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    return ap.parse_args()
def make_pixel_stream(cfg):
    # return a *callable* that creates a fresh generator every time
    def _factory():
        return stream_dataset(cfg["data"], mode="pixel")
    return _factory
def make_line_stream(cfg):
    # return a *callable* that creates a fresh generator every time
    def _factory():
        return stream_dataset(cfg["data"], mode="line")
    return _factory
def average_metric_dicts(dict_list):
    """
    Average metrics across folds.
    - For float metrics → compute mean
    - For *_support → compute sum
    """
    if not dict_list:
        return {}

    # collect all keys
    keys = dict_list[0].keys()
    out = {}

    for key in keys:
        values = [d[key] for d in dict_list]

        if key.endswith("_support"):
            # support = sum
            out[key] = int(np.sum(values))
        else:
            # numeric metric = average
            out[key] = float(np.mean(values))

    return out
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    #print(cfg)
    #print(cfg['model']['pca_svm']['kernel'])
    set_mlflow_tracking(cfg.get("mlflow", {}))
    mlflow.set_experiment("smartex_dev/test_run")
    #experiment_name = args.experiment or cfg["mlflow"]["experiment_name"]
    #set_experiment_and_tags(experiment_name, cfg.get("mlflow", {}).get("tags", {}))
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config_used.yaml")
        log_params_flat(cfg)
        model = load('/home/administrator/smartex_dev/baseline/mlruns/282429705850796833/models/m-f99408c0f03240d88b762c8afa181883/artifacts/model.pkl')
        
        data_saving_dir = '/home/administrator/smartex_dev/baseline/test_set'
        ae = SpectralAE(hidden_dim=128, latent_dim = 64, num_blocks=3, dropout=0.1, noise_std=0.0)
        ae.load_state_dict(torch.load('/home/administrator/smartex_dev/baseline/data_saving_materials_polyester_cotton_acrylic_nylon_wool_pca_components_0/autoencoder_fold_testrun.pth', map_location=torch.device('cpu')))
        ae = ae.to('cuda')
        

        x_lines_test = np.load(os.path.join(data_saving_dir, "pixel_data.npy"))
        print("Loaded line X shape: ",x_lines_test.shape)
        y_lines_test = np.load(os.path.join(data_saving_dir, "pixel_y.npy"))
       
        print("Loaded line Y shape: ",y_lines_test.shape)
        
        
         
        
        X_tensor = torch.from_numpy(x_lines_test).float().cuda()
        encoded_lines = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], batch_size):
                batch = X_tensor[i:i+batch_size]
                z = ae.encode(batch)
                encoded_lines.append(z.cpu().numpy())
        x_lines_enc = np.vstack(encoded_lines)
        del X_tensor
        del encoded_lines
        print("Encoded line X shape: ",x_lines_enc.shape)
        print("ground truth line Y shape: ",y_lines_test.shape)
        results_test,thresholds= eval_pixel(model, x_lines_enc, y_lines_test, cfg['data']['class_list'],"all_pixel_level_test")
        print("Evaluating on TESTSET ",x_lines_enc.shape," lines")
        print("*****************************************************All Pixel-level TESTSET evaluation results:")
        print(results_test['metrics'])
        mlflow.log_metrics(results_test['metrics'])
        x_lines_test_pure = x_lines_enc[np.all(y_lines_test.sum(axis=1, keepdims=True)==1, axis=1)]
        y_lines_test_pure = y_lines_test[np.all(y_lines_test.sum(axis=1, keepdims=True)==1, axis=1)]
        results_test_pure,_ = eval_pixel(model, x_lines_test_pure, y_lines_test_pure, cfg['data']['class_list'],"pure_pixel_level_test")
        print("Evaluating on TESTSET ",x_lines_test_pure.shape," pure lines")
        print("*****************************************************Pure Pixel-level TESTSET evaluation results:")
        print(results_test_pure['metrics'])
        mlflow.log_metrics(results_test_pure['metrics'])
        x_lines_test_mixed = x_lines_enc[np.all(y_lines_test.sum(axis=1, keepdims=True)>1, axis=1)]
        y_lines_test_mixed = y_lines_test[np.all(y_lines_test.sum(axis=1, keepdims=True)>1, axis=1)]
        results_test_mixed,_ = eval_pixel(model, x_lines_test_mixed, y_lines_test_mixed, cfg['data']['class_list'],"mixed_pixel_level_test")
        print("Evaluating on TESTSET ",x_lines_test_mixed.shape," mixed lines")
        print("*****************************************************Only mixed Pixel-level TESTSET evaluation results:")
        print(results_test_mixed['metrics'])
        mlflow.log_metrics(results_test_mixed['metrics'])
        
        #mlflow.sklearn.log_model(model, "pipeline")
        #mlflow.sklearn.log_model(model, "mlp_model")
        


if __name__ == "__main__":
    if mlflow.active_run() is not None:
        mlflow.end_run()
    main()