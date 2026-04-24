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

from models.factory import build_model
from models.autoencoder_contrastive import *
from eval.eval_pixel import run as eval_pixel



from splits import *
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
    experiment_name = args.experiment or cfg["mlflow"]["experiment_name"]
    #set_experiment_and_tags(experiment_name, cfg.get("mlflow", {}).get("tags", {}))
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config_used.yaml")
        log_params_flat(cfg)
        model = build_model(cfg["model"])
        data_saving_dir = f'data_saving_materials_{"_".join(cfg["data"]["class_list"])}_pca_components_{cfg["dimensionality_reduction"]["pca_components"]}'
        
        if(os.path.exists(data_saving_dir) and os.path.isdir(data_saving_dir) and cfg['data']['load_processed']):
            print("Loading processed data from ",data_saving_dir)
            X_all = np.load(os.path.join(data_saving_dir,"X_pixel_processed.npy"))
            print("Loaded X shape: ",X_all.shape)
            y_all = np.load(os.path.join(data_saving_dir,"y_pixel_processed.npy"))
            x_lines_val = np.load(os.path.join(data_saving_dir,"X_line_processed.npy"))
            print("Loaded line X shape: ",x_lines_val.shape)
            y_lines_val = np.load(os.path.join(data_saving_dir,"y_line_processed.npy"))
            if(cfg['dimensionality_reduction'].get('use_incremental_pca', False)==True):
                ipca = load(os.path.join(data_saving_dir,"pca.pkl"))
            
        X_lines = np.load(os.path.join(data_saving_dir,"X_line.npy"))
        y_lines = np.load(os.path.join(data_saving_dir,"y_line.npy"))
        files_ids = np.load(os.path.join(data_saving_dir,"fileid_line.npy"))
        
        all_line_totals = y_lines.sum(axis=0)
        splits = stratified_group_shuffle_split(y_lines, files_ids, test_frac=0.8, n_repeats=2, seed=72)
        mixed_results_folds = []
        pure_results_folds = []
        all_results_folds = []
        D = y_lines.shape[1]
        material_names = cfg['data']['class_list']
        for rep, (train_idx, val_idx) in enumerate(splits):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            val_idx   = np.asarray(val_idx, dtype=np.int64)
            train_line = y_lines[train_idx].sum(axis=0)
            val_line   = y_lines[val_idx].sum(axis=0)

            # -------- file-level --------
            tr_nf, tr_file = file_level_presence(y_lines, files_ids, train_idx,D)
            va_nf, va_file = file_level_presence(y_lines, files_ids, val_idx,D)

            print(f"Fold {rep}: {len(train_idx)} train samples, {len(val_idx)} val samples")
            print(f"\n=== Repeat {rep:02d} ===")
            print(f"TRAIN: n_files={tr_nf:3d}, n_lines={len(train_idx):5d}")
            for i, m in enumerate(material_names):
                print(f"  {m:10s}  lines={train_line[i]:4d}  "
                    f"files={tr_file[i]:3d}  "
                    f"line_frac={(train_line[i]/(all_line_totals[i]+1e-9)):.3f}")

            print(f"VAL  : n_files={va_nf:3d}, n_lines={len(val_idx):5d}")
            for i, m in enumerate(material_names):
                print(f"  {m:10s}  lines={val_line[i]:4d}  "
                    f"files={va_file[i]:3d}  "
                    f"line_frac={(val_line[i]/(all_line_totals[i]+1e-9)):.3f}")
            X_train, y_train = X_lines[train_idx], y_lines[train_idx]
            X_val,   y_val   = X_lines[val_idx],   y_lines[val_idx]
            X_train = X_train.reshape(-1, X_train.shape[-1])  # (num_pixels, B)
            X_val = X_val.reshape(-1, X_val.shape[-1])      # (num_pixels, B)
            y_train = y_train[np.repeat(np.arange(y_train.shape[0]), X_lines.shape[1]), :]  # (num_pixels, n_classes)
            y_val = y_val[np.repeat(np.arange(y_val.shape[0]), X_lines.shape[1]), :]      # (num_pixels, n_classes)
            print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
            ae_save = os.path.join(data_saving_dir,f"autoencoder_fold{rep}.pth")
            #ae = train_engine(X_train,X_val,cfg['autoencoder'],save_path=ae_save)
            ae = train_engine(X_train,X_val,cfg['autoencoder'],save_path=ae_save)
            ae.eval()
            lines_train = X_lines[train_idx]
            lines_val = X_lines[val_idx]
            y_train = y_lines[train_idx]
            y_val = y_lines[val_idx]
            #encode the pixel data
            with torch.no_grad():
                X_tensor = torch.from_numpy(lines_train).float().cuda()
                encoded_pixels = []
                batch_size = 512
                for i in range(0, X_tensor.shape[0], batch_size):
                    batch = X_tensor[i:i+batch_size]
                    z = ae.encode(batch)
                    encoded_pixels.append(z.cpu().numpy())
                X_train_encoded = np.vstack(encoded_pixels)
                del X_tensor
                del encoded_pixels
                #encode the line data
                X_tensor = torch.from_numpy(lines_val).float().cuda()
                encoded_lines = []
                batch_size = 512
                for i in range(0, X_tensor.shape[0], batch_size):
                    batch = X_tensor[i:i+batch_size]
                    z = ae.encode(batch)
                    encoded_lines.append(z.cpu().numpy())
                X_val_encoded = np.vstack(encoded_lines)
                del X_tensor
                del encoded_lines
        

                print("Encoded train shape: ",X_train_encoded.shape)
                print("Encoded val shape: ",X_val_encoded.shape)
                print('Y shape: ',y_train.shape)
                print("Y val shape: ",y_val.shape)
                latent_dim = cfg['autoencoder']['latent_dim']
                model.fit(X_train_encoded.reshape(-1,latent_dim), y_train[np.repeat(np.arange(y_train.shape[0]), X_lines.shape[1]), :])

                results,thresholds = eval_pixel(model, X_val_encoded.reshape(-1,latent_dim), y_val[np.repeat(np.arange(y_val.shape[0]), X_lines.shape[1]), :], cfg['data']['class_list'],"All pixel_level_val")
                print("Evaluating on ",X_val.shape," pixels")
                print("*****************************************************Pixel-level evaluation results:")
                print(results['metrics'])
                mlflow.log_metrics(results['metrics'])
                #pure pixels
                y_val_full = y_val[np.repeat(np.arange(y_val.shape[0]), X_lines.shape[1]), :]
                X_val_full = X_val_encoded.reshape(-1,latent_dim)
                x_pure_pixels =X_val_full[np.all(y_val_full.sum(axis=1, keepdims=True)==1, axis=1)]
                y_pure_pixels = y_val_full[np.all(y_val_full.sum(axis=1, keepdims=True)==1, axis=1)]
                
                print("Evaluating on ",x_pure_pixels.shape[0]," pure pixels, y shape is ",y_pure_pixels.shape)
                results_pure_pixel,thresholds = eval_pixel(model, x_pure_pixels, y_pure_pixels, cfg['data']['class_list'],"Pure pixel_level_val",thresholds=thresholds)
                print("Evaluating on ",x_pure_pixels.shape[0]," pure pixels")
                print("*****************************************************Pure Pixel-level evaluation results:")
                print(results_pure_pixel['metrics'])
                mlflow.log_metrics(results_pure_pixel['metrics'])
                #mixed pixels
                x_mixed_pixels =X_val_full[np.any(y_val_full.sum(axis=1, keepdims=True)>1, axis=1)]
                y_mixed_pixels = y_val_full[np.any(y_val_full.sum(axis=1, keepdims=True)>1, axis=1)]
                results_mixed_pixel,thresholds = eval_pixel(model, x_mixed_pixels, y_mixed_pixels, cfg['data']['class_list']  ,"Mixed pixel_level_val",thresholds=thresholds)
                print("Evaluating on ",x_mixed_pixels.shape[0]," mixed pixels")
                print("*****************************************************Mixed Pixel-level evaluation results:")
                print(results_mixed_pixel['metrics'])
                mlflow.log_metrics(results_mixed_pixel['metrics'])
                
        print("Averaged pure line-level results across folds:")
        pure_results_avg = average_metric_dicts(pure_results_folds)
        print(pure_results_avg)
        mlflow.log_metrics(pure_results_avg)
        mixed_results_avg = average_metric_dicts(mixed_results_folds)
        print("Averaged mixed line-level results across folds:")
        print(mixed_results_avg)
        mlflow.log_metrics(mixed_results_avg)
        all_results_avg = average_metric_dicts(all_results_folds)
        print("Averaged all line-level results across folds:")
        print(all_results_avg)
        mlflow.log_metrics(all_results_avg)
        if(cfg['data'].get('testset', False)==True):
            
            print("\n" + "="*80)
            print("FINAL TEST SET EVALUATION - Training on ALL available non-test data")
            print("="*80)
            
            # Prepare ALL data for final autoencoder training
            # Use 80/20 split of ALL data for autoencoder's internal validation
            X_all_pixels = X_lines.reshape(-1, X_lines.shape[-1])  # (num_pixels, B)
            y_all_pixels = y_lines[np.repeat(np.arange(y_lines.shape[0]), X_lines.shape[1]), :]  # (num_pixels, n_classes)
            
            # Split ALL data for autoencoder training (80/20 for early stopping)
            X_ae_train, X_ae_val, y_ae_train, y_ae_val = train_test_split(
                X_all_pixels, y_all_pixels, test_size=0.2, random_state=42
            )
            
            print(f"Training final autoencoder on ALL data: train={X_ae_train.shape}, val={X_ae_val.shape}")
            ae_save = os.path.join(data_saving_dir, f"autoencoder_final.pth")
            ae = train_engine(X_ae_train, X_ae_val, cfg['autoencoder'], 
                                         save_path=ae_save)
            ae.eval()
            
            # Encode ALL training data with the final autoencoder
            print(f"Encoding ALL {X_lines.shape[0]} lines for final classifier training")
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_lines).float().cuda()
                encoded_lines = []
                batch_size = 512
                for i in range(0, X_tensor.shape[0], batch_size):
                    batch = X_tensor[i:i+batch_size]
                    z = ae.encode(batch)
                    encoded_lines.append(z.cpu().numpy())
                X_all_encoded = np.vstack(encoded_lines)
                del X_tensor
                del encoded_lines
            
            # Train final classifier on ALL encoded data
            mlflow.pytorch.log_model(ae, "autoencoder")
            latent_dim = cfg['autoencoder']['latent_dim']
            print(f"Training final classifier on ALL {X_all_encoded.shape[0]} encoded lines")
            model.fit(X_all_encoded.reshape(-1, latent_dim), y_all_pixels)
            

            
            
        
        #mlflow.sklearn.log_model(pipe, "pipeline")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    if mlflow.active_run() is not None:
        mlflow.end_run()
    main()