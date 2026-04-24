import argparse
import yaml
from data.loader import stream_dataset
from data.loader import *
import numpy as np
import os
from joblib import dump, load
import h5py
import pickle
from sklearn.model_selection import train_test_split
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_ae.yaml")
    return ap.parse_args()
def make_pixel_stream(cfg):
    # return a *callable* that creates a fresh generator every time
    def _factory():
        return stream_dataset(cfg["data"], mode="line")
    return _factory
def make_line_stream(cfg):
    # return a *callable* that creates a fresh generator every time
    def _factory():
        return stream_dataset(cfg["data"], mode="line")
    return _factory
def make_test_stream(cfg):
    # return a *callable* that creates a fresh generator every time
    def _factory():
        return stream_dataset(cfg["data"], mode="line",split='test')
    return _factory
def append_X_libsvm(path, X_batch):
    B, D = X_batch.shape
    with open(path, "a") as f:
        for i in range(B):
            # Use dummy label 0 — XGBoost ignores labels when you load later
            parts = ["0"]
            for j, val in enumerate(X_batch[i]):
                parts.append(f"{j+1}:{val}")
            f.write(" ".join(parts) + "\n")

def append_Y_hdf5(path, Y_batch, start_idx):
    with h5py.File(path, "a") as f:
        f["Y"][start_idx:start_idx + Y_batch.shape[0]] = Y_batch
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_saving_dir = f'data_saving_materials_{"_".join(cfg["data"]["class_list"])}_pca_components_{cfg["dimensionality_reduction"]["pca_components"]}'
    print("Data saving dir: ",data_saving_dir)
    os.makedirs(data_saving_dir,exist_ok=True)
        
    # train_stream = make_pixel_stream(cfg)
    # tot_y = [] 
    # tot_x = []
    # for X_batch, Y_batch in train_stream():
    #     tot_x.append(X_batch)
    #     tot_y.append(Y_batch)
    # tot_x = np.concatenate(tot_x, axis=0)
    # tot_y = np.concatenate(tot_y, axis=0)
    # print("Total X shape: ",tot_x.shape)
    # np.save(os.path.join(data_saving_dir,"X_pixel_processed.npy"), tot_x)
    # np.save(os.path.join(data_saving_dir,"y_pixel_processed.npy"), tot_y)
    # del tot_x
    # del tot_y
    line_y = [] 
    line_x = []
    file_ids = []
    percentage_dicts = []
    line_stream = make_line_stream(cfg)
    i=100
    for X_batch, Y_batch,file_id,percentage_dict in line_stream():
        line_x.append(X_batch)
        #print(X_batch.shape)
        line_y.append(Y_batch)
        file_ids.append(file_id)
        percentage_dicts.append(percentage_dict)

    line_x = np.array(line_x)
    line_y = np.vstack(line_y)
    file_ids = np.array(file_ids)
    print("Total X shape: ",line_x.shape)
    print("Total Y shape: ",line_y.shape)
    print('total files: ',len(list(set(file_ids))))
    np.save(os.path.join(data_saving_dir,"X_line.npy"), line_x)
    np.save(os.path.join(data_saving_dir,"y_line.npy"), line_y)
    np.save(os.path.join(data_saving_dir,"fileid_line.npy"), file_ids)
    pickle.dump(percentage_dicts, open(os.path.join(data_saving_dir,"percentage_dicts.pkl"), "wb"))
    
    
if __name__ == "__main__":
    main()