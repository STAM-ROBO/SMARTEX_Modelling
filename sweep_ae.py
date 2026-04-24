#!/usr/bin/env python3
"""
Sweep controller with ZERO MLflow context.
It just builds config variants and launches independent train runs.
"""

import itertools
import yaml
import copy
import os
import subprocess
import sys
import mlflow
def dict_product(d):
    keys = list(d.keys())
    vals = [v if isinstance(v, list) else [v] for v in d.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    with open("config_ae.yaml") as f:
        base_cfg = yaml.safe_load(f)
    mlflow.set_experiment("smartex_dev/ae_sweep_gkf")
    model_sweep = base_cfg.get("model", {}).get("sweep", {})
    print("MODEL SWEEP IS ",model_sweep)
    pca_sweep = base_cfg.get("dimensionality_reduction", {}).get("sweep", {})
    data_train_sweep  = base_cfg.get("data", {}).get("line_level", {}).get("sweep", {})
    data_eval_sweep  = base_cfg.get("data", {}).get("pixel_level", {}).get("sweep", {})
    autoencoder_sweep = base_cfg.get("autoencoder", {}).get("sweep", {})
  
    grid = list(dict_product({**model_sweep, **data_train_sweep, **data_eval_sweep,**pca_sweep,**autoencoder_sweep}))
    print(f"Launching {len(grid)} training runs...")

    for i, combo in enumerate(grid):
        print(f"\n=== Combo {i+1}/{len(grid)}: {combo} ===")

        # Apply combo into a child configuration
        child_cfg = copy.deepcopy(base_cfg)
        for k, v in combo.items():
            keys = k.split(".")
            node = child_cfg
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = v
        
        tmp_path = f".tmp_cfg_{i:03d}.yaml"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(child_cfg, f, sort_keys=False)

        # Run train.py in a fresh process with a clean environment
        env = os.environ.copy()
        env.pop("MLFLOW_RUN_ID", None)
        env.pop("MLFLOW_PARENT_RUN_ID", None)
        class_list = "_".join(child_cfg["data"]["class_list"])
        #just log the type of the model
        run_name= f"sweep_combo_{combo['data.pixel_level.filter_mode']}_modeltype_{combo['model.type']}_materials_{class_list}"
        
        #if(combo['model.type']=="lr_chain"):
        #    run_name = f"sweep_combo_{combo['data.pixel_level.filter_mode']}_PCA{combo['dimensionality_reduction.pca_components']}_LRChain_materials_{class_list}"
        #else:
        #    run_name = f"sweep_combo_{combo['data.pixel_level.filter_mode']}_PCA{combo['dimensionality_reduction.pca_components']}_SVM_C{combo['model.svm.C']}_kernel{combo['model.svm.kernel']}_materials_{class_list}"
        cmd = [sys.executable, "train_gkf_ae.py", "--config", tmp_path, "--run_name", run_name,]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, env=env, check=True)
        finally:
            # Always attempt to clean up the temp config
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print("\n✅ Sweep complete.")

if __name__ == "__main__":
    main()
