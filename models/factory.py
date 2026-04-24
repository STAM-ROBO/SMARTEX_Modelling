from models.lr_chain import build_model as build_lr_chain

from models.xgboost_chain import build_model as build_xgb_chain
from models.svm_chain import build_model as build_svm_chain
from models.mlp_chain import build_model as build_mlp_chain

def build_model(config):
    model_type = config["type"]


    if model_type == "lr_chain":
        return build_lr_chain(config)

    elif model_type == "xgb_chain":
        return build_xgb_chain(config)
    elif model_type == "svm_chain":
        return build_svm_chain(config)
    elif model_type == "mlp_chain":
        return build_mlp_chain(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    