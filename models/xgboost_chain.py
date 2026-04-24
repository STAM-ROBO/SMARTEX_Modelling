from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain

def build_model(cfg,):
    """
    Fit OneVsRest SVM classifier on data X, y.
    Returns a fitted OneVsRestClassifier instance.
    """
    cfg_model = cfg.get("xgb_chain", {}  )
    n_estimators = cfg_model.get('n_estimators', 100)
    max_depth = cfg_model.get('max_depth', 4)
    learning_rate = cfg_model.get('learning_rate', 0.05)
    #model = ClassifierChain(XGBClassifier( n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,), order='random', random_state=0)
    model = ClassifierChain(XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",   # or 'gpu_hist' if you have a GPU
))
    
    
    return model