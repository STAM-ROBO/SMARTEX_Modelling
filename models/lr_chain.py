from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

def build_model(cfg,):
    """
    Fit OneVsRest SVM classifier on data X, y.
    Returns a fitted OneVsRestClassifier instance.
    """
    cfg_model = cfg.get("lr_chain", {}  )
    model = ClassifierChain(LogisticRegression(solver='lbfgs', random_state=0), order='random', random_state=0)
    
    
    return model