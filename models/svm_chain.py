from sklearn.svm import SVC
from sklearn.multioutput import ClassifierChain

def build_model(cfg,):
    """
    Fit OneVsRest SVM classifier on data X, y.
    Returns a fitted OneVsRestClassifier instance.
    """
    cfg_model = cfg.get("svm_chain", {}  )
    model = ClassifierChain(SVC(kernel="rbf", probability=True), order='random', random_state=0)
    
    
    return model