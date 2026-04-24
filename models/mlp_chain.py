from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.multioutput import ClassifierChain

def build_model(cfg,):
    """
    Fit OneVsRest SVM classifier on data X, y.
    Returns a fitted OneVsRestClassifier instance.
    """
    cfg_model = cfg.get("mlp_chain", {}  )
    hidden_layer_sizes = cfg_model.get('hidden_layer_sizes', (100, 50))
    alpha = cfg_model.get('alpha', 0.0001)
    solver = cfg_model.get('solver', 'adam')
    learning_rate_init = cfg_model.get('learning_rate_init', 0.1)
    
    model = OneVsRestClassifier(MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,                # L2 penalty
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-2,
        validation_fraction=0.2,
        shuffle=True,
        verbose=True
    ))
    
    
    return model