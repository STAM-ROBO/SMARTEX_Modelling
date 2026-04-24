"""
Pixel-level evaluation.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, accuracy_score
def run(model, X_test, y_test,class_list,prefix="pixel",thresholds=None):
    results = {"metrics": {}, "figures": {}, "tables": {}}
    n_classes = len(class_list)
    y_score = model.predict_proba(X_test)
    if(thresholds is None):
        
    
        
        thresholds = np.zeros(n_classes)
        
        for c in range(n_classes):
            best_t, best_m = 0.5, -1
            for t in np.linspace(0.1, 0.9, 17):
                y_c = (y_score[:, c] >= t).astype(int)
                m = f1_score(y_test[:, c], y_c,average='weighted')  # e.g. f1_score
                if m > best_m:
                    best_m, best_t = m, t
            thresholds[c] = best_t
        print("Optimal thresholds per class:", thresholds)
    """Evaluate classifier per pixel and compute basic metrics."""
    #print(X_test.shape,y_test.shape)
    #y_pred = model.predict(X_test)
    
    y_pred = (y_score >= thresholds).astype(int)
    results = {}
    # Compute metrics
    
    
    accuracy = accuracy_score(y_test, y_pred)
    results['metrics'] = {
        f"{prefix}_accuracy": accuracy_score(y_test, y_pred),
        f"{prefix}_precision": precision_score(y_test, y_pred, average="weighted"),
        f"{prefix}_recall": recall_score(y_test, y_pred, average="weighted"),
        f"{prefix}_f1_score": f1_score(y_test, y_pred, average="weighted"),
        
    }
    report = classification_report(y_test, y_pred, output_dict=True, target_names=class_list)
    for cls in class_list:
        if cls in report:
            m = report[cls]
            results['metrics'][f"{prefix}_{cls}_precision"] = float(m["precision"])
                
            results['metrics'][f"{prefix}_{cls}_recall"]=  float(m["recall"])
            results['metrics'][f"{prefix}_{cls}_f1"]= float(m["f1-score"])
            results['metrics'][f"{prefix}_{cls}_support"] =  int(m["support"])
            
    return results,thresholds

    
