import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore', message='n_jobs value .* overridden to 1 by setting random_state')

# precomputing representations for downstream tasks
def precompute_representations(adata, seed=33):
    # standardize
    scaler = StandardScaler()
    X_all = scaler.fit_transform(adata.layers["mClrs"])
    adata.obsm["X_pca_50"] = PCA(n_components=50, random_state=seed).fit_transform(X_all)
    print("PCA done.")
    adata.obsm["X_umap_50"] = umap.UMAP(n_components=50, random_state=seed).fit_transform(X_all)
    print("UMAP done.")

    return adata

# print classification results with class distribution
def print_classification_results(classification_results):
  
    for target, results in classification_results.items():
        #
        print(f"Classification Task: {target}")
        print("-" * 70)
        
        # print
        print("Class Distribution:")
        value_counts = results['value_counts']
        total = sum(value_counts.values())
        for class_name, count in sorted(value_counts.items()):
            percentage = (count / total) * 100
            print(f"  {class_name:20s}: {count:6d} ({percentage:5.2f}%)")
        print(f"  {'Total':20s}: {total:6d}")
        
        #  by representation
        reps = results["representations"]
        baseline = results["baseline"]["XGBoost on raw clrs"]
        
        print("Results by Representation:")
        print(f"{'Representation':20s} {'Method':12s} {'Accuracy':>10s} {'F1_macro':>10s}")
        print("-" * 70)
        
        for rep_name, models in reps.items():
            for model_name, metrics in models.items():
                acc = metrics['Acc']
                f1 = metrics['F1_macro']
                print(f"{rep_name:20s} {model_name:12s} {acc:10.3f} {f1:10.3f}")
        
        #baseline
        print(f"{'Baseline':20s} {'XGBoost':12s} {baseline['Acc']:10.3f} {baseline['F1_macro']:10.3f}")
        


# regression results printing
def print_regression_results(regression_results):
    reps = regression_results["representations"]
    baseline = regression_results["baseline"]["XGBoost on raw clrs"]
    
    print("Results by Representation:")
    print(f"{'Representation':20s} {'Method':12s} {'R²':>10s} {'RMSE':>10s}")
    print("-" * 70)
    
    for rep_name, models in reps.items():
        for model_name, metrics in models.items():
            r2 = metrics['R²']
            rmse = metrics['RMSE']
            print(f"{rep_name:20s} {model_name:12s} {r2:10.3f} {rmse:10.3f}")
    
    #baseline
    print(f"{'Baseline':20s} {'XGBoost':12s} {baseline['R²']:10.3f} {baseline['RMSE']:10.3f}")




# regression plots
def plot_regression_results(regression_results):
    reps = regression_results["representations"]
    baseline = regression_results["baseline"]["XGBoost on raw clrs"]
    
    rep_names = list(reps.keys())
    ridge_r2 = [reps[rep]["Ridge"]["R²"] for rep in rep_names]
    xgb_r2 = [reps[rep]["XGBoost"]["R²"] for rep in rep_names]
    ridge_rmse = [reps[rep]["Ridge"]["RMSE"] for rep in rep_names]
    xgb_rmse = [reps[rep]["XGBoost"]["RMSE"] for rep in rep_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = range(len(rep_names))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], ridge_r2, width, label="Ridge", alpha=0.8)
    axes[0].bar([i + width/2 for i in x], xgb_r2, width, label="XGBoost", alpha=0.8)
    axes[0].axhline(baseline["R²"], color="gray", linestyle="--", linewidth=2, label="Baseline XGBoost raw CLR")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Regression: R² by Representation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rep_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    
    axes[1].bar([i - width/2 for i in x], ridge_rmse, width, label="Ridge", alpha=0.8)
    axes[1].bar([i + width/2 for i in x], xgb_rmse, width, label="XGBoost", alpha=0.8)
    axes[1].axhline(baseline["RMSE"], color="gray", linestyle="--", linewidth=2, label="Baseline XGBoost raw CLR")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Regression: RMSE by Representation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rep_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.show()

# classification plots
def plot_classification_results(classification_results):
    for target, results in classification_results.items():
        reps = results["representations"]
        baseline = results["baseline"]["XGBoost on raw clrs"]
        
        rep_names = list(reps.keys())
        logreg_acc = [reps[rep]["LogReg"]["Acc"] for rep in rep_names]
        xgb_acc = [reps[rep]["XGBoost"]["Acc"] for rep in rep_names]
        logreg_f1 = [reps[rep]["LogReg"]["F1_macro"] for rep in rep_names]
        xgb_f1 = [reps[rep]["XGBoost"]["F1_macro"] for rep in rep_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = range(len(rep_names))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], logreg_acc, width, label="LogReg", alpha=0.8)
        axes[0].bar([i + width/2 for i in x], xgb_acc, width, label="XGBoost", alpha=0.8)
        axes[0].axhline(baseline["Acc"], color="gray", linestyle="--", linewidth=2, label="Baseline XGBoost raw CLR")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"{target}: Accuracy by Representation")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(rep_names, rotation=45, ha="right")
        #axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")
        
        axes[1].bar([i - width/2 for i in x], logreg_f1, width, label="LogReg", alpha=0.8)
        axes[1].bar([i + width/2 for i in x], xgb_f1, width, label="XGBoost", alpha=0.8)
        axes[1].axhline(baseline["F1_macro"], color="gray", linestyle="--", linewidth=2, label="Baseline XGBoost raw CLR")
        axes[1].set_ylabel("F1_macro")
        axes[1].set_title(f"{target}: F1_macro by Representation")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(rep_names, rotation=45, ha="right")
        #axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.show()



