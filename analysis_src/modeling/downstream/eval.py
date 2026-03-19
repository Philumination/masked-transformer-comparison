# best eval code now with saving results for plotting

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score
)
from xgboost import XGBRegressor, XGBClassifier

#filter warnings
import warnings
warnings.filterwarnings('ignore', message='n_jobs value .* overridden to 1 by setting random_state')

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

seed = 33


#  regression
def eval_regression(reps, y_series, baseline_X):
    
    print(f"Evaluating regression task: {y_series.name}")
    mask = ~np.isnan(y_series)
    valid_idx = np.where(mask)[0]
    train_idx, test_idx = train_test_split(valid_idx, test_size=0.2, random_state=seed)
    y_train = y_series.iloc[train_idx].values
    y_test = y_series.iloc[test_idx].values

    results = {
        'task': f"Regression: {y_series.name}",
        'representations': {},
        'baseline': {}
    }

    # Linear Models (Ridge) on all representations
    for name, X in reps.items():
        print(f"  Evaluating representation: {name}")

        X_train, X_test = X[train_idx], X[test_idx]
        

       
        # Ridge
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=seed))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        
        # XGBoost on representations
        xgb_reg = XGBRegressor(random_state=seed, n_jobs=-1, n_estimators=100, device ="cuda")
        xgb_reg.fit(X_train, y_train)
        xgb_preds = xgb_reg.predict(X_test)
        
        xgb_r2 = r2_score(y_test, xgb_preds)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        
        results['representations'][name] = {
            'Ridge': {'R²': r2, 'RMSE': rmse},
            'XGBoost': {'R²': xgb_r2, 'RMSE': xgb_rmse}
        }

    # XGBoost on raw CLRs
    xgb = XGBRegressor(random_state=seed, n_jobs=-1, n_estimators=100, device="cuda")
    xgb.fit(baseline_X[train_idx], y_train)
    xgb_preds = xgb.predict(baseline_X[test_idx])
    
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    results['baseline']['XGBoost on raw clrs'] = {'R²': xgb_r2, 'RMSE': xgb_rmse}
    
    return results

# classification
def eval_classification(reps, y_series, baseline_X):
    print(f"Evaluating classification task: {y_series.name}")

    mask = y_series.notna() & (y_series != "unknown")
    valid_idx = np.where(mask)[0]
    
    # NOW convert only valid values to string
    y_valid = y_series.iloc[valid_idx].astype(str)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_valid)
    print(pd.Series(y_encoded).value_counts())
    train_idx, test_idx, y_train, y_test = train_test_split(
        valid_idx, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )
    
    results = {
        'task': f"Classification: {y_series.name}",
        'value_counts': y_valid.value_counts().to_dict(),
        'representations': {},
        'baseline': {}
    }

    for name, X in reps.items():
        print(f"  Evaluating representation: {name}")
        X_train, X_test = X[train_idx], X[test_idx]
        # LogReg
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=seed, max_iter=100, class_weight='balanced'))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print("logreg done now xgb")
        # xgb on representations
        xgb_clf = XGBClassifier(random_state=seed, n_jobs=-1, n_estimators=100, device="cuda")
        xgb_clf.fit(X_train, y_train)
        xgb_preds = xgb_clf.predict(X_test)
        
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
        
        results['representations'][name] = {
            'LogReg': {'Acc': acc, 'F1_macro': f1},
            'XGBoost': {'Acc': xgb_acc, 'F1_macro': xgb_f1}
        }
        print("xgb done")
    print("baseline xgb now")
    # XGBoost on raw CLRs
    xgb = XGBClassifier(random_state=seed, n_estimators=100, n_jobs=-1, device="cuda")
    xgb.fit(baseline_X[train_idx], y_train)
    xgb_preds = xgb.predict(baseline_X[test_idx])
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
    results['baseline']['XGBoost on raw clrs'] = {'Acc': xgb_acc, 'F1_macro': xgb_f1}
    
    return results



def eval_regression_cv(reps, y_series, baseline_X, n_splits=10):
    print(f"Evaluating regression task (CV): {y_series.name}")

    mask = y_series.notna()
    valid_idx = np.where(mask)[0]


    effective_splits = min(n_splits, len(valid_idx))

    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    folds = list(kf.split(valid_idx))

    results = {
        "task": f"Regression CV: {y_series.name}",
        "n_splits": effective_splits,
        "representations": {},
        "baseline": {},
    }

    for name, X in reps.items():
        print(f"  Evaluating representation: {name}")

        ridge_r2_scores, ridge_rmse_scores = [], []
        xgb_r2_scores, xgb_rmse_scores = [], []

        for tr_fold, te_fold in folds:
            train_idx = valid_idx[tr_fold]
            test_idx = valid_idx[te_fold]

            y_train = y_series.iloc[train_idx].values
            y_test = y_series.iloc[test_idx].values
            X_train, X_test = X[train_idx], X[test_idx]

            ridge_model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(random_state=seed)),
            ])
            ridge_model.fit(X_train, y_train)
            ridge_preds = ridge_model.predict(X_test)

            ridge_r2_scores.append(r2_score(y_test, ridge_preds))
            ridge_rmse_scores.append(np.sqrt(mean_squared_error(y_test, ridge_preds)))

            xgb_reg = XGBRegressor(
                random_state=seed,
                n_jobs=-1,
                n_estimators=100,
                device="cuda",
            )
            xgb_reg.fit(X_train, y_train)
            xgb_preds = xgb_reg.predict(X_test)

            xgb_r2_scores.append(r2_score(y_test, xgb_preds))
            xgb_rmse_scores.append(np.sqrt(mean_squared_error(y_test, xgb_preds)))

        results["representations"][name] = {
            "Ridge": {
                "R2_mean": float(np.mean(ridge_r2_scores)),
                "R2_std": float(np.std(ridge_r2_scores)),
                "RMSE_mean": float(np.mean(ridge_rmse_scores)),
                "RMSE_std": float(np.std(ridge_rmse_scores)),
            },
            "XGBoost": {
                "R2_mean": float(np.mean(xgb_r2_scores)),
                "R2_std": float(np.std(xgb_r2_scores)),
                "RMSE_mean": float(np.mean(xgb_rmse_scores)),
                "RMSE_std": float(np.std(xgb_rmse_scores)),
            },
        }

    baseline_r2_scores, baseline_rmse_scores = [], []
    for tr_fold, te_fold in folds:
        train_idx = valid_idx[tr_fold]
        test_idx = valid_idx[te_fold]

        y_train = y_series.iloc[train_idx].values
        y_test = y_series.iloc[test_idx].values

        xgb = XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            n_estimators=100,
            device="cuda",
        )
        xgb.fit(baseline_X[train_idx], y_train)
        xgb_preds = xgb.predict(baseline_X[test_idx])

        baseline_r2_scores.append(r2_score(y_test, xgb_preds))
        baseline_rmse_scores.append(np.sqrt(mean_squared_error(y_test, xgb_preds)))

    results["baseline"]["XGBoost on raw clrs"] = {
        "R2_mean": float(np.mean(baseline_r2_scores)),
        "R2_std": float(np.std(baseline_r2_scores)),
        "RMSE_mean": float(np.mean(baseline_rmse_scores)),
        "RMSE_std": float(np.std(baseline_rmse_scores)),
    }

    return results

def eval_classification_cv(reps, y_series, baseline_X, n_splits=10):
    print(f"Evaluating classification task (CV): {y_series.name}")

    mask = y_series.notna() & (y_series != "unknown")
    valid_idx = np.where(mask)[0]

    y_valid = y_series.iloc[valid_idx].astype(str)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_valid)

    class_counts = np.bincount(y_encoded)
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0


    effective_splits = min(n_splits, min_class_count)

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(valid_idx, y_encoded))

    results = {
        "task": f"Classification CV: {y_series.name}",
        "n_splits": effective_splits,
        "value_counts": y_valid.value_counts().to_dict(),
        "representations": {},
        "baseline": {},
    }

    for name, X in reps.items():
        print(f"  Evaluating representation: {name}")

        logreg_acc_scores, logreg_f1_scores = [], []
        xgb_acc_scores, xgb_f1_scores = [], []

        for tr_fold, te_fold in folds:
            train_idx = valid_idx[tr_fold]
            test_idx = valid_idx[te_fold]

            y_train = y_encoded[tr_fold]
            y_test = y_encoded[te_fold]
            X_train, X_test = X[train_idx], X[test_idx]

            logreg_model = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    random_state=seed,
                    max_iter=100,
                    class_weight="balanced",
                )),
            ])
            logreg_model.fit(X_train, y_train)
            logreg_preds = logreg_model.predict(X_test)

            logreg_acc_scores.append(accuracy_score(y_test, logreg_preds))
            logreg_f1_scores.append(f1_score(y_test, logreg_preds, average="macro"))

            xgb_clf = XGBClassifier(
                random_state=seed,
                n_jobs=-1,
                n_estimators=100,
                device="cuda",
            )
            xgb_clf.fit(X_train, y_train)
            xgb_preds = xgb_clf.predict(X_test)

            xgb_acc_scores.append(accuracy_score(y_test, xgb_preds))
            xgb_f1_scores.append(f1_score(y_test, xgb_preds, average="macro"))

        results["representations"][name] = {
            "LogReg": {
                "Acc_mean": float(np.mean(logreg_acc_scores)),
                "Acc_std": float(np.std(logreg_acc_scores)),
                "F1_macro_mean": float(np.mean(logreg_f1_scores)),
                "F1_macro_std": float(np.std(logreg_f1_scores)),
            },
            "XGBoost": {
                "Acc_mean": float(np.mean(xgb_acc_scores)),
                "Acc_std": float(np.std(xgb_acc_scores)),
                "F1_macro_mean": float(np.mean(xgb_f1_scores)),
                "F1_macro_std": float(np.std(xgb_f1_scores)),
            },
        }

    baseline_acc_scores, baseline_f1_scores = [], []
    for tr_fold, te_fold in folds:
        train_idx = valid_idx[tr_fold]
        test_idx = valid_idx[te_fold]

        y_train = y_encoded[tr_fold]
        y_test = y_encoded[te_fold]

        xgb = XGBClassifier(
            random_state=seed,
            n_estimators=100,
            n_jobs=-1,
            device="cuda",
        )
        xgb.fit(baseline_X[train_idx], y_train)
        xgb_preds = xgb.predict(baseline_X[test_idx])

        baseline_acc_scores.append(accuracy_score(y_test, xgb_preds))
        baseline_f1_scores.append(f1_score(y_test, xgb_preds, average="macro"))

    results["baseline"]["XGBoost on raw clrs"] = {
        "Acc_mean": float(np.mean(baseline_acc_scores)),
        "Acc_std": float(np.std(baseline_acc_scores)),
        "F1_macro_mean": float(np.mean(baseline_f1_scores)),
        "F1_macro_std": float(np.std(baseline_f1_scores)),
    }

    return results