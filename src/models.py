"""
models.py

目的:
- LightGBM回帰モデルの学習・Optunaによるハイパーパラメータ探索
- 推定したWinRatioを (TeamID, OTeamID) 単位で計算
"""
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split

from .utils import get_device  # utils.py などにgpu/cpu判定を入れていると仮定

def score_dataset(params, X, y):
    """
    cross_val_score でR^2を測り、-mean + std で返す (Notebook例準拠)
    """
    reg = lgb.LGBMRegressor(**params)
    score_arr = cross_val_score(reg, X, y, scoring='r2', cv=3)
    score = -1.0 * score_arr.mean() + score_arr.std()
    return score

def objective(trial, X, y):
    """
    Optunaのobjective
    """
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree',[0.3,0.5,0.7,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample',[0.4,0.5,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',[0.006,0.01,0.014,0.02]),
        'max_depth': trial.suggest_categorical('max_depth',[10,20,100]),
        'num_leaves': trial.suggest_int('num_leaves', 5, 31),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
        'device_type': get_device(),
        'verbose': -1
    }
    return score_dataset(params, X, y)

def tune_hyperparams(X, y, n_trials=50):
    """
    X, yに対してOptunaでハイパーパラメータ探索し、ベストパラメータを返す
    """
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    return study.best_params

def train_model(X, y, params):
    """
    最終モデルを学習して返す
    """
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X, y)
    return reg

def build_x_y(df, target_col='WinRatio'):
    """
    Notebookで定義していたbuild_x_y相当
    """
    feature_cols = list(df.columns)
    feature_cols.remove(target_col)
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def predict_winratio(model, X):
    """
    学習済モデルを使ってWinRatioを推定したDataFrameを返す
    """
    preds = model.predict(X)
    ret_df = pd.DataFrame(index=X.index)
    ret_df['WinRatio'] = preds
    return ret_df
