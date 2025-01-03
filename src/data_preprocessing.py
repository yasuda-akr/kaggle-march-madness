"""
data_preprocessing.py

目的:
- Kaggle提供のCSVファイルを読み込み、基本的な統合処理を行う。
- チーム一覧、試合結果(レギュラーシーズン＋トーナメント)を結合。
"""

import glob
import os
import pandas as pd

def load_kaggle_data(data_dir):
    """
    data_dir: str
        Kaggleデータのディレクトリパス (例: 'data/raw')
    
    Returns:
    - CSV: Dict[str, pd.DataFrame]
        {ファイル名(拡張子除く): DataFrame} という辞書
    """
    CSV = {}
    for path in glob.glob(data_dir + "/*.csv"):
        basename = os.path.basename(path).split('.')[0]
        CSV[basename] = pd.read_csv(path, encoding='cp1252')
    return CSV


def build_results(CSV, gender):
    """
    指定したgender ('M' or 'W') における
    - NCAATourneyCompactResults
    - RegularSeasonCompactResults
    を連結して返す
    """
    csv_names = [f"{gender}NCAATourneyCompactResults", f"{gender}RegularSeasonCompactResults"]
    dfs = [CSV[name] for name in csv_names if name in CSV]
    if len(dfs) == 0:
        raise ValueError(f"No result files found for gender: {gender}")
    return pd.concat(dfs, ignore_index=True)


def build_teams(CSV, gender):
    """
    genderに応じたTeamsをDataFrameで返す。
    （TeamNameは削除し、TeamIDをインデックスに設定）
    """
    key = f"{gender}Teams"
    if key not in CSV:
        raise ValueError(f"{key} not found in CSV dictionary")

    teams = CSV[key].copy()
    if 'TeamName' in teams.columns:
        teams.drop('TeamName', axis=1, inplace=True)
    teams.set_index('TeamID', inplace=True)
    return teams

