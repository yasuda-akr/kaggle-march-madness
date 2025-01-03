"""
feature_engineering.py

目的:
- Eloレーティング計算
- RPI算出
- シード情報加工 (clean_seeds)
- Massey Ordinals 結合
- (TeamID, OTeamID) 単位の履歴テーブルなど。
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

# ===================== #
#     ELO 計算関連
# ===================== #
def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None):
    """
    ノートブックで実装していた Elo 計算処理
    """
    team_dict = {t: initial_rating for t in teams}
    r1, r2 = [], []

    for wteam, lteam, ws, ls in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore),
                                     total=len(data),
                                     desc="Calculating ELO"):
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])

        rateW = 1.0 / (1.0 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1.0 / (1.0 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))

        margin_of_victory = 1.0
        if alpha:
            margin_of_victory = (ws - ls) / alpha

        team_dict[wteam] += k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += k * margin_of_victory * (0 - rateL)
        if team_dict[lteam] < 1:
            team_dict[lteam] = 1

    return r1, r2


def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None):
    """
    ELOの履歴をまとめた上で、(TeamID,Season)単位に平均や最終値をまとめる
    """
    r1, r2 = calculate_elo(teams, data, initial_rating, k, alpha)

    # DataFrameとしてまとめ
    seasons = np.concatenate([data.Season, data.Season])
    days    = np.concatenate([data.DayNum, data.DayNum])
    tids    = np.concatenate([data.WTeamID, data.LTeamID])
    # 例として、0/1でトーナメントフラグを持つ場合
    if 'tourney' not in data.columns:
        data['tourney'] = 0
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])

    rating_df = pd.DataFrame({
        'Season' : seasons,
        'DayNum' : days,
        'TeamID' : tids,
        'Rating' : ratings,
        'Tourney': tourney
    })

    # レギュラーシーズンのみ
    rating_df = rating_df[rating_df['Tourney'] == 0].copy()
    rating_df.sort_values(['TeamID','Season','DayNum'], inplace=True)

    grouped = rating_df.groupby(['TeamID','Season'])
    results = grouped['Rating'].agg(['mean','median','std','min','max','last'])
    results.columns = ['Rating_Mean','Rating_Median','Rating_Std','Rating_Min','Rating_Max','Rating_Last']
    # 線形回帰の傾き計算
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope)

    results.reset_index(inplace=True)
    return results


def build_elo(teams, results, gender):
    """
    (RegularSeasonCompactResults + NCAATourneyCompactResults) を結合した results から、
    レギュラーシーズン分のみを抽出し Eloを計算
    """
    # 'tourney'のフラグをつける (最初の行数を境にレギュラー/トーナメントを分けるやり方は適宜修正)
    # あるいは、resultsに既にTournament列等あれば使う。
    # ここでは簡易的に'WLoc'などを元にtourneyフラグを推定しても良い
    # ノートブック内では len(csvs[0]) で切り分けていたので、そのロジックを適宜修正

    # まずは単純に全てをSeason,DayNum順でソート
    results = results.sort_values(['Season','DayNum']).copy()
    results['tourney'] = 0
    # 例えば "NCAATourneyCompactResults" 分だけフラグ1 にしたい場合は外部でセットしてbuild_eloに渡してもOK

    # チーム一覧
    team_list = teams.reset_index()['TeamID'].unique()
    elo_df = create_elo_data(team_list, results, initial_rating=2000, k=140, alpha=None)

    # (TeamID単位) か (TeamID,Season単位) で集計するかは要件次第
    # ここでは全シーズン平均をとる例:
    final_df = elo_df.groupby('TeamID').mean(numeric_only=True)
    return final_df


# ===================== #
#     勝率, RPI 関連
# ===================== #

def build_season_results(df):
    """
    (WTeamID, LTeamID)ベースのdfを
    TeamID視点で爆発させ、勝率などを求める
    """
    season_results = df.copy()
    # 行をチーム視点で爆発
    season_results['TeamID'] = season_results[['WTeamID','LTeamID']].values.tolist()
    season_results = season_results.explode('TeamID')
    season_results['Win']    = season_results.apply(lambda row: 1 if row['TeamID']==row['WTeamID'] else 0, axis=1)
    season_results['Defeat'] = 1 - season_results['Win']
    season_results['Games']  = 1

    # スコア差
    def score_diff(row):
        return row['WScore'] - row['LScore'] if row['TeamID'] == row['WTeamID'] else row['LScore'] - row['WScore']
    season_results['ScoreDiff'] = season_results.apply(score_diff, axis=1)

    # 相手チームID
    def opponent(row):
        return row['LTeamID'] if row['TeamID']==row['WTeamID'] else row['WTeamID']
    season_results['OTeamID'] = season_results.apply(opponent, axis=1)

    # ホーム判定(WLoc)
    # ただし "H" が勝者につく仕様なので注意
    # 簡易的に: 
    season_results['Home'] = season_results['WLoc'].apply(lambda x: 1 if x=='H' else 0)

    # 集計
    grouped = season_results.groupby(['TeamID','OTeamID']).agg({
        'Win': 'sum',
        'Defeat':'sum',
        'Games':'sum',
        'ScoreDiff':'sum',
        'Home':'sum'
    })
    grouped['WinRatio'] = grouped['Win'] / grouped['Games']
    return grouped


def build_rpi(results):
    """
    ノートブックで書いていたRPI計算
    results: (TeamID, OTeamID)単位
    """
    # WPはTeamID単位に勝率平均
    wp = results[['WinRatio']].groupby('TeamID').mean().rename(columns={'WinRatio':'WP'})
    # resultsにもWPをjoin
    rpi = results.reset_index().merge(wp, on='TeamID')
    rpi = rpi.merge(wp, left_on='OTeamID', right_on='TeamID', suffixes=('_T','_O'))
    # WP_OO (相手の相手)
    wp_oo = rpi.groupby('TeamID_x')['WP_O'].mean().rename('WP_OO').reset_index()
    rpi = rpi.merge(wp_oo, left_on='OTeamID', right_on='TeamID_x')
    # RPI
    rpi['RPI'] = (0.25*rpi['WP_T'] + 0.50*rpi['WP_O'] + 0.25*rpi['WP_OO'])

    return rpi[['TeamID_x','OTeamID','RPI']].set_index(['TeamID_x','OTeamID'])


# ===================== #
#     シード, ランキング
# ===================== #
def clean_seeds(seed_str):
    """
    'W16a' -> '16' などに変換
    """
    s = seed_str[1:]  # 最初のアルファベットはregion
    if len(s) > 2:
        s = s[:-1]  # 末尾の 'a' or 'b' を削除
    return int(s)

def build_seeds(CSV, gender):
    """
    NCAATourneySeedsからシード情報を(TeamID単位に平均して)返す
    """
    key = f"{gender}NCAATourneySeeds"
    df = CSV[key].copy()
    df['Seed'] = df['Seed'].apply(clean_seeds)
    df.drop('Season', axis=1, inplace=True)
    # シーズン毎の平均(雑に平均しているので、要件によっては最新年のみ使用等検討)
    return df.groupby('TeamID').mean(numeric_only=True)

def build_rankings(CSV, gender):
    """
    Massey Ordinals (men only) をteamIDごとに平均
    """
    # 'MasseyOrdinals_thruSeason2024_day128' という名前でcsvがある想定
    key = f"{gender}MasseyOrdinals_thruSeason2024_day128"
    if key not in CSV:
        return None  # 女子など存在しない場合
    df = CSV[key].copy()
    df.drop(['SystemName','RankingDayNum'], axis=1, inplace=True)
    df.drop('Season', axis=1, inplace=True, errors='ignore')
    return df.groupby('TeamID').mean(numeric_only=True)


# ===================== #
#   履歴テーブル構築
# ===================== #
def build_history(season_results, seeds, teams, elo, rpi, rankings=None):
    """
    (TeamID, OTeamID)単位の行に、シード/ランキング/Elo/RPIなど特徴量をjoinする
    """
    history = season_results.join(teams, on='TeamID') \
                           .join(seeds, on='TeamID', rsuffix='_Seed') \
                           .join(elo,   on='TeamID', rsuffix='_Elo') \
                           .join(rpi,   on=['TeamID','OTeamID'])
    history = history.reset_index()

    # 相手(OTeamID)側のRPIも同様にjoin (あるいはRPIdiffを求める)
    rpi_opp = rpi.reset_index().rename(columns={'TeamID_x':'OTeamID','OTeamID':'TeamID'})
    history = history.merge(rpi_opp, on=['TeamID','OTeamID'], suffixes=('_T','_O'))

    # seed差
    history['SeedDiff'] = history['Seed_T'] - history['Seed_O']

    # rankings
    if rankings is not None:
        history = history.merge(rankings, left_on='TeamID', right_on='TeamID', suffixes=(None,'_RankT'))
        history = history.merge(rankings, left_on='OTeamID', right_on='TeamID', suffixes=('_T','_O'))
        history['RankingsDiff'] = history['OrdinalRank_T'] - history['OrdinalRank_O']
        history.drop(['TeamID_T','TeamID_O'], axis=1, errors='ignore', inplace=True)

    history = history.set_index(['TeamID','OTeamID']).fillna(0)
    return history


def build_avg(history):
    """
    例: (TeamID, OTeamID)→TeamID単位に集約するときの処理例
    """
    agg_dict = {}
    for col in history.columns:
        if col in ['Games','Home']: # 総数を足し合わせたい場合
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'mean'
    return history.groupby('TeamID').agg(agg_dict)
