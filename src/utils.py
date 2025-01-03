"""
utils.py

目的:
- GPU/CPU判定ユーティリティ
- トーナメントのスロットごとのシミュレーション関数 etc.
"""
import numpy as np
import pandas as pd
import os

def get_device():
    """
    PyTorchを使用してGPU判定を行う
    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        # PyTorch未インストールの場合
        return 'cpu'\

def prepare_data(seeds):
    """
    seeds: pd.DataFrame  # 例: 2024_tourney_seeds.csv
    """
    seed_dict = seeds.set_index('Seed')['TeamID'].to_dict()
    inverted_seed_dict = {v:k for k,v in seed_dict.items()}
    return seed_dict, inverted_seed_dict


def simulate(round_slots, seeds, inverted_seeds, wins):
    """
    Notebook内のsimulateを抜粋
    """
    winners = []
    slots   = []

    for slot, strong, weak in zip(round_slots.Slot, round_slots.StrongSeed, round_slots.WeakSeed):
        team_1, team_2 = seeds[strong], seeds[weak]
        # team_1がteam_2に勝つ確率
        team_1_prob = wins.loc[(team_1, team_2),'WinRatio']
        winner = np.random.choice([team_1, team_2], p=[team_1_prob, 1-team_1_prob])
        winners.append(winner)
        slots.append(slot)

        # 勝者を次ラウンドのslot名でseedに入れておく
        seeds[slot] = winner

    # bracketの出力用に
    return [inverted_seeds[w] for w in winners], slots


def run_simulation(seeds, round_slots, wins, brackets=1000):
    seed_dict, inverted_seed_dict = prepare_data(seeds)
    results = []
    bracket = []
    slot_log = []
    for b in range(1, brackets+1):
        r, s = simulate(round_slots.copy(), seed_dict.copy(), inverted_seed_dict.copy(), wins)
        results.extend(r)
        bracket.extend([b]*len(r))
        slot_log.extend(s)

    result_df = pd.DataFrame({
        'Bracket': bracket,
        'Slot': slot_log,
        'Team': results
    })
    return result_df
