"""
Generates a synthetic but realistic IPL Player Performance dataset.
Mimics ball-by-ball / match aggregated stats used for predicting
a batsman's runs scored in the next match.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

players = [
    "V Kohli", "RG Sharma", "S Dhawan", "KL Rahul", "DA Warner",
    "MS Dhoni", "AB de Villiers", "JC Buttler", "Q de Kock", "F du Plessis",
    "SV Samson", "RR Pant", "SA Yadav", "HH Pandya", "RA Jadeja",
    "AT Rayudu", "EJG Morgan", "N Pooran", "WP Saha", "PP Shaw",
    "MK Pandey", "G Maxwell", "DJ Hooda", "RD Gaikwad", "Shubman Gill"
]

teams = ["MI", "CSK", "RCB", "KKR", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]
venues = [
    "Wankhede Stadium", "Eden Gardens", "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium", "Arun Jaitley Stadium", "Narendra Modi Stadium",
    "Sawai Mansingh Stadium", "Rajiv Gandhi Intl Stadium", "PCA Stadium", "Ekana Stadium"
]

n_rows = 2500
rows = []

# assign baseline "skill" to each player
player_skill = {p: np.random.uniform(20, 55) for p in players}
player_strike_rate = {p: np.random.uniform(115, 165) for p in players}

for i in range(n_rows):
    player = np.random.choice(players)
    team = np.random.choice(teams)
    opponent = np.random.choice([t for t in teams if t != team])
    venue = np.random.choice(venues)
    season = np.random.choice([2019, 2020, 2021, 2022, 2023, 2024])

    batting_position = np.random.randint(1, 8)
    matches_played_season = np.random.randint(5, 16)

    # rolling form features
    avg_runs_last5 = max(0, np.random.normal(player_skill[player], 15))
    avg_sr_last5 = max(60, np.random.normal(player_strike_rate[player], 20))
    balls_faced_last5_avg = max(5, np.random.normal(25, 10))

    home_advantage = 1 if np.random.rand() < 0.5 else 0
    toss_won = np.random.choice([0, 1])
    pitch_type = np.random.choice(["Batting", "Bowling", "Balanced"], p=[0.35, 0.25, 0.4])
    opponent_bowling_rank = np.random.randint(1, 11)  # 1 = best bowling attack

    # target variable: runs scored in next match
    base = player_skill[player]
    position_effect = max(0, (5 - batting_position) * 3)
    form_effect = (avg_runs_last5 - player_skill[player]) * 0.4
    pitch_effect = {"Batting": 8, "Bowling": -8, "Balanced": 0}[pitch_type]
    opp_effect = (6 - opponent_bowling_rank) * 1.5
    home_effect = 3 if home_advantage else 0
    toss_effect = 2 if toss_won else 0
    noise = np.random.normal(0, 12)

    runs_scored = base + position_effect + form_effect + pitch_effect + opp_effect + home_effect + toss_effect + noise
    runs_scored = max(0, round(runs_scored))

    rows.append({
        "player_name": player,
        "team": team,
        "opponent": opponent,
        "venue": venue,
        "season": season,
        "batting_position": batting_position,
        "matches_played_season": matches_played_season,
        "avg_runs_last5": round(avg_runs_last5, 2),
        "avg_strike_rate_last5": round(avg_sr_last5, 2),
        "avg_balls_faced_last5": round(balls_faced_last5_avg, 2),
        "home_advantage": home_advantage,
        "toss_won": toss_won,
        "pitch_type": pitch_type,
        "opponent_bowling_rank": opponent_bowling_rank,
        "runs_scored": runs_scored
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ipl-prediction/data/ipl_player_performance.csv", index=False)
print("Dataset created:", df.shape)
print(df.head())
