"""
Load the trained model and predict runs for a sample input.
Run: python src/predict.py
"""

import joblib
import pandas as pd

model = joblib.load("/home/claude/ipl-prediction/models/best_model.pkl")

sample_input = pd.DataFrame([{
    "player_name": "V Kohli",
    "team": "RCB",
    "opponent": "CSK",
    "venue": "M Chinnaswamy Stadium",
    "season": 2024,
    "batting_position": 3,
    "matches_played_season": 12,
    "avg_runs_last5": 48.5,
    "avg_strike_rate_last5": 140.2,
    "avg_balls_faced_last5": 32.0,
    "home_advantage": 1,
    "toss_won": 1,
    "pitch_type": "Batting",
    "opponent_bowling_rank": 4
}])

prediction = model.predict(sample_input)
print(f"Predicted Runs for {sample_input['player_name'][0]}: {prediction[0]:.1f}")
