"""
Exploratory Data Analysis for IPL Player Performance dataset.
Generates and saves plots to the images/ directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("/home/claude/ipl-prediction/data/ipl_player_performance.csv")

print(df.describe())
print(df.isnull().sum())

# 1. Distribution of runs scored
plt.figure(figsize=(8, 5))
sns.histplot(df["runs_scored"], bins=30, kde=True, color="#1f77b4")
plt.title("Distribution of Runs Scored")
plt.xlabel("Runs Scored")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/01_runs_distribution.png", dpi=120)
plt.close()

# 2. Average runs by batting position
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="batting_position", y="runs_scored", palette="viridis")
plt.title("Average Runs Scored by Batting Position")
plt.xlabel("Batting Position")
plt.ylabel("Average Runs Scored")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/02_runs_by_position.png", dpi=120)
plt.close()

# 3. Runs scored by pitch type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="pitch_type", y="runs_scored", palette="Set2")
plt.title("Runs Scored by Pitch Type")
plt.xlabel("Pitch Type")
plt.ylabel("Runs Scored")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/03_runs_by_pitch.png", dpi=120)
plt.close()

# 4. Correlation heatmap (numeric features)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/04_correlation_heatmap.png", dpi=120)
plt.close()

# 5. Avg runs in last 5 vs runs scored (scatter)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="avg_runs_last5", y="runs_scored", hue="pitch_type", alpha=0.5)
plt.title("Recent Form (Avg Runs Last 5) vs Runs Scored")
plt.xlabel("Avg Runs (Last 5 Matches)")
plt.ylabel("Runs Scored")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/05_form_vs_runs.png", dpi=120)
plt.close()

# 6. Top players by average runs
plt.figure(figsize=(10, 6))
top_players = df.groupby("player_name")["runs_scored"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_players.values, y=top_players.index, palette="rocket")
plt.title("Top 10 Players by Average Runs Scored")
plt.xlabel("Average Runs Scored")
plt.ylabel("Player")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/06_top_players.png", dpi=120)
plt.close()

print("EDA plots saved to images/ directory.")
