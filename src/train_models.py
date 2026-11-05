"""
Trains and compares multiple regression models for IPL player
runs prediction. Saves comparison metrics and the best model.
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/claude/ipl-prediction/data/ipl_player_performance.csv")

target = "runs_scored"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = ["player_name", "team", "opponent", "venue", "pitch_type"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
}

results = {}
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}
    trained_pipelines[name] = pipe
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")

# Save comparison table
results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
results_df.to_csv("/home/claude/ipl-prediction/models/model_comparison.csv")
print("\nModel Comparison:\n", results_df)

# Save best model
best_model_name = results_df.index[0]
best_pipeline = trained_pipelines[best_model_name]
joblib.dump(best_pipeline, "/home/claude/ipl-prediction/models/best_model.pkl")

with open("/home/claude/ipl-prediction/models/results.json", "w") as f:
    json.dump({"best_model": best_model_name, "results": results}, f, indent=2)

print(f"\nBest Model: {best_model_name}")

# Plot model comparison (R2 scores)
plt.figure(figsize=(9, 5))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="mako")
plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/07_model_comparison_r2.png", dpi=120)
plt.close()

# Plot MAE comparison
plt.figure(figsize=(9, 5))
sns.barplot(x=results_df.index, y=results_df["MAE"], palette="flare")
plt.title("Model Comparison - MAE (Lower is Better)")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Model")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/08_model_comparison_mae.png", dpi=120)
plt.close()

# Feature importance for best tree-based model (if available)
if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]:
    model = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="crest")
    plt.title(f"Top 15 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig("/home/claude/ipl-prediction/images/09_feature_importance.png", dpi=120)
    plt.close()

# Actual vs Predicted plot for best model
preds = best_pipeline.predict(X_test)
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.4, color="#2ca02c")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Runs")
plt.ylabel("Predicted Runs")
plt.title(f"Actual vs Predicted Runs - {best_model_name}")
plt.tight_layout()
plt.savefig("/home/claude/ipl-prediction/images/10_actual_vs_predicted.png", dpi=120)
plt.close()

print("All plots and models saved.")
