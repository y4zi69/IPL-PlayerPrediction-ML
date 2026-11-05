import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="IPL Player Predictor", page_icon="🏏", layout="wide")

st.title("🏏 IPL Player Performance Predictor")
st.markdown("Enter the exact match conditions and player stats below to predict runs.")

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found at `models/best_model.pkl`.")
    st.stop()

st.divider()

# Create a clean 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Match Info")
    season = st.number_input("Season (Year)", min_value=2008, max_value=2025, value=2024)
    team = st.selectbox("Player's Team", ["CSK", "MI", "KKR", "DC", "RR", "SRH", "PBKS", "LSG", "GT", "RCB"])
    opponent = st.selectbox("Opponent Team", ["CSK", "MI", "KKR", "DC", "RR", "SRH", "PBKS", "LSG", "GT", "RCB"])
    matches_played_season = st.number_input("Matches Played This Season", min_value=0, max_value=17, value=5)

with col2:
    st.subheader("Match Conditions")
    home_advantage = st.selectbox("Home Advantage?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    toss_won = st.selectbox("Toss Won by Player's Team?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    opponent_bowling_rank = st.number_input("Opponent Bowling Rank (1-10)", min_value=1, max_value=10, value=5)
    venue = st.text_input("Venue", value="M. Chinnaswamy Stadium")
    pitch_type = st.selectbox("Pitch Behavior", ["Batting-friendly", "Bowling-friendly", "Balanced", "Spin-friendly"])

with col3:
    st.subheader("Player Form (Last 5 Matches)")
    player_name = st.text_input("Player Name", value="V Kohli")
    batting_position = st.number_input("Batting Position", min_value=1, max_value=11, value=3)
    avg_runs_last5 = st.number_input("Avg Runs (Last 5)", min_value=0.0, value=45.5)
    avg_balls_faced_last5 = st.number_input("Avg Balls Faced (Last 5)", min_value=0.0, value=30.0)
    
    # Auto-calculate the strike rate
    if avg_balls_faced_last5 > 0:
        calculated_strike_rate = (avg_runs_last5 / avg_balls_faced_last5) * 100
    else:
        calculated_strike_rate = 0.0
        
    # Display the calculated strike rate visually instead of as an input field
    st.metric(label="Calculated Strike Rate (Last 5)", value=f"{calculated_strike_rate:.2f}")

st.divider()

if st.button("🔮 Predict Performance", use_container_width=True):
    # Constructing the dataframe with the EXACT column names
    input_data = pd.DataFrame([{
        'player_name': player_name,
        'team': team,
        'opponent': opponent,
        'venue': venue,
        'pitch_type': pitch_type,
        'season': season,
        'matches_played_season': matches_played_season,
        'home_advantage': home_advantage,
        'toss_won': toss_won,
        'opponent_bowling_rank': opponent_bowling_rank,
        'avg_runs_last5': avg_runs_last5,
        'avg_balls_faced_last5': avg_balls_faced_last5,
        'avg_strike_rate_last5': calculated_strike_rate,  # Using the auto-calculated value here
        'batting_position': batting_position
    }])

    try:
        with st.spinner("Analyzing data..."):
            predicted_runs = model.predict(input_data)[0]
            predicted_runs = max(0, predicted_runs) # Prevent negative runs

        st.success("Prediction Generated Successfully!")
        st.metric(label=f"Projected Runs for {player_name}", value=f"{predicted_runs:.0f} Runs")
        
    except Exception as e:
        st.error(f"Error: {e}")