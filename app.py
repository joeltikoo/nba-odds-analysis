import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import poisson
from nba_api.stats.endpoints import leaguegamelog, leaguestandings
import requests

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Odds Analysis",
    page_icon="🏀",
    layout="wide"
)

st.title("NBA Odds Analysis")
st.caption("Model probabilities vs market consensus — educational purposes only")

# ── data loading ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_results():
    gamelog = leaguegamelog.LeagueGameLog(
        season="2025-26",
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T"
    )
    df = gamelog.get_data_frames()[0]
    df = df[["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "WL", "PTS"]].copy()
    home = df[df["MATCHUP"].str.contains("vs[.]")].copy()
    away = df[df["MATCHUP"].str.contains(" @ ")].copy()
    home = home[["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS", "WL"]]
    home.columns = ["game_id", "date", "home_team", "home_pts", "home_wl"]
    away = away[["GAME_ID", "TEAM_NAME", "PTS"]]
    away.columns = ["game_id", "away_team", "away_pts"]
    results = pd.merge(home, away, on="game_id")
    results["home_win"] = results["home_wl"] == "W"
    return results[["date", "home_team", "away_team", "home_pts", "away_pts", "home_win"]]

@st.cache_data(ttl=3600)
def load_standings():
    df = leaguestandings.LeagueStandings(season="2025-26").get_data_frames()[0]
    df = df[["TeamName", "TeamCity", "WINS", "LOSSES", "WinPCT", "PointsPG", "OppPointsPG"]].copy()
    df.columns = ["team_name", "team_city", "W", "L", "win_pct", "pts_scored", "pts_allowed"]
    df["team"] = df["team_city"] + " " + df["team_name"]
    df["team"] = df["team"].str.replace("*", "", regex=False)
    df["net_pts"] = df["pts_scored"] - df["pts_allowed"]
    return df[["team", "W", "L", "win_pct", "pts_scored", "pts_allowed", "net_pts"]]

@st.cache_data(ttl=3600)
def load_odds(api_key):
    response = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
    )
    return response.json()

# ── helper functions ──────────────────────────────────────────
def normalize_odds(bookmaker):
    outcomes = bookmaker["markets"][0]["outcomes"]
    raw = {o["name"]: 1 / o["price"] for o in outcomes}
    total = sum(raw.values())
    return {name: prob / total for name, prob in raw.items()}, (total - 1) * 100

def predict_win_prob(home_team, away_team, standings, home_advantage=3.0):
    home = standings[standings["team"] == home_team]
    away = standings[standings["team"] == away_team]
    if home.empty or away.empty:
        return 0.5
    diff = (home.iloc[0]["net_pts"] - away.iloc[0]["net_pts"]) + home_advantage
    return float(expit(diff * 0.15))

def build_elo(results, k=15, home_advantage=50):
    elo = {}
    for _, game in results.sort_values("date").iterrows():
        for team in [game["home_team"], game["away_team"]]:
            if team not in elo:
                elo[team] = 1500
        home_elo = elo[game["home_team"]] + home_advantage
        away_elo = elo[game["away_team"]]
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        actual = 1 if game["home_win"] else 0
        elo[game["home_team"]] += k * (actual - exp_home)
        elo[game["away_team"]] += k * ((1 - actual) - (1 - exp_home))
    return elo

# ── load data ─────────────────────────────────────────────────
with st.spinner("Loading NBA data..."):
    results = load_results()
    standings = load_standings()
    elo_ratings = build_elo(results)

# ── sidebar ───────────────────────────────────────────────────
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("The Odds API key", type="password")

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Tonight's games", "Team ratings", "Backtest"])

# ── tab 1: tonight's games ────────────────────────────────────
with tab1:
    if not api_key:
        st.info("Enter your Odds API key in the sidebar to see tonight's games.")
    else:
        with st.spinner("Fetching live odds..."):
            odds_data = load_odds(api_key)

        if not odds_data:
            st.warning("No games found right now.")
        else:
            for game in odds_data:
                home = game["home_team"]
                away = game["away_team"]

                # consensus market probability
                team_probs = {}
                for book in game["bookmakers"]:
                    probs, _ = normalize_odds(book)
                    for team, prob in probs.items():
                        team_probs.setdefault(team, []).append(prob)

                market_home = np.mean(team_probs.get(home, [0.5]))
                market_away = np.mean(team_probs.get(away, [0.5]))

                # model probability
                model_home = predict_win_prob(home, away, standings)
                model_away = 1 - model_home
                edge = model_home - market_home

                with st.container():
                    st.markdown(f"### {away} @ {home}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Model (home)", f"{model_home:.1%}")
                    col2.metric("Market (home)", f"{market_home:.1%}")
                    col3.metric("Edge", f"{edge:+.1%}",
                                delta_color="normal" if edge > 0 else "inverse")
                    col4.metric("Books", len(game["bookmakers"]))

                    if abs(edge) > 0.10:
                        st.warning(f"Large gap detected — check injuries and rest before trusting this edge.")
                    st.divider()

# ── tab 2: team ratings ───────────────────────────────────────
with tab2:
    st.subheader("Season standings + Elo ratings")

    display = standings[["team", "W", "L", "win_pct", "net_pts"]].copy()
    display["elo"] = display["team"].map(elo_ratings).round(0).astype(int)
    display = display.sort_values("elo", ascending=False).reset_index(drop=True)
    display.index += 1
    display.columns = ["Team", "W", "L", "Win %", "Net pts/g", "Elo"]
    display["Win %"] = display["Win %"].apply(lambda x: f"{x:.1%}")
    display["Net pts/g"] = display["Net pts/g"].apply(lambda x: f"{x:+.1f}")

    st.dataframe(display, use_container_width=True)

# ── tab 3: backtest ───────────────────────────────────────────
with tab3:
    st.subheader("Season backtest — flat $10 bets, model v1")

    FLAT_STAKE = 10
    SIMULATED_ODDS = 1.91
    STARTING_BANKROLL = 1000

    stats = {}
    bankroll = STARTING_BANKROLL
    history = [bankroll]
    correct = 0
    total = 0

    for _, game in results.sort_values("date").iterrows():
        for team in [game["home_team"], game["away_team"]]:
            if team not in stats:
                stats[team] = {"scored": [], "allowed": []}

        if (len(stats[game["home_team"]]["scored"]) >= 5 and
                len(stats[game["away_team"]]["scored"]) >= 5):

            home_net = np.mean(stats[game["home_team"]]["scored"]) - np.mean(stats[game["home_team"]]["allowed"])
            away_net = np.mean(stats[game["away_team"]]["scored"]) - np.mean(stats[game["away_team"]]["allowed"])
            diff = (home_net - away_net) + 3.0
            home_prob = float(expit(diff * 0.15))
            away_prob = 1 - home_prob
            model_favors_home = home_prob > 0.5

            if home_prob > 0.55:
                bet_wins = game["home_win"]
            elif away_prob > 0.55:
                bet_wins = not game["home_win"]
            else:
                history.append(bankroll)
                stats[game["home_team"]]["scored"].append(game["home_pts"])
                stats[game["home_team"]]["allowed"].append(game["away_pts"])
                stats[game["away_team"]]["scored"].append(game["away_pts"])
                stats[game["away_team"]]["allowed"].append(game["home_pts"])
                continue

            if model_favors_home == game["home_win"]:
                correct += 1
            total += 1

            if bet_wins:
                bankroll += FLAT_STAKE * (SIMULATED_ODDS - 1)
            else:
                bankroll -= FLAT_STAKE
            history.append(bankroll)

        stats[game["home_team"]]["scored"].append(game["home_pts"])
        stats[game["home_team"]]["allowed"].append(game["away_pts"])
        stats[game["away_team"]]["scored"].append(game["away_pts"])
        stats[game["away_team"]]["allowed"].append(game["home_pts"])

    roi = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Final bankroll", f"${bankroll:.0f}", f"{roi:+.1f}% ROI")
    col2.metric("Accuracy", f"{correct/total:.1%}" if total > 0 else "n/a")
    col3.metric("Bets placed", total)

    chart_df = pd.DataFrame({"Bankroll": history})
    st.line_chart(chart_df)
