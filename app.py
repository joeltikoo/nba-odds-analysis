import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import expit
import requests

st.set_page_config(
    page_title="NBA Odds Analysis",
    page_icon="🏀",
    layout="wide"
)

# ── custom css ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .game-card {
        background: #0f1117;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .game-title {
        font-size: 13px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        font-family: 'DM Mono', monospace;
    }
    .matchup {
        font-size: 22px;
        font-weight: 600;
        color: #f9fafb;
        margin-bottom: 20px;
    }
    .prob-bar-container {
        background: #1e2130;
        border-radius: 6px;
        height: 10px;
        margin: 8px 0 16px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    .stat-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: 'DM Mono', monospace;
    }
    .stat-value {
        font-size: 20px;
        font-weight: 600;
        color: #f9fafb;
        font-family: 'DM Mono', monospace;
    }
    .verdict-worth {
        background: #052e16;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 12px 16px;
        color: #86efac;
        font-size: 14px;
        font-weight: 500;
    }
    .verdict-skip {
        background: #1c1917;
        border: 1px solid #44403c;
        border-radius: 8px;
        padding: 12px 16px;
        color: #a8a29e;
        font-size: 14px;
    }
    .verdict-caution {
        background: #1c1400;
        border: 1px solid #854d0e;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fde68a;
        font-size: 14px;
    }
    .pick-tag {
        display: inline-block;
        background: #1d4ed8;
        color: #bfdbfe;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 99px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: 'DM Mono', monospace;
        margin-bottom: 8px;
    }
    .vig-tag {
        display: inline-block;
        background: #1e2130;
        color: #9ca3af;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        margin-left: 8px;
    }
    .section-header {
        font-size: 11px;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'DM Mono', monospace;
        border-bottom: 1px solid #1e2130;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    div[data-testid="metric-container"] {
        background: #0f1117;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────
def normalize_odds(bookmaker):
    outcomes = bookmaker["markets"][0]["outcomes"]
    raw = {o["name"]: 1 / o["price"] for o in outcomes}
    total = sum(raw.values())
    vig = (total - 1) * 100
    return {name: prob / total for name, prob in raw.items()}, vig

def predict_win_prob(home_team, away_team, standings, home_advantage=3.0):
    home = standings[standings["team"] == home_team]
    away = standings[standings["team"] == away_team]
    if home.empty or away.empty:
        return 0.5
    diff = (home.iloc[0]["net_pts"] - away.iloc[0]["net_pts"]) + home_advantage
    return float(expit(diff * 0.15))

def kelly_fraction(model_prob, odds_decimal, fraction=0.25):
    edge = model_prob - (1 / odds_decimal)
    kelly = edge / (odds_decimal - 1)
    return max(0, kelly * fraction)

def betting_verdict(model_prob, market_prob, best_odds, gap):
    edge = model_prob - market_prob

    if abs(gap) > 0.15:
        return "caution", "Large gap — check injuries and rest before acting on this edge."

    if edge > 0.05:
        kelly = kelly_fraction(model_prob, best_odds)
        if kelly > 0:
            return "worth", f"Model sees +{edge:.1%} edge. Kelly suggests staking {kelly:.1%} of bankroll."
        else:
            return "skip", "Edge exists but odds aren't good enough to overcome the vig."
    elif edge < -0.05:
        return "skip", f"Market is more confident than model by {abs(edge):.1%}. No edge here."
    else:
        return "skip", f"Edge is too small ({edge:+.1%}) to overcome the vig. Skip."

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


# ── data loading ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_results():
    from nba_api.stats.endpoints import leaguegamelog
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
    from nba_api.stats.endpoints import leaguestandings
    df = leaguestandings.LeagueStandings(season="2025-26").get_data_frames()[0]
    df = df[["TeamName", "TeamCity", "WINS", "LOSSES", "WinPCT", "PointsPG", "OppPointsPG"]].copy()
    df.columns = ["team_name", "team_city", "W", "L", "win_pct", "pts_scored", "pts_allowed"]
    df["team"] = df["team_city"] + " " + df["team_name"]
    df["team"] = df["team"].str.replace("*", "", regex=False)
    df["net_pts"] = df["pts_scored"] - df["pts_allowed"]
    return df[["team", "W", "L", "win_pct", "pts_scored", "pts_allowed", "net_pts"]]

@st.cache_data(ttl=1800)
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


# ── load base data ────────────────────────────────────────────
with st.spinner("Loading NBA data..."):
    results = load_results()
    standings = load_standings()
    elo_ratings = build_elo(results)

league_avg_pts = results["home_pts"].mean()


# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### NBA Odds Analysis")
    st.caption("Educational project — not financial advice")
    st.divider()
    api_key = st.text_input("Odds API key", type="password",
                             help="Get a free key at the-odds-api.com")
    st.divider()
    st.markdown("**Model used:** V1 season average net rating")
    st.markdown("**Accuracy:** 70.1% on 2025-26 season")
    st.markdown("**Vig threshold:** bets only flagged when Kelly > 0")
    st.divider()
    st.caption("Data refreshes every 30 min. Always verify injury reports before betting.")


# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Tonight's games", "Team ratings", "Backtest"])


# ── tab 1: tonight's games ────────────────────────────────────
with tab1:
    if not api_key:
        st.info("Enter your Odds API key in the sidebar to see tonight's games and betting analysis.")
    else:
        with st.spinner("Fetching live odds..."):
            odds_data = load_odds(api_key)

        if not odds_data or isinstance(odds_data, dict):
            st.warning("No games found right now — check back on a game day.")
        else:
            st.markdown(f"<div class='section-header'>{len(odds_data)} games found</div>",
                        unsafe_allow_html=True)

            for game in odds_data:
                home = game["home_team"]
                away = game["away_team"]

                # consensus market probs
                team_probs = {}
                vigs = []
                best_home_odds = 0
                best_away_odds = 0

                for book in game["bookmakers"]:
                    probs, vig = normalize_odds(book)
                    vigs.append(vig)
                    for team, prob in probs.items():
                        team_probs.setdefault(team, []).append(prob)
                    for outcome in book["markets"][0]["outcomes"]:
                        if outcome["name"] == home:
                            best_home_odds = max(best_home_odds, outcome["price"])
                        else:
                            best_away_odds = max(best_away_odds, outcome["price"])

                market_home = float(np.mean(team_probs.get(home, [0.5])))
                market_away = float(np.mean(team_probs.get(away, [0.5])))
                avg_vig = float(np.mean(vigs))
                num_books = len(game["bookmakers"])

                # model prediction
                model_home = predict_win_prob(home, away, standings)
                model_away = 1 - model_home
                gap = abs(model_home - market_home)

                # who does model pick
                model_pick = home if model_home > model_away else away
                model_pick_prob = max(model_home, model_away)

                # betting verdict for each side
                home_verdict, home_msg = betting_verdict(model_home, market_home, best_home_odds, gap)
                away_verdict, away_msg = betting_verdict(model_away, market_away, best_away_odds, gap)

                # pick the stronger verdict to show
                primary_verdict = home_verdict if model_home > model_away else away_verdict
                primary_msg = home_msg if model_home > model_away else away_msg

                with st.container():
                    st.markdown(f"""
                    <div class='game-card'>
                        <div class='game-title'>{game.get('commence_time', '')[:10]} &nbsp;·&nbsp; {num_books} books &nbsp;<span class='vig-tag'>avg vig {avg_vig:.1f}%</span></div>
                        <div class='matchup'>{away} <span style='color:#374151'>@</span> {home}</div>
                        <div class='pick-tag'>Model pick: {model_pick}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4, col5 = st.columns(5)

                    col1.markdown(f"<div class='stat-label'>Model (home)</div><div class='stat-value'>{model_home:.1%}</div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='stat-label'>Market (home)</div><div class='stat-value'>{market_home:.1%}</div>", unsafe_allow_html=True)
                    col3.markdown(f"<div class='stat-label'>Edge</div><div class='stat-value' style='color:{'#86efac' if model_home > market_home else '#f87171'}'>{model_home - market_home:+.1%}</div>", unsafe_allow_html=True)
                    col4.markdown(f"<div class='stat-label'>Best home odds</div><div class='stat-value'>{best_home_odds:.2f}</div>", unsafe_allow_html=True)
                    col5.markdown(f"<div class='stat-label'>Best away odds</div><div class='stat-value'>{best_away_odds:.2f}</div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # probability bar
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-bottom:4px;font-family:DM Mono,monospace'>
                        <span>{home} {model_home:.1%}</span>
                        <span>{away} {model_away:.1%}</span>
                    </div>
                    <div class='prob-bar-container'>
                        <div class='prob-bar-fill' style='width:{model_home*100:.1f}%;background:linear-gradient(90deg,#1d4ed8,#3b82f6)'></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # verdict
                    if primary_verdict == "worth":
                        st.markdown(f"<div class='verdict-worth'>Worth considering — {primary_msg}</div>", unsafe_allow_html=True)
                    elif primary_verdict == "caution":
                        st.markdown(f"<div class='verdict-caution'>Caution — {primary_msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='verdict-skip'>Skip — {primary_msg}</div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)


# ── tab 2: team ratings ───────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>All 30 teams — sorted by Elo</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        display = standings[["team", "W", "L", "win_pct", "pts_scored", "pts_allowed", "net_pts"]].copy()
        display["elo"] = display["team"].map(elo_ratings).round(0).astype(int)
        display = display.sort_values("elo", ascending=False).reset_index(drop=True)
        display.index += 1

        display.columns = ["Team", "W", "L", "Win %", "Pts/g", "Opp Pts/g", "Net Pts/g", "Elo"]
        display["Win %"] = display["Win %"].apply(lambda x: f"{x:.1%}")
        display["Net Pts/g"] = display["Net Pts/g"].apply(lambda x: f"{x:+.1f}")

        st.dataframe(
            display,
            use_container_width=True,
            height=600,
            column_config={
                "Elo": st.column_config.ProgressColumn(
                    "Elo", min_value=1300, max_value=1800, format="%d"
                ),
                "Net Pts/g": st.column_config.TextColumn("Net Pts/g"),
            }
        )

    with col2:
        st.markdown("<div class='section-header'>Top 5 offenses</div>", unsafe_allow_html=True)
        top_off = standings.nlargest(5, "pts_scored")[["team", "pts_scored"]]
        for _, row in top_off.iterrows():
            st.markdown(f"**{row['team']}** — {row['pts_scored']:.1f} pts/g")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Top 5 defenses</div>", unsafe_allow_html=True)
        top_def = standings.nsmallest(5, "pts_allowed")[["team", "pts_allowed"]]
        for _, row in top_def.iterrows():
            st.markdown(f"**{row['team']}** — {row['pts_allowed']:.1f} opp pts/g")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Biggest Elo gainers</div>", unsafe_allow_html=True)
        elo_series = pd.Series(elo_ratings).sort_values(ascending=False)
        for team, rating in elo_series.head(5).items():
            delta = rating - 1500
            st.markdown(f"**{team}** — {rating:.0f} ({delta:+.0f})")


# ── tab 3: backtest ───────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Season backtest — flat $10 bets, model v1</div>",
                unsafe_allow_html=True)

    FLAT_STAKE = 10
    SIMULATED_ODDS = 1.91
    STARTING_BANKROLL = 1000

    stats = {}
    bankroll = STARTING_BANKROLL
    history = [bankroll]
    correct = 0
    total = 0
    wins = 0
    losses = 0

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
                wins += 1
            else:
                bankroll -= FLAT_STAKE
                losses += 1

            history.append(bankroll)

        stats[game["home_team"]]["scored"].append(game["home_pts"])
        stats[game["home_team"]]["allowed"].append(game["away_pts"])
        stats[game["away_team"]]["scored"].append(game["away_pts"])
        stats[game["away_team"]]["allowed"].append(game["home_pts"])

    roi = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100

    # max drawdown
    peak = STARTING_BANKROLL
    max_dd = 0
    for val in history:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Starting bankroll", f"${STARTING_BANKROLL:,.0f}")
    col2.metric("Final bankroll", f"${bankroll:,.0f}", f"{roi:+.1f}%")
    col3.metric("Accuracy", f"{correct/total:.1%}" if total > 0 else "n/a")
    col4.metric("Max drawdown", f"{max_dd:.1f}%")
    col5.metric("Bets placed", f"{total} ({wins}W / {losses}L)")

    st.markdown("<br>", unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Bankroll ($)": history,
        "Starting bankroll": [STARTING_BANKROLL] * len(history)
    })
    st.line_chart(chart_df, color=["#3b82f6", "#374151"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model leaderboard</div>", unsafe_allow_html=True)

    leaderboard = pd.DataFrame([
        {"Model": "V1 season average", "Accuracy": "70.1%", "Notes": "Simplest, most accurate"},
        {"Model": "V3 Poisson", "Accuracy": "69.3%", "Notes": "Useful for score prediction"},
        {"Model": "V4 Elo (tuned)", "Accuracy": "66.4%", "Notes": "Dynamic but noisy"},
        {"Model": "V2 recent form", "Accuracy": "65.4%", "Notes": "Too sensitive to hot/cold streaks"},
        {"Model": "Baseline (always home)", "Accuracy": "54.9%", "Notes": "No model"},
    ])
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.caption("Simulated odds of 1.91 (-110) used for all bets. This is a backsimulation only — past performance does not guarantee future results.")
