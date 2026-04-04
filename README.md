# NBA Odds Analysis

A data science project that models NBA game probabilities and compares them against bookmaker odds to identify market inefficiencies.

No real money is involved — this is purely a data science and modeling exercise.

---

## What this project does

Bookmakers set odds that imply a probability for each game outcome. Those implied probabilities always sum to more than 100% — the excess is the bookmaker's margin (the "vig"). This project:

1. Pulls live NBA odds from multiple bookmakers via the Odds API
2. Strips out the vig to get clean implied probabilities
3. Computes a consensus probability across all books
4. Builds independent models to estimate true win probabilities
5. Compares model estimates against the market to find edges

---

## Project structure

```
1\sports_odds_analysis.ipynb    # Day 1 (Stage 1 & 2)
2\nba_odds_analysis.ipynb       # Day 2 (Stage 3 & 4)
3\nba_odds_analysis_3.ipynb     # Day 3 (Stage 5)
app.py                          # Streamlit dashboard
README.md
```

---

## Stages

### Stage 1 — Odds collection & vig removal (complete)
- Pull live NBA odds from The Odds API across 9+ bookmakers
- Convert decimal odds to implied probabilities
- Strip the vig and normalize to clean probabilities
- Compute consensus probability and bookmaker spread
- Visualize tonight's games with team colors

### Stage 2 — Team stats & win probability models (complete)
- Pull current NBA standings via nba_api
- Build model v1 using season-average net rating
- Build model v2 using recent form (last 10 games)
- Compare both models against market consensus
- Flag large gaps between model and market for manual review

### Stage 3 — Backtesting & betting simulation (complete)
- Built rolling standings to avoid look-ahead bias
- Simulated a full season of flat bets ($10 per game)
- Tracked ROI, max drawdown, and losing streaks
- Result: 70.1% accuracy, 305.9% ROI, 4.3% max drawdown

### Stage 4 — Poisson score modeling (complete)
- Built attack and defense ratings for all 30 teams
- Modeled expected scores using a Poisson distribution
- Backtested against V1 and V2
- Result: 69.3% accuracy — more complex but not more accurate than V1

### Stage 5 — Bayesian Elo ratings (complete)
- Built a dynamic Elo rating system that updates after every game
- Tuned K factor and home advantage across 30 parameter combinations
- Result: 66.4% accuracy (best at K=15, home advantage=50)

---

## Model leaderboard

| Model | Accuracy | Notes |
|-------|----------|-------|
| V1 season average | 70.1% | simplest, most accurate |
| V3 Poisson | 69.3% | useful for score prediction |
| V4 Elo (tuned) | 66.4% | dynamic but noisy |
| V2 recent form | 65.4% | too sensitive to hot/cold streaks |
| Baseline (always home) | 54.9% | |

Key finding: Occam's Razor holds. The simplest model (season-average net rating) outperforms all more complex approaches. In the NBA, 82-game averages are stable enough that dynamic updates add noise rather than signal.

---

## Streamlit Dashboard

A live Streamlit dashboard brings the full pipeline into a single interface. It pulls live odds and NBA data automatically and refreshes every 30 minutes.

- Tonight's games — for each game on the schedule, the dashboard shows the model's predicted win probability alongside the market consensus, highlights the edge, displays the best available odds across all books, and gives a plain-English verdict: worth considering, skip, or caution. The caution flag triggers automatically when the model-market gap exceeds 15%, which almost always indicates injuries or missing roster context.
- Team ratings — a full 30-team table sorted by Elo rating, with net points per game, win percentage, and a progress bar for Elo strength. A side panel surfaces the top five offenses, defenses, and biggest Elo movers of the season.
- Backtest — the full season simulation with a bankroll curve, accuracy, ROI, max drawdown, and win/loss count. The model leaderboard is shown at the bottom for context.

1. Go to ![https://nbaoddsanalysis.streamlit.app](https://nbaoddsanalysis.streamlit.app)
2. Get your Odds API key from ![https://the-odds-api.com](https://the-odds-api.com) and enter it in the dashboard.

<img width="888" height="678" alt="image" src="https://github.com/user-attachments/assets/184aeffb-7218-4d3b-8ca1-4ee47467bacd" />
<img width="1424" height="673" alt="image" src="https://github.com/user-attachments/assets/46c4109e-43dc-47ce-bded-cc676fbcde47" />
<img width="1404" height="373" alt="image" src="https://github.com/user-attachments/assets/32599b58-ef70-4b5d-98f8-e3499bfd67ab" />
<img width="1592" height="782" alt="image" src="https://github.com/user-attachments/assets/6e0cea9c-ee30-4dc2-a521-135c37ca7411" />


## Key concepts covered

- Implied probability and vig removal
- Market consensus and bookmaker spread
- Net rating as a proxy for team strength
- Home court advantage modeling
- Sigmoid function for probability estimation
- Recent form vs season averages
- Look-ahead bias and rolling backtests
- Kelly Criterion for bet sizing
- Poisson distribution for score modeling
- Elo rating systems and K factor tuning
- Occam's Razor in model selection
- Why large model-market gaps signal missing context (injuries, rest, back-to-backs)

---

## Data sources

- [The Odds API](https://the-odds-api.com) — live and historical bookmaker odds
- [nba_api](https://github.com/swar/nba_api) — NBA standings and game logs via the official NBA stats API

---

## Setup

```bash
pip install requests pandas matplotlib scipy nba_api
```

Get a free API key at [the-odds-api.com](https://the-odds-api.com) and add it to the notebook.

---

## Important disclaimer

This project is for educational purposes only. Sports betting involves real financial risk. The models built here are simple and should not be used as the basis for actual betting decisions.

---

## License

MIT License — see LICENSE file.
