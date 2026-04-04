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
app.py                          # Streamlit dashboard (upcoming)
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
