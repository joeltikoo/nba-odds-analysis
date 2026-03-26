# NBA Odds Analysis

A little data science project that models NBA game probabilities and compares them against bookmaker odds to identify market inefficiencies.

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
sports_odds_analysis.ipynb   # main notebook
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
- Scrape current NBA standings from basketball-reference.com
- Build model v1 using season-average net rating
- Build model v2 using recent form (last 10 games)
- Compare both models against market consensus
- Flag large gaps between model and market for manual review

### More upcoming

---

## Key concepts covered

- Implied probability and vig removal
- Market consensus and bookmaker spread
- Net rating as a proxy for team strength
- Home court advantage modeling
- Sigmoid function for probability estimation
- Recent form vs season averages
- Why large model-market gaps signal missing context (injuries, rest, back-to-backs)

---

## Data sources

- [The Odds API](https://the-odds-api.com) — live and historical bookmaker odds
- [Basketball Reference](https://www.basketball-reference.com) — NBA standings and game logs

---

## Setup

```bash
# in Google Colab or locally
pip install requests pandas matplotlib scipy
```

Get a free API key at [the-odds-api.com](https://the-odds-api.com) and add it to the notebook.

---

## Important disclaimer

This project is for educational purposes only. Sports betting involves real financial risk. The models built here are simple and should not be used as the basis for actual betting decisions.

---

## License

MIT License — see LICENSE file.
