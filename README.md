# Media Mix Studio

A marketing mix model (MMM) and budget optimiser, built as a Streamlit app. It estimates how much revenue each media channel drives, how quickly each channel saturates, and how to split a fixed budget for the most revenue.

## What it does

- **Model fit:** geometric adstock (carryover) and Hill saturation per channel, with ridge regression and media coefficients kept at zero or above. Carryover and saturation settings are found by a random-plus-local search scored on held-out weeks, with Robyn's DECOMP.RSSD penalty against implausible splits.
- **Model averaging:** the best fit is averaged with its 25 nearest competitors. Single fits proved unstable across random seeds; the averaged response curves gave identical recommendations across seeds.
- **Channel returns:** average ROI versus return on the next unit of spend, with the spread across models, response curves and carryover half-lives.
- **Scenario planner:** set spend per channel, see the revenue impact, and save and compare scenarios.
- **Budget optimiser:** SLSQP with multiple starts, per-channel floors, ceilings and locks, a budget-versus-revenue frontier, and a count of how many models agree with each move.
- **Your own data:** upload a weekly CSV and map its columns in the app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests   # optional: engine tests
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository, with `app.py` at the root.
2. At share.streamlit.io, choose **Create app**, pick the repository and branch, and set the main file to `app.py`.
3. Deploy. The first model fit takes a few seconds; after that it is cached.

## Project layout

```
app.py              navigation and page setup
mmm/                modelling engine (no Streamlit dependency)
  transforms.py     adstock, Hill curve, derivatives
  data.py           config, validation, feature preparation, VIF
  model.py          search, bounded ridge, decomposition, ensemble
  optimiser.py      budget allocation, frontier, robustness check
ui/                 theme, formatting, session state and caching
views/              one file per page
tests/              engine tests, including recovery of known effects
data/               Robyn simulated weekly dataset (Meta, MIT licence)
```

## Limits

- Results are correlational. Spend that follows demand, such as search budgets tracking seasonal interest, gets credit it may not deserve. Calibrate with geo-lift or holdout tests.
- Plans assume steady weekly spend, and curves beyond the highest spend in the data are extrapolations.
- Returns are in revenue, not profit. Compare them with 1 ÷ gross margin.

## Data

The sample data is `dt_simulated_weekly` from Meta's open-source [Robyn](https://github.com/facebookexperimental/Robyn) package. It is simulated, and its units are not real currency.
