# Bitcoin Volatility Forecasting: A Risk-Aware Time Series Analysis

Volatility modeling, Value-at-Risk backtesting, and direction prediction on 11+ years of Bitcoin daily price data (Sept 2014–2026).

## Problem Statement

Bitcoin is notoriously volatile, but most public "Bitcoin prediction" projects focus narrowly on price direction using simple ML models. This project instead asks two more rigorous questions:
1. Can Bitcoin's volatility (not price) be modeled and forecasted reliably?
2. Can a forecasted volatility model produce a usable risk estimate (Value-at-Risk) for downside risk management?
3. As a comparison, can engineered technical features actually predict next-day direction?

## Dataset

Daily OHLCV Bitcoin price data, Sept 17, 2014 – June 2026 (~4,280 trading days). Verified continuous with zero missing values and no date gaps, eliminating the need for imputation.

## Methodology

Phase 1 — EDA: Confirmed price is non-stationary (ADF p=0.58) while log returns are stationary (ADF p≈0.000) — establishing returns/volatility as the correct modeling target. Identified clear volatility clustering and a long-term structural decline in volatility as the market matured.

Phase 2 — Feature Engineering: Built RSI, MACD, Bollinger Bands, lagged returns, and rolling skew/kurtosis. Confirmed excess kurtosis (fat tails) in returns, motivating a Student-t distribution over a Gaussian assumption.

Phase 3 — GARCH(1,1) Volatility Modeling: Fit a GARCH(1,1) model with Student-t innovations. Found extremely high volatility persistence (α+β ≈ 0.999) and low degrees of freedom (ν ≈ 3.18), confirming heavy-tailed return behavior. Built both a static 90-day forecast and a more realistic rolling 1-step-ahead forecast.

Phase 4 — Value-at-Risk (VaR): Used the fitted GARCH + Student-t model to compute 95% 1-day VaR, backtested against 90 days of out-of-sample returns.

Phase 5 — ML Direction Classification: Trained Random Forest and Logistic Regression classifiers on engineered features to predict next-day up/down direction, as a comparison against the volatility-focused approach above.

## Key Findings

- Volatility is highly persistent: GARCH(1,1) α+β ≈ 0.999 — once Bitcoin gets volatile, it stays volatile for an extended period before reverting.
- Returns are heavy-tailed: Student-t degrees of freedom ν ≈ 3.18, confirming that extreme moves are far more common than a normal distribution would predict.
- The VaR model is well-calibrated: backtested 95% VaR produced a 4.4% breach rate (4 breaches out of 90 days) — closely matching the theoretical 5% expectation.
- Direction is not predictable from technical indicators at this timeframe: both Random Forest (46.9% accuracy) and Logistic Regression (48.2% accuracy) failed to beat the 50% naive baseline. This consistent result across two model families supports weak-form market efficiency.
- The central insight of this project: volatility (magnitude) is forecastable; direction is not. This asymmetry is well-documented in financial econometrics literature and is borne out clearly here.

## Limitations

- A single static long-horizon GARCH forecast smooths out short-term spikes; the rolling 1-step forecast is more realistic but computationally heavier.
- VaR backtesting was done on only 90 out-of-sample days — a longer backtest window would give more statistical confidence in the calibration result.
- ML classification used only price-derived technical features; no external data (news sentiment, on-chain metrics, macro indicators) was incorporated, which may explain the lack of directional signal.

## Tech Stack

Python · pandas · NumPy · statsmodels · arch · scikit-learn · matplotlib · seaborn · Jupyter

## Project Structure

bitcoin-volatility-forecasting/
├── data/
│   └── bitcoin_dataset.csv
├── notebooks/
│   └── 01_edabitcoin.ipynb
├── .gitignore
└── README.md

## How to Run

'''bash
git clone https://github.com/MADEKUOROYE/bitcoin-volatility-forecasting.git
cd bitcoin-volatility-forecasting
pip install pandas numpy statsmodels arch scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/01_edabitcoin.ipynb'''

# Author
Moses Adekuoroye — LinkedIn (https://www.linkedin.com/in/madekuoroye) · GitHub (https://github.com/MADEKUOROYE)
