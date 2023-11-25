# %% [markdown]
# Import packages

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from prettytable import PrettyTable
import os
import sys
from dotenv import load_dotenv
from joblib import Memory

# Load environment variables from a .env file
load_dotenv(dotenv_path='environment_variables.env')

# Setup joblib Memory for caching
cachedir = './cache_directory'  # Set your cache directory
memory = Memory(cachedir, verbose=0)

# %% [markdown]
# Import Custom Classes

# %%
# Add custom directories to the system path for importing classes
sys.path.extend(['../src/data', '../src/models', '../src/visualization'])

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

# Import custom classes
from src.models.evaluate_strategy import EvaluateStrategy
from src.models.momentum_strategy import MomentumStrategy
from src.visualization.visualize_results import Visualizer

# Create instances of the imported classes
strategy_evaluator = EvaluateStrategy()
momentum = MomentumStrategy()
visualizer = Visualizer()

# %% [markdown]
# Set Environment Variable

# %%
# Set and retrieve the project path
ProjectName = 'SPX-Industry-Momentum'
os.environ['RESEARCH_PATH'] = os.environ.get('PROJECT_ROOT')
research_path = os.environ.get('RESEARCH_PATH')

# Check if the path is set and print it
if research_path:
    path = os.path.join(research_path, ProjectName)
    file_path = os.path.join(path, 'data', 'raw')
    picture_path = os.path.join(path, 'reports', 'figures')
    print(f"Project Path: {path}\nData Path: {file_path}")
else:
    print("RESEARCH_PATH environment variable is not set.")

# %% [markdown]
# Import and Process Data

# %%
# Import functions for data loading and processing
from load_data import load_data
from process_data import process_data

# Decorate the load_data and process_data functions with joblib cache
load_data_cached = memory.cache(load_data)
process_data_cached = memory.cache(process_data)

# Load and process the data using the cached functions
file_name = 'Bloomberg_Download.csv'
dates_dateformat, SPXT, Sectors, Rf, Industry_Groups = load_data_cached(file_path, file_name)
dates_datetime, numericDate_d, firstDayList, lastDayList, dates4plot, Sectors_returns_d, Sectors_returns_m, Industry_Groups_returns_d, Industry_Groups_returns_m, SPXT_returns_d, SPXT_returns_m, SPXT_Xsreturns_m, rf_d_unadjusted, rf_d, rf_d_monthly = process_data_cached(dates_dateformat, Sectors, Industry_Groups, SPXT, Rf)

# %% [markdown]
# Define Meta-variables for Analysis

# %%
# Define analysis parameters
trx_cost = 0.001
nLong, nShort = 3, 0
startMonth = 13
lookback_period_start, lookback_period_end = 9, 1
holding_period = 3

# %% [markdown]
# Run Sector Momentum Analysis

# %%
# Perform the momentum strategy backtest
xsReturns_TC, totalReturns_TC, weights = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, nShort, trx_cost)
xsReturns_TC_IG, totalReturns_TC_IG, weights_IG = momentum.backtest_momentum(Industry_Groups_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, nShort, trx_cost)

# %% [markdown]
# Analysis Output

# %%
# Visualize the weights using a heatmap
visualizer.create_colorheatmap(weights, weights_IG)

# Visualize strategy returns against the benchmark
fig = visualizer.plot_strategies_with_benchmark(dates4plot, totalReturns_TC, SPXT_returns_m, 'Sector Momentum', startMonth, totalReturns_TC_IG, 'Industry Group Momentum')
fig.savefig(os.path.join(picture_path, 'strategy_plot.png'), dpi=300)


# %%
# Summarize and print performance metrics
ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic, MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn, alphaArithmetic, tvaluealpha, betas, summary_table = strategy_evaluator.summarize_performance(xsReturns_TC, rf_d_monthly, SPXT_Xsreturns_m, 12, startMonth, 'Sectors')
ArithmAvgTotalReturn_IG, ArithmAvgXsReturn_IG, StdXsReturns_IG, SharpeArithmetic_IG, MinXsReturn_IG, MaxXsReturn_IG, SkewXsReturn_IG, KurtXsReturn_IG, alphaArithmetic_IG, tvaluealpha_IG, betas_IG, summary_table_IG = strategy_evaluator.summarize_performance(xsReturns_TC_IG, rf_d_monthly, SPXT_Xsreturns_m, 12, startMonth, 'Industry Groups')
print(summary_table)
print(summary_table_IG)

# %% [markdown]
# Robustness Checks

# %%
# Define parameters and initialize empty lists or arrays
lookback_period_range = range(3, 13)  # Example range from 3 to 12

# Perform the robustness check for lookback period
sharpe_ratios_lookback = EvaluateStrategy.perform_robustness_check(
    momentum, 'lookback', lookback_period_range, Sectors_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

sharpe_ratios_lookback_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'lookback', lookback_period_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

# Plot the results for lookback period robustness check
visualizer.plot_robustness_check(lookback_period_range, sharpe_ratios_lookback, 'lookback', sharpe_ratios_lookback_IG, ['Sectors', 'Industry Groups']) 

# %%
# Define the range for the investment horizon
investment_horizon_range = range(1, 6)  # Example range

# Perform the robustness check for investment horizon
sharpe_ratios_investment_horizon = EvaluateStrategy.perform_robustness_check(
    momentum, 'investment_horizon', investment_horizon_range, Sectors_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

sharpe_ratios_investment_horizon_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'investment_horizon', investment_horizon_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

# Plot the results for investment horizon robustness check
visualizer.plot_robustness_check(investment_horizon_range, sharpe_ratios_investment_horizon, 'investment_horizon', sharpe_ratios_investment_horizon_IG, ['Sectors', 'Industry Groups'])

# %%
# Define the range for the number of holdings
holdings_range = range(1, 6)  # Example range

# Perform the robustness check for number of holdings
sharpe_ratios_holdings = EvaluateStrategy.perform_robustness_check(
    momentum, 'holdings', holdings_range, Sectors_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

sharpe_ratios_holdings_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'holdings', holdings_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period
)

# Plot the results for number of holdings robustness check
visualizer.plot_robustness_check(holdings_range, sharpe_ratios_holdings, 'holdings', sharpe_ratios_holdings_IG,['Sectors', 'Industry Groups'])

# %%
tx_costs = [0.001, 0.0025, 0.005, 0.01]
total_returns_tx_costs = []
total_returns_tx_costs_IG  = []
labels = []

for trx_cost in tx_costs:
    _, totalReturns_TC, _ = momentum.backtest_momentum(
        Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end,
        holding_period, startMonth, nLong, nShort, trx_cost
    )

    _, totalReturns_TC_IG, _ = momentum.backtest_momentum(
        Industry_Groups_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end,
        holding_period, startMonth, nLong, nShort, trx_cost
    )

    # Storing the total returns in a list
    total_returns_tx_costs.append(totalReturns_TC)
    total_returns_tx_costs_IG.append(totalReturns_TC_IG)

    # Creating corresponding labels
    labels.append(f'Trx Cost: {trx_cost}')

visualizer.plot_strategies_with_benchmark(dates4plot, total_returns_tx_costs_IG, SPXT_returns_m, labels, startMonth)