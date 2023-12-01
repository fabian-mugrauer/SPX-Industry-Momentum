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
from src.data.load_data import Dataloader
from src.data.process_data import Dataprocessor

# Create instances of the imported classes
strategy_evaluator = EvaluateStrategy()
momentum = MomentumStrategy()
visualizer = Visualizer()
dataloader = Dataloader()
dataprocessor = Dataprocessor()

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

# Decorate the load_data functions with joblib cache
load_data_csv_cached = memory.cache(dataloader.load_data_csv)
load_data_bb_cached = memory.cache(dataloader.load_data_BB)

# Define to load with BB or with csv (BB = False = csv)
BB = False

# Define data to load with BB
start_date = "1989-09-11"   
end_date = "2023-11-24" 
ticker_index = [
'SPXT Index', 'S5INFT Index', 'S5TELS Index', 'S5CONS Index', 
'S5COND Index', 'S5ENRS Index', 'S5FINL Index', 'S5HLTH Index', 
'S5INDU Index', 'S5MATR Index', 'S5RLST Index', 'S5UTIL Index', 
'S5SFTW Index', 'S5TECH Index', 'S5MEDA Index', 
'S5SSEQX Index', 'S5DIVF Index', 'S5PHRM Index', 'S5RETL Index', 
'S5CPGS Index', 'S5HCES Index', 'S5ENRSX Index', 'S5FDBT Index', 
'S5BANKX Index', 'S5MATRX Index', 'S5UTILX Index', 'S5REAL Index', 
'S5INSU Index', 'S5HOTR Index', 'S5AUCO Index', 'S5FDSR Index', 
'S5HOUS Index', 'S5TRAN Index', 'S5COMS Index', 'S5CODU Index', 
'S5TELSX Index' 
]

ticker_rf = ['GB03 Govt']

field_index = ['TOT_RETURN_INDEX_GROSS_DVDS']
field_rf = ['PX_LAST']

# Define csv file to load
file_name = 'Bloomberg_Download.csv'

# Load the data using the cached functions
if BB:
    dates_dateformat, SPXT, Sectors, Rf, Industry_Groups = load_data_bb_cached(start_date, end_date, ticker_index, ticker_rf, field_index, field_rf)
else:
    dates_dateformat, SPXT, Sectors, Rf, Industry_Groups = load_data_csv_cached(file_path, file_name)

# Decorate the load_data functions with joblib cache
process_data_cached = memory.cache(dataprocessor.process_data)

# Process the data using the cached functions
dates_datetime, numericDate_d, firstDayList, lastDayList, dates4plot, Sectors_returns_d, Sectors_returns_m, sector_names, Industry_Groups_returns_d, Industry_Groups_returns_m, IG_names, SPXT_returns_d, SPXT_returns_m, SPXT_Xsreturns_m, rf_d_unadjusted, rf_d, rf_d_monthly = process_data_cached(dates_dateformat, Sectors, Industry_Groups, SPXT, Rf)


# %% [markdown]
# Define Meta-variables for Analysis

# %%
# Define analysis parameters
trx_cost = 0.001
nLong = 3
startMonth = 13
lookback_period_start, lookback_period_end = 9, 1
holding_period = 3

# %% [markdown]
# Run Sector Momentum Analysis

# %%
# Perform the momentum strategy backtest
xsReturns_TC, totalReturns_TC, weights = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, 0, trx_cost)
xsReturns_TC_IG, totalReturns_TC_IG, weights_IG = momentum.backtest_momentum(Industry_Groups_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, 0, trx_cost)

# Long / Short implementation
xsReturns_TC_LS, totalReturns_TC_LS, weights_LS = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, 3, trx_cost)
xsReturns_TC_IG_LS, totalReturns_TC_IG_LS, weights_IG_LS = momentum.backtest_momentum(Industry_Groups_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, 3, trx_cost)

# %% [markdown]
# Analysis Output

# %%
# Visualize the weights using a heatmap

fig1 = visualizer.create_colorheatmap(weights, weights_IG, sector_names, IG_names)
fig1.savefig(os.path.join(picture_path, 'strategy_weights_long.png'), dpi=300)

# Long/Short
fig2 = visualizer.create_colorheatmap(weights_LS, weights_IG_LS, sector_names, IG_names)
fig2.savefig(os.path.join(picture_path, 'strategy_weights_long_short.png'), dpi=300)

# Visualize strategy returns against the benchmark
fig3 = visualizer.plot_strategies_with_benchmark(dates4plot, totalReturns_TC, SPXT_returns_m, 'Sector Momentum', startMonth, totalReturns_TC_IG, 'Industry Group Momentum')
fig3.savefig(os.path.join(picture_path, 'strategy_plot.png'), dpi=300)

fig4 = visualizer.plot_strategies_with_benchmark(dates4plot, totalReturns_TC_LS, rf_d_monthly, 'Sector Momentum Long/Short', startMonth, totalReturns_TC_IG_LS, 'Industry Group Momentum Long/Short')
fig4.savefig(os.path.join(picture_path, 'strategy_plot_long_short.png'), dpi=300)

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
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

sharpe_ratios_lookback_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'lookback', lookback_period_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

# Plot the results for lookback period robustness check
fig5 = visualizer.plot_robustness_check(lookback_period_range, sharpe_ratios_lookback, 'lookback', sharpe_ratios_lookback_IG, ['Sectors', 'Industry Groups']) 
fig5.savefig(os.path.join(picture_path, 'robustness_check_lb.png'), dpi=300)

# %%
# Define the range for the investment horizon
investment_horizon_range = range(1, 6)  # Example range

# Perform the robustness check for investment horizon
sharpe_ratios_investment_horizon = EvaluateStrategy.perform_robustness_check(
    momentum, 'investment_horizon', investment_horizon_range, Sectors_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

sharpe_ratios_investment_horizon_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'investment_horizon', investment_horizon_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

# Plot the results for investment horizon robustness check
fig6 = visualizer.plot_robustness_check(investment_horizon_range, sharpe_ratios_investment_horizon, 'investment_horizon', sharpe_ratios_investment_horizon_IG, ['Sectors', 'Industry Groups'])
fig6.savefig(os.path.join(picture_path, 'robustness_check_ih.png'), dpi=300)

# %%
# Define the range for the number of holdings
holdings_range = range(1, 6)  # Example range

# Perform the robustness check for number of holdings
sharpe_ratios_holdings = EvaluateStrategy.perform_robustness_check(
    momentum, 'holdings', holdings_range, Sectors_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

sharpe_ratios_holdings_IG = EvaluateStrategy.perform_robustness_check(
    momentum, 'holdings', holdings_range, Industry_Groups_returns_m, rf_d_monthly, 
    SPXT_returns_m, lookback_period_end, startMonth, nLong, 0, trx_cost, holding_period
)

# Plot the results for number of holdings robustness check
fig7 = visualizer.plot_robustness_check(holdings_range, sharpe_ratios_holdings, 'holdings', sharpe_ratios_holdings_IG,['Sectors', 'Industry Groups'])
fig7.savefig(os.path.join(picture_path, 'robustness_check_holdings.png'), dpi=300)

# %%
tx_costs = [0.001, 0.0025, 0.005, 0.01]
total_returns_tx_costs = []
total_returns_tx_costs_IG  = []
labels = []

for trx_cost in tx_costs:
    _, totalReturns_TC, _ = momentum.backtest_momentum(
        Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end,
        holding_period, startMonth, nLong, 0, trx_cost
    )

    _, totalReturns_TC_IG, _ = momentum.backtest_momentum(
        Industry_Groups_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end,
        holding_period, startMonth, nLong, 0, trx_cost
    )

    # Storing the total returns in a list
    total_returns_tx_costs.append(totalReturns_TC)
    total_returns_tx_costs_IG.append(totalReturns_TC_IG)

    # Creating corresponding labels
    labels.append(f'Trx Cost: {trx_cost}')

fig8 = visualizer.plot_strategies_with_benchmark(dates4plot, total_returns_tx_costs_IG, SPXT_returns_m, labels, startMonth)
fig8.savefig(os.path.join(picture_path, 'robustness_check_tc.png'), dpi=300)