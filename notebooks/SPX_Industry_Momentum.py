# %% [markdown]
# Import packagels s

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

# Specify the custom .env file
load_dotenv(dotenv_path='environment_variables.env')

# %% [markdown]
# Import self written classes

# %%
# Add the src/data directory to the system path
sys.path.append('../src/data')
sys.path.append('../src/models')
sys.path.append('../src/visualization')

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
parent_dir

# import functions 
from src.models.financial_data_processor import FinancialDataProcessor
data_processor = FinancialDataProcessor()

from src.models.momentum_strategy import MomentumStrategy
momentum = MomentumStrategy()

from src.visualization.graphs import graphs
visualization = graphs()


# %% [markdown]
# Set the env variable

# %%
ProjectName = 'SPX-Industry-Momentum'

os.environ['RESEARCH_PATH']  = os.environ.get('PROJECT_ROOT')
research_path = os.environ.get('RESEARCH_PATH')
if research_path:
    path = os.path.join(research_path, ProjectName)
    print(path)
    file_path = os.path.join(path, 'data', 'raw')
    print(file_path)
else:
    print("RESEARCH_PATH environment variable is not set.")

# %% [markdown]
# # Define all the functions

# %%
def summarizePerformance(xsReturns, Rf, factorXsReturns, annualizationFactor, startM, txt):

    # Shorten Series
    xsReturns = xsReturns[startM:]
    factorXsReturns = factorXsReturns[startM:]
    Rf = Rf[startM:]
    if len(xsReturns.shape) == 1:
      nPeriods = len(xsReturns)
      nAssets = 1
      xsReturns = xsReturns[:, np.newaxis]  # Convert to 2D for consistency
    else:
      nPeriods, nAssets = xsReturns.shape
    totalReturns = xsReturns + np.outer(Rf, np.ones(nAssets))

    # Compute the terminal value of the portfolios to get the geometric mean return per period
    FinalPfValRf = np.prod(1 + Rf)
    FinalPfValTotalRet = np.prod(1 + totalReturns, axis=0)
    GeomAvgRf = 100 * (FinalPfValRf ** (annualizationFactor / nPeriods) - 1)
    GeomAvgTotalReturn = 100 * (FinalPfValTotalRet ** (annualizationFactor / nPeriods) - 1)
    GeomAvgXsReturn = GeomAvgTotalReturn - GeomAvgRf

    # Regress returns on benchmark to get alpha and factor exposures
    X = np.column_stack((np.ones(nPeriods), factorXsReturns))
    betas, _, _, _ = np.linalg.lstsq(X, xsReturns, rcond=None)
    betas = betas[1:]

    bmRet = factorXsReturns.dot(betas) + np.outer(Rf, np.ones(nAssets))
    FinalPfValBm = np.prod(1 + bmRet, axis=0)
    GeomAvgBmReturn = 100 * (FinalPfValBm ** (annualizationFactor / nPeriods) - 1)
    alphaGeometric = GeomAvgTotalReturn - GeomAvgBmReturn

    alpha_monthly_timeseries = totalReturns - factorXsReturns.dot(np.ones((nAssets, nAssets)))
    GeoAlphaDeannualized = (1 + (alphaGeometric / 100)) ** (1 / annualizationFactor) - 1
    std_alpha_monthly = np.std(alpha_monthly_timeseries, axis=0)
    SE_alpha = std_alpha_monthly / np.sqrt(nPeriods)
    tvaluealpha = GeoAlphaDeannualized / SE_alpha

    # Rescale the returns to be in percentage points
    xsReturns = 100 * xsReturns
    totalReturns = 100 * totalReturns

    # Compute first three autocorrelations
    AC1 = np.diag(np.corrcoef(xsReturns[:-1].T, xsReturns[1:].T)[nAssets:2*nAssets, :nAssets])
    AC2 = np.diag(np.corrcoef(xsReturns[:-2].T, xsReturns[2:].T)[nAssets:2*nAssets, :nAssets])
    AC3 = np.diag(np.corrcoef(xsReturns[:-3].T, xsReturns[3:].T)[nAssets:2*nAssets, :nAssets])

    # Report the statistics
    ArithmAvgTotalReturn = annualizationFactor * np.mean(totalReturns, axis=0)
    ArithmAvgXsReturn = annualizationFactor * np.mean(xsReturns, axis=0)
    StdXsReturns = np.sqrt(annualizationFactor) * np.std(xsReturns, axis=0)
    SharpeArithmetic = ArithmAvgXsReturn / StdXsReturns
    SharpeGeometric = GeomAvgXsReturn / StdXsReturns
    MinXsReturn = np.min(xsReturns, axis=0)
    MaxXsReturn = np.max(xsReturns, axis=0)
    SkewXsReturn = skew(xsReturns, axis=0)
    KurtXsReturn = kurtosis(xsReturns, axis=0)
    alphaArithmetic = annualizationFactor * 100 * betas[0]

    # Create table
    x = PrettyTable()
    x.field_names = ["Statistic", "Value"]

    x.add_row(["ArithmAvgTotalReturn", ArithmAvgTotalReturn])
    x.add_row(["ArithmAvgXsReturn", ArithmAvgXsReturn])
    x.add_row(["StdXsReturns", StdXsReturns])
    x.add_row(["SharpeArithmetic", SharpeArithmetic])
    x.add_row(["SharpeGeometric", SharpeGeometric])
    x.add_row(["MinXsReturn", MinXsReturn])
    x.add_row(["MaxXsReturn", MaxXsReturn])
    x.add_row(["SkewXsReturn", SkewXsReturn])
    x.add_row(["KurtXsReturn", KurtXsReturn])
    x.add_row(["alphaArithmetic", alphaArithmetic])

    return (ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic, MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn, alphaArithmetic, tvaluealpha, betas, x)

# %% [markdown]
# Import the Data

# %%
# Import the Function to load the Data
from load_data import load_data

# Define which File to Load
file_name = 'Bloomberg_Download.csv'

# Load the Data
dates_dateformat, SPXT, Sectors, Rf, Industry_Groups = load_data(file_path, file_name)

# %% [markdown]
# Process the Data

# %%
# Import the Function to process the Data
from process_data import process_data

# Process the Data
dates_datetime, numericDate_d, firstDayList, lastDayList, dates4plot, Sectors_returns_d, Sectors_returns_m, Industry_Groups_returns_d, Industry_Groups_returns_m, SPXT_returns_d, SPXT_returns_m, rf_d_unadjusted, rf_d, rf_d_monthly = process_data(dates_dateformat, Sectors, Industry_Groups, SPXT, Rf)

# %% [markdown]
# Define Metavariables for Analysis

# %%
trx_cost = 0.001;
nLong = 3;
nShort = 0;
startMonth = 101;
tradingLag = 0;
lookback_period_start = 9
lookback_period_end = 1
holding_period = 3
trading_lag = 0
myLineWidth = 2
RobustnessCheckColor = 'blue'

# %% [markdown]
# Run Sector Momentum Analysis

# %%
xsReturns_TC, totalReturns_TC, weights = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, nShort, trx_cost)

# %% [markdown]
# # Analysis Output

# %% [markdown]
# Heatmap Analysis of Weights

# %%
visualization.create_colorheatmap(weights)

# %% [markdown]
# Return Analysis

# %%
visualization.plot_strategies_with_benchmark(dates4plot, totalReturns_TC, SPXT_returns_m, 'Momentum Strategy', startMonth)

# %%
ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic, MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn, alphaArithmetic, tvaluealpha, betas, summary_table = summarizePerformance(xsReturns_TC, rf_d_monthly, SPXT_returns_m, 12, startMonth, "Momentum")
print(summary_table)

# %% [markdown]
# Robustness Checks

# %%
# Define parameters and initialize empty lists or arrays
lookback_period_loop = range(3, 13)
sharpRatio_mom_lb_loop = [0] * len(lookback_period_loop)

# Iterate over the lookback periods
for i, lb_period in enumerate(lookback_period_loop):
    xsReturns_mom_TC_lb, _, _ = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lb_period, lookback_period_end, holding_period, startMonth, nLong, nShort, trx_cost)
    _, _, _, sharpRatio_mom, _, _, _, _, _, _, _, _ = summarizePerformance(xsReturns_mom_TC_lb, rf_d_monthly, SPXT_returns_m, 12, startMonth, "Momentum")

    sharpRatio_mom_lb_loop[i] = sharpRatio_mom

# Plot the results
plt.figure()
plt.plot(lookback_period_loop, sharpRatio_mom_lb_loop, linewidth=myLineWidth, color=RobustnessCheckColor)
plt.xlabel("lookback period")
plt.ylabel("Sharpe Ratio")
plt.show()

# %%
# Define parameters and initialize empty lists or arrays
investment_horizon_loop = range(1, 6)
sharpRatio_investment_horizon_loop = [0] * len(investment_horizon_loop)

# Iterate over the investment horizons
for i, inv_horizon in enumerate(investment_horizon_loop):
    # Based on your MATLAB code, lookback period start is fixed at 9, lookback period end is fixed at 1
    xsReturns_mom_TC_IH, _, _ = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lb_period, lookback_period_end, inv_horizon, startMonth, nLong, nShort, trx_cost)
    _, _, _, sharpRatio_mom, _, _, _, _, _, _, _, _= summarizePerformance(xsReturns_mom_TC_IH, rf_d_monthly, SPXT_returns_m, 12, startMonth, 'mom_inv_horizon')

    sharpRatio_investment_horizon_loop[i] = sharpRatio_mom

# Plot the results
plt.figure()
plt.plot(investment_horizon_loop, sharpRatio_investment_horizon_loop, linewidth=myLineWidth, color=RobustnessCheckColor)
plt.xlabel("investment horizon")
plt.ylabel("Sharpe Ratio")
plt.show()

# %%
# Define parameters and initialize empty lists or arrays
holdings_loop = range(1, 6)
sharpRatio_holdings_loop = [0] * len(holdings_loop)

# Iterate over the investment horizons
for i, holdings in enumerate(holdings_loop):
    # Based on your MATLAB code, lookback period start is fixed at 9, lookback period end is fixed at 1
    xsReturns_mom_TC_holdings, _, _ = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lb_period, lookback_period_end, inv_horizon, startMonth, holdings, nShort, trx_cost)
    _, _, _, sharpRatio_mom, _, _, _, _, _, _, _, _ = summarizePerformance(xsReturns_mom_TC_holdings, rf_d_monthly, SPXT_returns_m, 12, startMonth, 'mom_inv_horizon')

    sharpRatio_holdings_loop[i] = sharpRatio_mom

# Plot the results
plt.figure()
plt.plot(holdings_loop, sharpRatio_holdings_loop, linewidth=myLineWidth, color=RobustnessCheckColor)
plt.xlabel("# of holdings")
plt.ylabel("Sharpe Ratio")
plt.show()

# %%
tx_costs = [0.001, 0.0025, 0.005, 0.01]
total_returns_tx_costs = []
labels = []

for trx_cost in tx_costs:
    _, totalReturns_TC, _ = momentum.backtest_momentum(
        Sectors_returns_m, rf_d_monthly, lookback_period_start, lookback_period_end,
        holding_period, startMonth, nLong, nShort, trx_cost
    )
    # Storing the total returns in a list
    total_returns_tx_costs.append(totalReturns_TC)
    # Creating corresponding labels
    labels.append(f'Trx Cost: {trx_cost}')

visualization.plot_strategies_with_benchmark(dates4plot, total_returns_tx_costs, SPXT_returns_m, labels, startMonth)


