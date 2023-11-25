import numpy as np
from scipy.stats import skew, kurtosis
from prettytable import PrettyTable
from src.models.momentum_strategy import MomentumStrategy  # Import the MomentumStrategy class

class EvaluateStrategy:
    @staticmethod
    def summarize_performance(xsReturns, Rf, factorXsReturns, annualizationFactor, startM, headline):
        """
        Calculate various performance metrics for the portfolio.

        Parameters:
        xsReturns (np.array): Excess returns of the portfolio.
        Rf (np.array): Risk-free rate.
        factorXsReturns (np.array): Excess returns of the benchmark or factor.
        annualizationFactor (float): Factor to annualize returns.
        startM (int): Starting month for the analysis.

        Returns:
        tuple: A tuple containing various performance metrics and a PrettyTable object.
        """

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
        alphaArithmetic = annualizationFactor * betas[0]

        # Create table
        x = PrettyTable()

        x.title = headline
        x.field_names = ["Statistic", "Value"]
 

        x.add_row(["ArithmAvgTotalReturn", f"{ArithmAvgTotalReturn[0]:.3f}"])
        x.add_row(["ArithmAvgXsReturn", f"{ArithmAvgXsReturn[0]:.3f}"])
        x.add_row(["StdXsReturns", f"{StdXsReturns[0]:.3f}"])
        x.add_row(["SharpeArithmetic", f"{SharpeArithmetic[0]:.3f}"])
        x.add_row(["SharpeGeometric", f"{SharpeGeometric[0]:.3f}"])
        x.add_row(["MinXsReturn", f"{MinXsReturn[0]:.3f}"])
        x.add_row(["MaxXsReturn", f"{MaxXsReturn[0]:.3f}"])
        x.add_row(["SkewXsReturn", f"{SkewXsReturn[0]:.3f}"])
        x.add_row(["KurtXsReturn", f"{KurtXsReturn[0]:.3f}"])
        x.add_row(["Beta", f"{betas[0][0]:.3f}"])

        return (ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic, MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn, alphaArithmetic, tvaluealpha, betas, x)
    
    def perform_robustness_check(momentum, check_type, check_range, Sectors_returns_m, rf_d_monthly, SPXT_returns_m, lookback_period_end, startMonth, nLong, nShort, trx_cost, holding_period):
        """
        Perform robustness checks for different parameters.

        Parameters:
        momentum (MomentumStrategy): An instance of MomentumStrategy to use its backtest_momentum method.
        check_type (str): Type of check ('lookback', 'investment_horizon', 'holdings').
        check_range (range): Range of values to iterate over for the specified check.
        [Other parameters]

        Returns:
        list: A list of Sharp Ratios for each value in the check range.
        """
        sharpRatio_loop = [0] * len(check_range)

        for i, value in enumerate(check_range):
            if check_type == 'lookback':
                lb_period = value
                inv_horizon = holding_period
                holdings = nLong
            elif check_type == 'investment_horizon':
                lb_period = 9  # or some fixed value
                inv_horizon = value
                holdings = nLong
            elif check_type == 'holdings':
                lb_period = 9  # or some fixed value
                inv_horizon = holding_period
                holdings = value
            else:
                raise ValueError("Invalid check type")

            xsReturns, _, _ = momentum.backtest_momentum(Sectors_returns_m, rf_d_monthly, lb_period, lookback_period_end, inv_horizon, startMonth, holdings, nShort, trx_cost)
            _, _, _, sharpRatio, _, _, _, _, _, _, _, _ = EvaluateStrategy.summarize_performance(xsReturns, rf_d_monthly, SPXT_returns_m, 12, startMonth, headline=None)

            sharpRatio_loop[i] = sharpRatio

        return sharpRatio_loop
