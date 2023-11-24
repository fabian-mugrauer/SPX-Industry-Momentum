import numpy as np
from scipy.stats import skew, kurtosis
from prettytable import PrettyTable

class PerformanceSummarizer:
    @staticmethod
    def summarize_performance(xsReturns, Rf, factorXsReturns, annualizationFactor, startM):
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
