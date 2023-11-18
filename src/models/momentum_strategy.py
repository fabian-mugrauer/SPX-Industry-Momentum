import numpy as np

class MomentumStrategy:
    def __init__(self):
        # Initialization can set up any default values or configurations
        pass

    def compute_sort_weights(self, sortVariable, nLongs, nShorts, longHighValues):
        """
        This function calculates portfolio weights based on a specified sorting variable.
        It is designed to allocate weights for a portfolio by selecting a certain number of
        assets to hold in long and short positions. The sorting variable determines the
        basis for this selection.

        Parameters:
        - sortVariable (array-like): A collection of values used to rank the assets. The function
        expects non-NaN values for valid processing. Assets associated with NaN values in
        this array are excluded from the selection.
        - nLongs (int): The number of assets to be held in a long position. The function selects
        the top 'nLongs' assets based on the sorting variable.
        - nShorts (int): The number of assets to be held in a short position. The function selects
        the bottom 'nShorts' assets based on the sorting variable.
        - longHighValues (bool): A flag that determines the sorting order. If True, the function
        treats higher values of 'sortVariable' as preferable for long positions (and vice versa
        for short positions). If False, lower values are favored for long positions.

        The function first filters out NaN values from the sortVariable, then sorts the remaining
        assets based on their values. It assigns equal weights to selected assets for long and
        short positions. Assets not selected for either position receive a weight of zero.

        Returns:
        - weights (numpy array): An array of weights corresponding to each asset in the
        sortVariable array. The weights are positive for long positions, negative for short
        positions, and zero for non-selected assets.
        """

        # Ensure sortVariable is a numpy array
        sortVariable = np.array(sortVariable)

        # Initialize weights with zeros
        weights = np.zeros(len(sortVariable))

        # Filter out NaN values and get their indices
        valid_indices = np.where(~np.isnan(sortVariable))[0]
        valid_values = sortVariable[valid_indices]

        # Get sorted indices based on valid_values
        if longHighValues:
            sorted_indices = np.argsort(valid_values)[::-1] # From high to low
        else:
            sorted_indices = np.argsort(valid_values)      # From low to high

        # Assign weights to longs only if nLongs is not zero
        if nLongs != 0:
            long_indices = valid_indices[sorted_indices[:nLongs]]
            weights[long_indices] = 1.0 / nLongs

        # Assign weights to shorts only if nShorts is not zero
        if nShorts != 0:
            short_indices = valid_indices[sorted_indices[-nShorts:]]
            weights[short_indices] = -1.0 / nShorts

        return weights
    
    def compute_turnover(self, previous_weights, new_weights, asset_returns):
        """
        Computes the turnover of a portfolio between two periods and the portfolio return excluding transaction costs.

        Parameters:
        - previous_weights: An array of portfolio weights in the previous period.
        - new_weights: An array of portfolio weights for the current period.
        - asset_returns: An array of asset returns between the two periods.

        The function calculates the total turnover, which is the sum of the absolute differences between new and previous weights. It also computes the portfolio return (Rp) excluding transaction costs, which is the weighted sum of asset returns. Current weights are calculated based on the portfolio return and the value per asset.

        Returns:
        - turnover: The total turnover of the portfolio.
        - rp: Portfolio return excluding transaction costs.
        """
        # Ensuring the inputs are NumPy arrays for correct operations
        previous_weights = np.abs(np.array(previous_weights))
        new_weights = np.abs(np.array(new_weights))
        asset_returns = np.array(asset_returns)

        # Calculating portfolio return excluding transaction costs, Rp
        rp = np.nansum(previous_weights * (1 + asset_returns))

        # Calculating value per asset
        value_per_asset = previous_weights * (1 + asset_returns)

        # Calculating current weights
        if rp == 0:
            current_weights = np.zeros_like(previous_weights)  # Set current_weights to zero if rp is zero
        else:
            current_weights = value_per_asset / rp  # No indexing needed since rp is a scalar

        current_weights[np.isnan(current_weights)] = 0  # Replacing NaN values with 0

        # Calculating turnover
        turnover = np.nansum(np.abs(new_weights - current_weights))

        return turnover, rp
    
    def backtest_momentum(self, returns_m, rf_m, lookback_period_start, lookback_period_end, holding_period, startMonth, nLong, nShort, trx_cost):
        """
        Performs a backtest of a momentum-based trading strategy.

        Parameters:
        - returns_m: Matrix of monthly returns for different assets or sectors.
        - rf_m: Array of risk-free monthly returns.
        - lookback_period_start: The start of the lookback period for momentum calculation.
        - lookback_period_end: The end of the lookback period for momentum calculation.
        - holding_period: The number of months each asset is held after being selected.
        - startMonth: The starting month for the backtest.
        - nLong: Number of assets to hold in the long portfolio.
        - nShort: Number of assets to hold in the short portfolio.
        - trx_cost: Transaction costs as a proportion of the trade amount.

        The function calculates past returns, forms long and short portfolios based on these returns, and computes the turnover and total returns with and without transaction costs. It handles the portfolio rebalancing at each trading interval and ensures that the sum of absolute weights in both long and short portfolios equals one.

        Returns:
        - xsReturns_TC: Excess returns after transaction costs.
        - totalReturns_TC: Total returns including risk-free returns, after transaction costs.
        - weights: Portfolio weights over the backtest period.
        """

        # define shape parameters
        nMonths, nSectors = returns_m.shape

        # possible assets to choose from
        possible_assets = np.ones((nMonths, nSectors))
        possible_assets[:lookback_period_start, :] = 0
        possible_assets[lookback_period_start:, :] = ~np.isnan(returns_m[lookback_period_start:, :])

        # Calculate cumulative monthly returns over lookback_period
        past_r_temp = np.cumprod(returns_m + 1, axis=0)
        past_r_temp[np.isnan(returns_m)] = np.nan

        past_returns_m = np.zeros((nMonths, nSectors))
        past_returns_m[lookback_period_start:] = (past_r_temp[lookback_period_start - lookback_period_end:-lookback_period_end] / past_r_temp[:-lookback_period_start]) * (1 + returns_m[:-lookback_period_start])
        past_returns_m[~possible_assets.astype(bool)] = np.nan

        # Predefine matrices
        weights_long = np.zeros((nMonths, nSectors))
        weights_short = np.zeros((nMonths, nSectors))
        turnover_long = np.zeros(nMonths)
        turnover_short = np.zeros(nMonths)

        for month in range(startMonth, nMonths):

            if month == startMonth or (month - startMonth) % holding_period == 0:
                # Compute new target weights
                new_weights_long = self.compute_sort_weights(past_returns_m[month, :], nLong, 0, True)
                new_weights_short = self.compute_sort_weights(past_returns_m[month, :], 0, nShort, True)

                # Adjust for holding period
                delta_long = (new_weights_long - weights_long[month-1, :]) / holding_period
                delta_short = (new_weights_short - weights_short[month-1, :]) / holding_period

                weights_long[month, :] = weights_long[month-1, :] + delta_long
                weights_short[month, :] = weights_short[month-1, :] + delta_short
            else:
                weights_long[month, :] = weights_long[month-1, :]
                weights_short[month, :] = weights_short[month-1, :]

        # Ensure the sum of absolute weights equals one
        total_long_weight = np.sum(np.abs(weights_long[month, :]))
        total_short_weight = np.sum(np.abs(weights_short[month, :]))

        weights_long[month, :] /= total_long_weight
        weights_short[month, :] /= total_short_weight

        turnover_long[month], _ = self.compute_turnover(weights_long[month-1, :], weights_long[month, :], returns_m[month-1, :])
        turnover_short[month], _ = self.compute_turnover(weights_short[month-1, :], weights_short[month, :], returns_m[month-1, :])

        totalReturnsLong = np.nansum(weights_long * returns_m, axis=1)
        totalReturnsShort = np.nansum(weights_short * returns_m, axis=1)
        xsReturns = np.zeros(nMonths)

        if nShort == 0:
            xsReturns[startMonth:] = totalReturnsLong[startMonth:] - rf_m[startMonth:].squeeze()
            weights = weights_long
        else:
            xsReturns = totalReturnsLong + totalReturnsShort
            weights = weights_long + weights_short

        xsReturns_TC = xsReturns - (np.abs(turnover_long) + np.abs(turnover_short)) * trx_cost
        totalReturns_TC = xsReturns_TC + np.concatenate([np.zeros(startMonth), rf_m[startMonth:].squeeze()])

        return xsReturns_TC, totalReturns_TC, weights


