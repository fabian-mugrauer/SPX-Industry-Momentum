import numpy as np
import pandas as pd

class FinancialDataProcessor:
    def __init__(self):
        # Initialization can be used to set up any necessary properties or default values
        pass

    def getFirstAndLastDayInPeriod(self, dateList, nDigits):
        """
        Generate arrays listing the first and last observation in each period.
        """
        dateList = [int(date) for date in dateList]
        nObs = len(dateList)
        scalingFactor = 1 / 10**nDigits
        trimmedDate = np.round(np.array(dateList) * scalingFactor)
        isNewPeriod = np.diff(trimmedDate)

        lastDayList = np.where(isNewPeriod)[0]
        firstDayList = lastDayList + 1
        lastDayList = np.append(lastDayList, nObs - 1)
        firstDayList = np.insert(firstDayList, 0, 0)

        return firstDayList, lastDayList

    def adjust_interest_rates(self, interest_rates, numericDate_d):
        """
        Adjusts interest rates for different period lengths.
        """
        date_in_datetime_format = pd.to_datetime(numericDate_d.astype(str), format='%Y%m%d')
        day_counts = (date_in_datetime_format - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')
        df = np.diff(day_counts)
        df = np.append(df, 0)
        rf_d = interest_rates * (df[:, None] * np.ones((1, interest_rates.shape[1]))) / 360

        return rf_d

    def aggregateReturns(self, originalReturns, dateList, nDigits):
        """
        Aggregates returns over time per asset.
        """
        originalReturns = originalReturns.values
        firstDayList, lastDayList = self.getFirstAndLastDayInPeriod(dateList, nDigits)
        
        if np.ndim(originalReturns) == 1:
            originalReturns = originalReturns[:, np.newaxis]

        nPeriods = len(firstDayList)
        nAssets = originalReturns.shape[1]
        aggregatedReturns = np.zeros((nPeriods, nAssets))

        for n in range(nPeriods):
            first = firstDayList[n]
            last = lastDayList[n]
            for asset in range(nAssets):
                returns_subset = originalReturns[first:last + 1, asset]
                aggregatedReturns[n, asset] = np.prod(1 + returns_subset) - 1

        return aggregatedReturns