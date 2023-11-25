# first line: 8
def process_data(dates_dateformat, Sectors, Industry_Groups, SPXT, Rf):
    # Process dates
    dates_datetime = pd.to_datetime(dates_dateformat.iloc[:, 0], errors='coerce')
    numericDate_d = dates_datetime.dt.strftime('%Y%m%d').astype(int)
    firstDayList, lastDayList = data_processor.getFirstAndLastDayInPeriod(numericDate_d, 2)
    dates4plot = [dates_datetime[i] for i in firstDayList]

    # Fill up NaN for Real Estate Sector and Industry Group Index because they were only launched at 19.09.2016, otherwise we would have lookahead bias
    Sectors.loc[0:7048, "S5RLST Index"] = float("NaN")
    Industry_Groups.loc[0:7048, "S5REAL Index"] = float("NaN")

    # Process returns
    Sectors_returns_d = Sectors.pct_change().iloc[1:]
    Sectors_returns_m = data_processor.aggregateReturns(Sectors_returns_d, numericDate_d, 2)
    Industry_Groups_returns_d = Industry_Groups.pct_change().iloc[1:]
    Industry_Groups_returns_m = data_processor.aggregateReturns(Industry_Groups_returns_d, numericDate_d, 2)
    SPXT_returns_d = SPXT.pct_change().iloc[1:]
    SPXT_returns_m = data_processor.aggregateReturns(SPXT_returns_d, numericDate_d, 2)

    # Process Rf
    rf_d_unadjusted = Rf / 100
    rf_d = data_processor.adjust_interest_rates(rf_d_unadjusted, numericDate_d)
    rf_d_monthly = data_processor.aggregateReturns(rf_d, numericDate_d, 2)

    # Make SPXT excess returns
    SPXT_Xsreturns_m = SPXT_returns_m - rf_d_monthly

    return dates_datetime, numericDate_d, firstDayList, lastDayList, dates4plot, Sectors_returns_d, Sectors_returns_m, Industry_Groups_returns_d, Industry_Groups_returns_m, SPXT_returns_d, SPXT_returns_m, SPXT_Xsreturns_m, rf_d_unadjusted, rf_d, rf_d_monthly
