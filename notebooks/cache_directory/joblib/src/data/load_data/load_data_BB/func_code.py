# first line: 46
    def load_data_BB(self, start_date, end_date, ticker_index, ticker_rf, field_index, field_rf):
        from xbbg import blp
        
        index_values = blp.bdh(ticker_index,field_index,start_date,end_date, Per ='D')
        rf_values = blp.bdh(ticker_rf,field_rf,start_date,end_date, Per ='D')

        # Convert to normal dataframe
        rf_values.columns.set_names('date', level=0, inplace=True)
        index_values.columns.set_names('date', level=0, inplace=True)

        # Step 1: Extract specific columns
        index_values = index_values[ticker_index]

        # Step 2: Reset the index
        index_values = index_values.reset_index()
        rf_values = rf_values.reset_index()

        # Step 3: Rename the columns to 'dates' and 'names'
        rf_values.columns = ['dates', 'GB03 Govt']
        index_values.columns = ['dates'] + ticker_index

        # Merge the dataframe
        rf_values['dates'] = pd.to_datetime(rf_values['dates'])
        index_values['dates'] = pd.to_datetime(index_values['dates'])

        # Merge the dataframes
        all_data = pd.merge(index_values, rf_values, on='dates', how='left')

        # Extract the columns into separate DataFrames
        dates_dateformat = all_data.iloc[0:, 0:1]
        SPXT = all_data.iloc[0:, 1:2]
        Sectors = all_data.iloc[0:, 2:13]
        Rf = all_data.iloc[0:, 37:38]
        Industry_Groups = all_data.iloc[0:, 13:37]

        # Make sure that numbers are numeric

        SPXT = SPXT.apply(pd.to_numeric, errors='coerce')
        Sectors = Sectors.apply(pd.to_numeric, errors='coerce')
        Rf = Rf.apply(pd.to_numeric, errors='coerce')
        Industry_Groups = Industry_Groups.apply(pd.to_numeric, errors='coerce')
       
        # Return data frames
        return dates_dateformat, SPXT, Sectors, Rf, Industry_Groups
