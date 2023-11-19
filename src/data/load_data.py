import os
import pandas as pd

def load_data(file_path, Excel_file_name):
    # Load the Excel file
    Excel_file_path = os.path.join(file_path, Excel_file_name)
    xls = pd.ExcelFile(Excel_file_path)
    sheet_name = xls.sheet_names[0]
    df = xls.parse(sheet_name)

    # Extract the columns into separate DataFrames
    dates_dateformat = df.iloc[6:, 0:1]
    SPXT = df.iloc[6:, 1:2]
    Sectors = df.iloc[6:, 2:13]
    Rf = df.iloc[6:, 13:14]
    Industry_Groups = df.iloc[6:, 14:39]

    # Extract the column names from the sixth row
    column_names = df.iloc[2]

    # Rename the columns for each DataFrame
    dates_dateformat.columns = [column_names[0]]
    SPXT.columns = [column_names[1]]
    Sectors.columns = column_names[2:13]
    Rf.columns = [column_names[13]]
    Industry_Groups.columns = column_names[14:39]

    # Reset the indexing
    dates_dateformat = dates_dateformat.reset_index(drop=True)
    SPXT = SPXT.reset_index(drop=True)
    Sectors = Sectors.reset_index(drop=True)
    Rf = Rf.reset_index(drop=True)
    Industry_Groups = Industry_Groups.reset_index(drop=True)

    # Return data frames
    return dates_dateformat, SPXT, Sectors, Rf, Industry_Groups

