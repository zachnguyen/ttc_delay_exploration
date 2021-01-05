# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:26:31 2020

@author: Zach Nguyen
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from sklearn import preprocessing

### PREPROCESSING SCRIPTS

def read_data_into_dataframe(data_path, filename):
    """ A function which reads raw data into a pandas dataframe.
    Arguments:
        data_path (str): the path of the data directory
        filename (str): the name of the raw data file
    return:
        A pandas dataframe of the data """
    # Read the data and save to a pandas dataframe
    if filename.endswith('.csv'):
        dataframe = pd.read_csv(os.path.join(data_path, filename))
    elif filename.endswith('.xlsx'):
        dataframe = pd.read_excel(os.path.join(data_path, filename), sheet_name=None)
    return dataframe

def one_dataframe(path):
    """ Compile all TTC delay data into one dataframe.
    Arguments:
        path (str): Path folder of the data directory, which should contain excel files for each year and a readme metafata file.
    Return:
        A pandas dataframe with all data compiled into one dataframe.
    """
    # Initialize a dictionary to store all yearly data
    df_year_dict = {}
    
    # Initialize a mapper to change column names (they used to have different names)
    col_mapper = {'Min Delay':'Delay',
                ' Min Delay':'Delay',
                'Min Gap':'Gap'}
    # Loop for all excel files in a path to get all the data
    for excel_filename in os.listdir(path):
        if 'readme' not in excel_filename:
            # Get data for every month of the year
            df_month_dict = read_data_into_dataframe(path, excel_filename)
            # Rename the columns in every month
            for month in df_month_dict.keys():
                df_month_dict[month] = df_month_dict[month].rename(columns=col_mapper)
            print('Excel sheets processed ...')
            print(list(df_month_dict.keys()))
            
            # Concatenate all month data to get year data
            df_year_dict[excel_filename] = pd.concat([df for df in df_month_dict.values()]).reset_index(drop=True)
        print(list(df_year_dict.keys()))
    # Concatenate all year data to get final data
    df_merged = pd.concat([df for df in df_year_dict.values()]).reset_index(drop=True)
    return df_merged

def understand_dataframe(df):
    """ A function which helps me understand the dataframe 
    Arguments:
        df - the pandas dataframe with the data
    Return:
        printed statements with information about the data 
    """
    
    print(f"{df.info()} \n ----------- \n")
    print(f"There are {df.shape[0]} rows and {df.shape[1]} columns. \n ----------- \n")
    print(f"There are {df.isnull().sum().values.sum()} missing values in the dataframe. \n ----------- \n")
    print(f"The number of unique values for each column is: \n{df.nunique()}. \n ----------- \n")
    
def process_dataframe(df, dt_format=None, dt_cols=[], num_cols=[], cat_cols=[]):
    """ A function which process the columns of a dataframe into the correct datatypes
    Arguments:
        df - the pandas dataframe to be processed
        dt_cols (list) - a list of column names which has datetimes datatype
        dt_format (str) - the format of the datetime columns
        num_cols (list) - a list of column names which has numerical datatype
        cat_cols (list) - a list of column names which has categories datatype 
    Return:
        the processed dataframe"""
        
    # Loop through all columns in the dataframe
    for column in list(df.columns):
        
        # change to correct datatype of each column
        if column in dt_cols:
            df[column] = pd.to_datetime(df[column], 
                                        format=dt_format)
        elif column in num_cols:
            df[column] = pd.to_numeric(df[column],errors='coerce')
            
        elif column in cat_cols:
            df[column] = df[column].astype('category')
    
    return df

def join_date_time(df, date_col, time_col, dt_format="%Y-%m-%d %H:%M:%S"):
    """ For situations where date and time are in different columns. Takes a date column and time column in the dataframe, join them and create a datetime column
    ARG:
        df - the pandas frame to create datetime column
        date_col (str) - the name of the column with dates
        time_col (str) - the name of the column with time
        dt_format (str) - format of the datetime
    Return:
        Dataframe with only datetime column 
    """
    # Join date and time to create datetime column
    df['datetime'] = pd.to_datetime(arg=df[date_col].astype(str)+' '+df[time_col].astype(str),
                                    format=dt_format)
    
    # Drop used date and time column
    df = df.drop([date_col, time_col], axis=1)
    return df

def make_time(df, datetime_column):
    """ A function which generate new columns (year, month, week, dayofweek) for dataframe
    Arguments:
        df - the pandas dataframe to add columns to
        datetime_column (str) - The datetime column to parse
    Return:
        new dataframe with added columns"""
    
    # Add year column 
    df["year"] = df[datetime_column].dt.year
    
    # Add month column
    df["month"] = df[datetime_column].dt.month
    
    # Add day of week column
    df["dayofweek"] = df[datetime_column].dt.dayofweek
    
    # Add hour column
    df["hour"] = df[datetime_column].dt.hour
    
    # Add a date column
    df["date"] = df[datetime_column].dt.date
    return df
    
    
### EDA SCRIPTS

def plot_multiple_hist(df, unit_map_dict, num_cols, rows, cols, bins='auto'):
    """ Draw multiple histograms of numerical variables. Requires pandas as pd and matplotlib.pyplot as plt
    ARGS:
        df = Dataframe with given features
        unit_map_dict (dict): A dictionary which maps the column names to the units of their axis
        num_cols (list): A list of names of numeric columns
        rows (int): The number of rows to display histograms
        cols (int): The number of columns to display histograms
        bins (int/str): The number of bins
    Return:
        All the histograms of numeric columns """
    
    # Make subplots    
    f,a = plt.subplots(nrows=rows, ncols=cols, figsize=(15,5))
    a = a.ravel()
    
    # Loop through subplots to plot histograms
    for idx,ax in enumerate(a):
        ax.hist(df[num_cols[idx]], bins=bins, color='crimson', alpha = 0.8, rwidth=0.85)
        ax.set_title(num_cols[idx] + " Distribution")
        ax.set_xlabel(num_cols[idx] + f" ({unit_map_dict[num_cols[idx]]})")
        ax.set_ylabel("frequency")
    plt.tight_layout()
    plt.show()
    
def plot_multiple_bar(df, cat_cols, rows, cols, params):
    """ Draw multiple bar graphs of categorical variables. Requires pandas as pd and matplotlib.pyplot as plt
    ARGS:
        df = Dataframe with given features
        unit_map_dict (dict): A dictionary which maps the column names to the units of their axis
        num_cols (list): A list of names of numeric columns
        rows (int): The number of rows to display histograms
        cols (int): The number of columns to display histograms
        params (dict): parameter dictionary to adjust the figure
    Return:
        All the histograms of categorical columns """
    plt.rcParams.update(params)

    # Make subplots    
    f,a = plt.subplots(nrows=rows, ncols=cols, figsize=(40,12))
    a = a.ravel()
    
    # Loop through subplots to plot histograms
    for idx, ax in enumerate(a):
        ax.bar(x=df[cat_cols[idx]].value_counts(sort=False).index,
               height=df[cat_cols[idx]].value_counts(sort=False).values,
               color='darkgreen', alpha=0.8, width=0.7)
        ax.set_title(cat_cols[idx] + " Distribution")
        ax.set_xlabel(f"{cat_cols[idx]} Categories")
        ax.set_ylabel("frequency")
        ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()
    
def plot_multiple_timeseries(time_series, resample, rows, cols):
    """ Plot time series of different granuity.
    ARGS:
        time_series: pandas time series with pandas datetime index and a dummy column
        resample (list): list of all intervals to resample by (example: year, month, week ...)
        rows (int): The number of rows to display line graph
        cols (int): The number of columns to display line graph 
    Return:
        All line graph of different intervals"""
    
    # Initiate a mapping of the intervals    
    time_map = {'year': 'y', 'month': 'M', 'week': 'w', 'hour': 'H', 'day': 'D', 'minute': 'm'}
    
    # Make subplots 
    f,a = plt.subplots(nrows=rows, ncols=cols, figsize=(15,5))
    a = a.ravel()
    
    # Loop through subplots to plot time series
    for idx,ax in enumerate(a):
        series = time_series.resample(time_map[resample[idx]]).count()
        ax.plot(np.array(series.index),
              np.array(series.values))
        ax.set_title(f"Number of bus delay incidents over the {resample[idx]}s")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()
    

def plot_multiple_timeseries_sum(time_series, resample, rows, cols):
    """ Plot time series of different granuity, with sum of dummy column
    ARGS:
        time_series: pandas time series with pandas datetime index and a dummy column
        resample (list): list of all intervals to resample by (example: year, month, week ...)
        rows (int): The number of rows to display line graph
        cols (int): The number of columns to display line graph 
    Return:
        All line graph of different intervals"""
    
    # Initiate a mapping of the intervals    
    time_map = {'year': 'y', 'month': 'M', 'week': 'w', 'hour': 'H', 'day': 'D', 'minute': 'm'}
    
    # Make subplots 
    f,a = plt.subplots(nrows=rows, ncols=cols, figsize=(15,5))
    a = a.ravel()
    
    # Loop through subplots to plot time series
    for idx,ax in enumerate(a):
        series = time_series.resample(time_map[resample[idx]]).sum()
        ax.plot(np.array(series.index),
              np.array(series.values))
        ax.set_title(f"Number of Minutes delayed over the {resample[idx]}s")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sum of units")
    
    plt.tight_layout()
    plt.show()
    

def get_association(df_cat):
    """ Get Crammer's V Assiociation measure for a given dataframe of (raw) categorical data. 
    Requires seaborn, sklearn's label encoding, chi2_contingency.
    ARG:
        df_cat (df) - A pandas dataframe with raw categorical features. Make sure that dtype = categorical
    Return:
        corr (df) - An association matrix of all the categorical features (0.1+ implies significance)
        A heat map of the matrix.
    """
    
    # Define the cramer's V measure to be applied on features
    def cramers_V(var1,var2) :
        crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
        stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
        obs = np.sum(crosstab) # Number of observations
        mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
        return (stat/(obs*mini))
    
    # Use label encoding to convert categorical features into numeric levels
    le = preprocessing.LabelEncoder()
    df_encoded = pd.DataFrame() 
    
    for col in df_cat.columns:
        df_encoded[col] = le.fit_transform(df_cat[col])
     
    # Construct the correlation matrix object by calculating Crammer's V measure for all pairs
    rows= []
    for var1 in df_encoded:
        col = []
        for var2 in df_encoded :
            cramers = cramers_V(df_encoded[var1], df_encoded[var2]) # Cramer's V test
            col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
        rows.append(col)
      
    cramers_results = np.array(rows)
    corr = pd.DataFrame(cramers_results, columns = df_encoded.columns, index =df_encoded.columns)
    
    ## Plot the correlation matrix with seaborn
    # Generate a mask for the upper triangle so that we can simplify the plot
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
    return corr
    
    
def slice_and_dice(df, var1, var2, dummy, norm=False, color_scheme=None):
    """ Slice and dice two categorical variables with a stacked bar graph.
    ARG:
        df - The pandas dataframe to be sliced
        var1 (str) - The first variable to be sliced, the variable is the bar's length.
        var2 (str) - The second variable to be sliced, the variable are stacked pieces (or percentage if norm) to a bar.
        dummy (str) - The dummy variable to be used as count (usually 'id')
    RETURN:
        A plot of the slice
        pivot (df) - The pivot table of the sliced data.
    """
    
    # Construct the pivot table. If normalized, the absolute values will be converted into percentage 
    if norm:
        pivot = df[[var1, var2, dummy]].groupby([var1, var2]).count().unstack(-1).fillna(0)
        pivot = pivot.div(pivot.sum(axis=1), axis=0).multiply(100)
    else:
        pivot = df[[var1, var2, dummy]].groupby([var1, var2]).count().unstack(-1).fillna(0)
    
    # Use percentage for label
    ylab = ['percentage' if norm else ''][0]
    
    # Plot the pivot
    if not color_scheme:
        pivot.plot(kind='bar', stacked=True, figsize=(17,10))
    else:
        pivot.plot(kind='bar', stacked=True, figsize=(17,10), colormap=color_scheme)
    plt.title(f'Distribution of Incident {var1} and {var2}')
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2} -- {ylab}')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()
    
    return pivot