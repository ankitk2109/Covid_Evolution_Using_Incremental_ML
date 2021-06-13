# Imports for incremental learner
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeRegressor
from content.src.evaluate_prequential import EvaluatePrequential
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

# Imports for static Learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from time import perf_counter as pc_timer
from functools import wraps
from os import walk

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from pandas.core.common import SettingWithCopyWarning
from collections import Counter

# For significance tests
from scipy.stats import normaltest
from scipy import stats
from math import sqrt

# pd.set_option('display.max_colwidth', 500)
# General Imports
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

import numpy as np


path = "covid_data_nov_30.csv"
df = pd.read_csv(path)
df.head()

# Setting path variables for both experiments
csv_processed_path = 'content/csv_files/processed'
csv_processed_with_null_path = 'content/csv_files/processed_null'
exp1_path = 'content/Result/exp1'
exp2_path = 'content/Result/exp2'
exp1_runtime_path = 'content/Result/exp1/runtime'
exp2_runtime_path = 'content/Result/exp2/runtime'
exp1_summary_path = 'content/Result/exp1/summary'
exp2_summary_path = 'content/Result/exp2/summary'
bar_plot_path = r'content/Plots/barplot'
box_plot_path = r'content/Plots/boxplots'

# Grouping countries together for analysis
total_countries = df['countriesAndTerritories'].unique()
df_grouped = df.groupby('countriesAndTerritories')
pretrain_days = [30, 60, 90, 120, 150, 180, 210, 240]  # List of pretrain days
valid_countries = []
decimal = 3  # Specify the scale of decimal places 
error_metrics = ['MAE', 'MAPE', 'RMSE']


# Create lags
def create_features_with_lags(df):
    for i in range(89, 0,
                   -1):  # Loop in reverse order for creating ordered lags eg: cases_t-10, cases_t-9... cases_t-1. t=current cases
        df[f'cases_t-{i}'] = df['cases'].shift(i, axis=0)
    return df


# Creating countrywise csv files
'''
# Pre-Processing dataset and saving them into csv's.
for country in total_countries:
    df = df_grouped.get_group(country)

    # Selecting required features
    df = df[['dateRep', 'cases', 'countriesAndTerritories']]

    # Rename features
    df.rename(columns={'countriesAndTerritories': 'country', 'dateRep': 'date'}, inplace=True)

    # Convert to date, sort and set index
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Adding feature
    df['day_no'] = pd.Series([i for i in range(1, len(df)+1)], index=df.index)

    # Reordering features
    df = df[['day_no', 'country', 'cases']]

    # Adding features through lags
    df = create_features_with_lags(df)

    # Creating target with last 10 days cases
    df['target'] = df.iloc[:, [2]+[i*-1 for i in range(1, 10)]].mean(axis=1)

    # Dropping mid columns
    drop_columns = list(df.loc[:, 'cases_t-39':'cases_t-1'].columns)  # list(df.loc[:,'cases_t-38':'cases_t-1'].columns)
    df.drop(drop_columns, axis=1, inplace=True)

    # Country name
    filename = df['country'].unique()[0]

    # Saving file
    df.to_csv(f'{csv_processed_with_null_path}/{filename}.csv')

    # Dropping null records
    df.dropna(how='any', axis=0, inplace=True)

    # Valid countries that have records more than max of pretrain
    if len(df) > max(pretrain_days):
        valid_countries.append(country)
        df.to_csv(f'{csv_processed_path}/{filename}.csv')
  
print('Done!')
'''


"""## Total cases of top selected countries"""

# Added just for plots. Remove later
Number_of_countries = 50


# Replaces underscore from country names
def format_names(list_countries):
    updated_country_list = []
    for country_name in list_countries:
        updated_country_list.append(country_name.replace("_", " "))
    return updated_country_list

valid_country_path = r'content/csv_files/processed'
_, _, valid_countries_list = next(walk(valid_country_path))
valid_countries = [country.split('.')[0] for country in valid_countries_list]


# A dictionary of all countries
dict_countries = Counter(valid_countries)

for country in valid_countries:
    dict_countries[country] = df_grouped.get_group(country)['cases'].sum()

top_countries_num = 50
# Select top_countries and order(Ascending/Decending) 
top_countries = sorted(dict_countries.items(), key=lambda dict_countries: dict_countries[1], reverse=True)[
                0:top_countries_num]  #Added Now_25-2

# Creating dataframe of top selected countries
df_top_countries = pd.DataFrame.from_dict(dict(top_countries), orient='index', columns=['Total Cases'])

# Plotting graph
"""sns.set_theme(style='white')
plt.figure(figsize=(12, 16))
top_countries_list = format_names(df_top_countries.index)
plt.barh(top_countries_list[::-1], df_top_countries['Total Cases'].values[
                                   ::-1])  # Reversing the order to have heighest values at the top of bar chart
# plt.title(f'Top {len(top_countries)} Countries with Most Cases')
plt.xscale('log')
ax = plt.axes()  # for updating axes values to plain text
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.margins(y=0)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Number of Cases', fontsize=20)
plt.ylabel('Countries', fontsize=20)
plt.tight_layout()
plt.savefig(f'{bar_plot_path}/top_selected_country_cases.pdf')
# plt.show()
"""


dict_countries_avg = Counter(total_countries)

for country in dict_countries.keys():
    dict_countries_avg[country] = df_grouped.get_group(country)['cases'].mean()

# Average cases for all countries
df_avg_cases_countries = pd.DataFrame.from_dict(dict_countries_avg, orient='index', columns=['Average'])

# List of top selected countries
top_countries = list(df_top_countries.index)

# Average of selected top countries
avg_df = df_avg_cases_countries[df_avg_cases_countries.index.isin(top_countries)]

"""# Common Methods for All Experiments

## Common Methods

#### Updated Number of countries
"""

# TODO: Update this to 25 later
#Number_of_countries = 50

# Global variables for countries
countries = top_countries[0:Number_of_countries]


# Return a combined dataframe for a each error statistics(MAE,RMSE,MAPE etc) along with the newly added mean row.
def get_metric_with_mean(result: pd.DataFrame, error_metric: str) -> pd.DataFrame:
    df_grouped = result.groupby('EvaluationMeasurement')
    df = df_grouped.get_group(error_metric).reset_index(drop=True)
    df = df.append(df.describe().loc['mean'])
    return df


def calc_mean_to_max_error(df, max_of_pretrain_days, max_of_df):
    i = -1
    for row_num in range(len(df) - 1):  # Go before mean row
        i += 1
        for col_num in df.columns[2:]:
            df.loc[row_num, col_num] = df.loc[row_num, col_num] / max_of_pretrain_days[i]

    for col in df.columns[2:]:
        df.loc['mean', col] = df.loc['mean', col] / max_of_df

    return df


# Note: Do not change the filenames, since they are later being used for visualizations 
def save_runtime(df, path, country=None, static_learner=True, alternate_batch=False, transpose=False):
    df = df.apply(pd.to_numeric, errors='ignore')  # Converting the dataframe to numeric
    df = df.round(decimal)  # Setting the precision

    # if transpose flag is set to true
    if transpose:
        df = df.transpose()

    if country == None:
        if static_learner:
            df.to_latex(f'{path}/combined25country_runtime_static.tex')
            df.to_csv(f'{path}/combined25country_runtime_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined25country_runtime_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined25country_runtime_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined25country_runtime_incremental.tex')
                df.to_csv(f'{path}/combined25country_runtime_incremental.csv')
    else:
        if static_learner:
            df.to_latex(f'{path}/{country}_runtime_static.tex')
            df.to_csv(f'{path}/{country}_runtime_static.csv')
        else:
            df.to_latex(f'{path}/{country}_runtime_incremental.tex')
            df.to_csv(f'{path}/{country}_runtime_incremental.csv')


# Note: Do not change the filenames, since they are later being used for visualizations 
def save_summary_table(df, path, country=False, static_learner=True, alternate_batch=False, transpose=False):
    df = df.apply(pd.to_numeric, errors='ignore')  # Converting the dataframe to numeric
    df = df.round(decimal)  # Setting the precision

    # if transpose flag is set to true
    if transpose:
        df = df.transpose()

    if country:
        metric = df.loc['EvaluationMeasurement'].unique()[0]
        if static_learner:
            df.to_latex(f'{path}/top_countries_{metric}_summary_table_static.tex')
            df.to_csv(f'{path}/top_countries_{metric}_summary_table_static.csv')
        else:
            df.to_latex(f'{path}/top_countries_{metric}_summary_table_incremental.tex')
            df.to_csv(f'{path}/top_countries_{metric}_summary_table_incremental.csv')

    else:
        if static_learner:
            df.to_latex(f'{path}/combined25country_summary_table_static.tex')
            df.to_csv(f'{path}/combined25country_summary_table_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined25country_summary_table_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined25country_summary_table_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined25country_summary_table_incremental.tex')
                df.to_csv(f'{path}/combined25country_summary_table_incremental.csv')


# Note: Do not change the filenames since they are later being used for visualizations
def save_metrics(df, path, country=None, static_learner=True, alternate_batch=False, transpose=False):
    df = df.apply(pd.to_numeric, errors='ignore')  # Converting the dataframe to numeric
    df = df.round(decimal)  # Setting the precision

    # if transpose flag is set to true
    if transpose:
        df = df.transpose()

    metric_type = df.loc['EvaluationMeasurement'].unique()[0]
    if country == None:
        if static_learner:
            df.to_latex(f'{path}/combined25country_{metric_type}_static.tex')
            df.to_csv(f'{path}/combined25country_{metric_type}_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined25country_{metric_type}_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined25country_{metric_type}_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined25country_{metric_type}_incremental.tex')
                df.to_csv(f'{path}/combined25country_{metric_type}_incremental.csv')
    else:
        if static_learner:
            df.to_latex(f'{path}/{country}_{metric_type}_static.tex')
            df.to_csv(f'{path}/{country}_{metric_type}_static.csv')
        else:
            df.to_latex(f'{path}/{country}_{metric_type}_incremental.tex')
            df.to_csv(f'{path}/{country}_{metric_type}_incremental.csv')


def save_combined_summary_table(df, path, static_learner=False, transpose=False):
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.round(decimal)
    if transpose:
        df = df.transpose()

    if static_learner:
        save_path = f'{path}/summary_table_combined_mean_static'
    else:
        save_path = f'{path}/summary_table_combined_mean_incremental'

    df.to_csv(f'{save_path}.csv')
    df.to_latex(f'{save_path}.tex')


def display_runtime_per_country(results_runtime, countries):
    for i in range(len(countries)):
        print(f'_____________Running Time for {countries[i]}________________')
        print(results_runtime[i].to_string())
        print('\n')


def calc_save_err_metric_countrywise(countries, error_metrics, results, max_of_pretrain_per_country,
                                     max_cases_per_country, path, static_learner, transpose):
    countrywise_error_scores = {}
    for i in range(len(countries)):
        country_error_score = []
        for error_metric in error_metrics:
            df_error_metric = get_metric_with_mean(results[i], error_metric=error_metric)
            country_error_score.append(df_error_metric)
            display_countrywise_scores(countries[i], df_error_metric)

            # Transposing the metrics while saving
            save_metrics(df_error_metric, path=path, country=countries[i], static_learner=static_learner,
                         transpose=transpose)

        countrywise_error_scores[countries[i]] = pd.concat(country_error_score, ignore_index=True)

    return countrywise_error_scores


def calc_save_err_metric_combined(error_metrics, results, max_of_pretrain_days, max_selected_countries, path,
                                  static_learner, alternate_batch, transpose):
    combined_err_metric = []
    for error_metric in error_metrics:
        df_error_metric = get_metric_with_mean(results, error_metric=error_metric)

        # if error_metric != 'MAPE':
        #  df_error_metric = calc_mean_to_max_error(df_error_metric, max_of_pretrain_days, max_selected_countries)

        # Transposing the metrics while saving
        save_metrics(df_error_metric, path=path, static_learner=static_learner, alternate_batch=alternate_batch,
                     transpose=transpose)

        combined_err_metric.append(df_error_metric)
    return (pd.concat(combined_err_metric, ignore_index=True))


def get_summary_table(df_result, df_runtime_result, error_metrics, static_learner=True):
    sum_metric = []
    measure_col_name = 'Metric'

    # Setting start row and column for static and incremental learner
    for metric in error_metrics:
        start_row = 'mean'
        if static_learner:
            start_col = 'RandomForest'
        else:
            start_col = 'HT_Reg'

        df_metric = get_metric_with_mean(df_result, metric)
        df_row = pd.DataFrame([df_metric.loc[start_row][start_col:]])

        df_row[measure_col_name] = str(metric)
        sum_metric.append(df_row)

    # Adding run time
    df_runtime_row = pd.DataFrame([df_runtime_result.describe().loc[start_row][start_col:]])
    df_runtime_row[measure_col_name] = 'Time(sec)'
    sum_metric.append(df_runtime_row)

    df_summary = pd.concat(sum_metric, ignore_index=True)
    df_summary.set_index(measure_col_name, inplace=True)

    return df_summary


def get_summary_table_countrywise(df_result_dict, error_metrics, static_learner=True):  # df_runtime_result,
    summary_metric = []
    measure_col_name = f'Country({str(error_metrics[0])})'
    eval_measure_col = 'EvaluationMeasurement'
    start_row = 'mean'
    if static_learner:
        start_col = 'RandomForest'
    else:
        start_col = 'HT_Reg'

    for country in df_result_dict.keys():
        df_result = df_result_dict[country]

        # Setting start row and column for static and incremental learner
        for metric in error_metrics:
            df_metric = get_metric_with_mean(df_result, metric)
            df_row = pd.DataFrame([df_metric.loc[start_row][start_col:]])
            df_row[eval_measure_col] = metric
            df_row[measure_col_name] = country
            summary_metric.append(df_row)

    df_summary = pd.concat(summary_metric, ignore_index=True)
    df_summary.set_index(measure_col_name, inplace=True)

    return df_summary


def get_sum_table_combined_mean(countrywise_error_score_incremental, results_runtime, static_learner=False):
    sum_table_combined_mean = []
    measure_col_name = 'Metric'
    start_row = 'mean'
    if static_learner:
        start_col = 'RandomForest'
    else:
        start_col = 'HT_Reg'

    for metric in error_metrics:
        df_sum_cur_metric = get_summary_table_countrywise(countrywise_error_score_incremental, [metric],
                                                          static_learner=static_learner)
        df_row = pd.DataFrame([df_sum_cur_metric.describe().loc[start_row]])

        df_row[measure_col_name] = metric
        sum_table_combined_mean.append(df_row)

    # Adding run time
    df_runtime = pd.concat(results_runtime, ignore_index=True).describe().loc[start_row][start_col:]
    df_runtime_row = pd.DataFrame([df_runtime])
    df_runtime_row[measure_col_name] = 'Time(sec)'
    sum_table_combined_mean.append(df_runtime_row)

    # Concating results to one dataframe
    sum_table_combined_mean = pd.concat(sum_table_combined_mean, ignore_index=True)
    sum_table_combined_mean.set_index(measure_col_name, inplace=True)
    return sum_table_combined_mean


def check_significance(target_pop, competitor_pop, significance_at: float):
    """
    Comparing algorithms per batch or per country pairs (exp 2 or 1 respectively), 
      so for each pair, we compare the significance of the best algo to all of the the other algos.
    Ttest performed if the distribution is normal, otherwise we perform a non-parametric test.
    """
    model_pop, population = target_pop, competitor_pop

    # Normality tests
    if len(model_pop) >= 8:  # skew test not valid for smaller populations
        value_mdl, p_mdl = normaltest(model_pop.values)
        value_pop, p_pop = normaltest(population.values)
        if (p_mdl >= 0.05) & (p_pop >= 0.05):
            # print('It is likely that both populations are normal. Thus, running T-Test...')
            tset, pval = stats.ttest_ind(model_pop, population)
            if pval < significance_at:  # alpha value is 0.05 or 5%
                significant = 'Significant (Ttest)'
            else:
                significant = 'Not Significant (Ttest)'
        else:
            # print('It is unlikely that the result is normal. Thus, running Wilcoxon test...')
            if np.sum(np.subtract(list(model_pop), list(
                    competitor_pop))) != 0.0:  # if values are identical the test will crash, but we now it's not significant
                tset, pval = stats.wilcoxon(model_pop, population)
                if pval < significance_at:  # alpha value is 0.05 or 5%
                    significant = 'Significant (Wilcox Test)'
                else:
                    significant = 'Not Significant (Wilcox Test)'
            else:
                # print('Warning: results are identical')
                tset, pval = stats.ttest_ind(model_pop, population)
                significant = 'Not Significant (Wilcox Test)'
    else:
        print('Population too small.')
        if np.sum(np.subtract(list(model_pop), list(
                competitor_pop))) != 0.0:  # if values are identical the test will crash, but we now it's not significant
            tset, pval = stats.wilcoxon(model_pop, population)
            if pval < significance_at:  # alpha value is 0.05 or 5%
                significant = 'Significant (Wilcox Test)'
            else:
                significant = 'Not Significant (Wilcox Test)'
    return pval, significant


"""## Combining Dataset"""


def sortby_date_and_set_index(df):  # Updated
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values('date', inplace=True)
    # df.set_index('date', inplace=True) #TODO: Might need to remove this line later, Not setting date as idx
    return df


def get_dataset_with_target(countries, df_grouped):
    # Empty list to store Dataframes of each country
    frames = []

    for country in countries:
        df = df_grouped.get_group(country)

        # Creating feature 'day_no'
        df['day_no'] = pd.Series([i for i in range(1, len(df) + 1)], index=df.index)

        # Reordering features
        # df = df[['day_no', 'country', 'cases']]
        df = df[['date', 'day_no', 'country', 'cases']]  # Added: Date column

        # Adding features through lags
        df = create_features_with_lags(df)

        # Creating target with last 10 days cases
        idx_cases = list(df.columns).index('cases')  # Added: Earlier hard coded idx
        df['target'] = df.iloc[:, [idx_cases] + [i * -1 for i in range(1, 10)]].mean(axis=1)  # Updated: Replacing idx with idx_cases

        # Dropping null columns
        df.dropna(how='any', axis=0, inplace=True)

        # Dropping mid columns
        drop_columns = list(df.loc[:, 'cases_t-39':'cases_t-1'].columns)  # Updated: cases_t-38 to t-39 for exact 50 columns of lags
        df.drop(drop_columns, axis=1, inplace=True)

        frames.append(df)

    return (pd.concat(frames, ignore_index=True))


def reshape_dataframe(*data: np.ndarray):
    # This function adds an extra dimension which is necessary in the LSTM
    arr = []
    for d in data:
        arr.append(np.reshape(np.array(d), (d.shape[0], 1, d.shape[1])))
    return arr


def get_countries_sortedby_cases(valid_countries, df_grouped):
    # A dictionary of all countries
    dict_countries = Counter(valid_countries)

    for country in dict_countries.keys():
        dict_countries[country] = df_grouped.get_group(country)['cases'].sum()

    # Sorting countries based on number of cases
    countries_sortedby_cases = sorted(dict_countries.items(), key=lambda dict_countries: dict_countries[1],
                                      reverse=True)

    # Creating dataframe
    df_countries_sortedbycases = pd.DataFrame.from_dict(dict(countries_sortedby_cases), orient='index',
                                                        columns=['Total Cases'])

    return df_countries_sortedbycases


# Getting a list of valid countries
def get_countries_with_valid_size(df):
    total_countries = list(df_grouped.groups.keys())

    # A list for countries with required datasize
    valid_countries = []

    # List of countries with more than 230 records. Because, max training size = 150, lags removed = 50, prediction = 30.
    for country in total_countries:
        if len(df_grouped.get_group(country)) >= 230:
            valid_countries.append(country)

    return valid_countries


def preprocess_dataset(df):
    # Selecting required features
    df = df[['dateRep', 'cases', 'countriesAndTerritories']]

    # Rename features
    df.rename(columns={'countriesAndTerritories': 'country', 'dateRep': 'date'}, inplace=True)

    # Convert to date, sort and set index
    df = sortby_date_and_set_index(df)

    return df


# Calculating maximum of dataframe for every pretrain size
def calc_max_of_pretrain_days(pretrain_days, df) -> list:
    max_of_pretrain_days = []

    for day in pretrain_days:
        df_subset = create_subset(df, day)
        max_of_pretrain_days.append(df_subset['cases'].max())

    return max_of_pretrain_days


def display_scores(results):
    # print(f'_________________________________{country}____________________________________________')
    df_MAE = get_metric_with_mean(results, 'MAE')
    df_RMSE = get_metric_with_mean(results, 'RMSE')
    df_MAPE = get_metric_with_mean(results, 'MAPE')
    print('MAE Score')
    print(df_MAE.to_string())
    print('-----------------------------------------------------------------------------------')
    print('RMSE Score')
    print(df_RMSE.to_string())
    print('-----------------------------------------------------------------------------------')
    print('MAPE Score')
    print(df_MAPE.to_string())
    print('\n\n')


"""## Alternate Batch"""


def get_alternate_batch_records_idx(batch_size, total_records):
    total_batches = total_records // batch_size
    current_batch = 1
    start_idx = 0
    end_idx = batch_size
    idx_list = []

    while current_batch <= total_batches:
        if current_batch % 2 != 0:
            idx_list.extend([x for x in range(start_idx, end_idx)])
            start_idx = idx_list[-1] + (batch_size + 1)
            end_idx = start_idx + batch_size
        current_batch += 1

    '''Added code for odd records'''
    if total_batches == 0 and total_records != 0:  # records less the batch size
        idx_list.extend([x for x in range(start_idx, total_records)])

    elif total_batches % 2 == 0 and total_records % batch_size != 0:  # few records present in odd batches
        extra_records = total_records % batch_size
        s = idx_list[-1]+(batch_size+1)  # index start
        idx_list.extend([x for x in range(s, s+extra_records)])

    return idx_list


def create_alternate_batch_subset(df, days, batch_size):
    df_grouped = df.groupby('country')
    countries = df['country'].unique()
    frame1, frame2 = [], []

    for country in countries:
        df_cur_country = df_grouped.get_group(country)

        df1 = df_cur_country.iloc[0:days // 2]
        df2 = df_cur_country.iloc[days:days + 30]  # Adding 30 for a testing batch that is one month ahead

        # Selecting alternate batches
        idx = get_alternate_batch_records_idx(batch_size, total_records=len(df2))
        df2 = df2.iloc[idx]

        # Appending dataframes
        frame1.append(df1)
        frame2.append(df2)

    r1 = pd.concat(frame1, ignore_index=True)
    r2 = pd.concat(frame2, ignore_index=True)
    r = r1.append(r2, ignore_index=True)

    return r


"""## Incremental Learner"""


def instantiate_regressors():
    ht_reg = HoeffdingTreeRegressor()
    hat_reg = HoeffdingAdaptiveTreeRegressor()
    arf_reg = AdaptiveRandomForestRegressor()
    pa_reg = PassiveAggressiveRegressor(max_iter=1, random_state=0, tol=1e-3)

    model = [ht_reg, hat_reg, arf_reg, pa_reg]
    model_names = ['HT_Reg', 'HAT_Reg', 'ARF_Reg', 'PA_Reg']

    return model, model_names


def get_error_scores_per_model(evaluator, mdl_evaluation_scores, inc_alt_batches=False) -> pd.DataFrame:
    for i in range(len(evaluator.model_names)):
        # Desired error metrics
        mse = evaluator.mean_eval_measurements[i].get_mean_square_error()
        mae = evaluator.mean_eval_measurements[i].get_average_error()
        if not inc_alt_batches:
            mae = mae[0]  # get_average_error() is returning a List instead of single value.
        mape = evaluator.mean_eval_measurements[i].get_mean_absolute_percentage_error()
        rmse = sqrt(mse)

        # Dictionary of errors per model
        mdl_evaluation_scores[str(evaluator.model_names[i])] = [rmse, mae, mape]

    return (pd.DataFrame(mdl_evaluation_scores))


def get_running_time_per_model_incremental_learner(evaluator, day):
    cols = ['PretrainDays']  # Adding pretrain as first column
    cols += evaluator.model_names  # Adding remaining columns of different algorithm
    running_time = []
    running_time.append(day)
    for i in range(len(evaluator.model_names)):
        running_time.append(evaluator.running_time_measurements[i]._total_time)

    return (pd.DataFrame([running_time], columns=cols))  # Passing running_time as a list of list to insert it as a row


def display_countrywise_scores(country, df_error_metric):
    print(f'_________________________________{country}____________________________________________')
    print(df_error_metric.to_string())
    print('\n\n')


# Create a dataframe of all countries with pre-train size = pretrain days and test&train size = pretrain days
def create_subset(result, days):
    result_grouped = result.groupby('country')
    countries = result['country'].unique()
    frame1, frame2 = [], []
    for country in countries:
        df = result_grouped.get_group(country)
        df1 = df.iloc[0:days]
        df2 = df.iloc[days:days + 30]
        frame1.append(df1)
        frame2.append(df2)

    r1 = pd.concat(frame1, ignore_index=True)
    r2 = pd.concat(frame2, ignore_index=True)
    r = r1.append(r2, ignore_index=True)

    return r


"""## Static Learner"""


def mean_absolute_percentage_error(actual, predicted):
    """
    Mean absolute percentage error (MAPE).
    :return error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0
    return (np.fabs(actual - predicted) / np.fabs(actual))[mask].mean()


def get_scores(y_true, model_predictions, days):
    mdl_evaluation_scores = {}
    mdl_evaluation_scores['EvaluationMeasurement'] = ['RMSE', 'MAE', 'MAPE']
    mdl_evaluation_scores['PretrainDays'] = [days] * len(mdl_evaluation_scores['EvaluationMeasurement'])

    for model in model_predictions:
        y_pred = model_predictions[model]
        if model == 'LSTM':
            rmse = mean_squared_error(y_true[:, np.newaxis], y_pred, squared=False)
            mae = mean_absolute_error(y_true[:, np.newaxis], y_pred)
            mape = mean_absolute_percentage_error(y_true[:, np.newaxis], y_pred)
        else:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)

        mdl_evaluation_scores[model] = [rmse, mae, mape]
    return pd.DataFrame(mdl_evaluation_scores)


def get_running_time_per_model_static_learner(model_predictions, total_execution_time):
    cols = ['PretrainDays']
    cols += model_predictions.keys()
    return pd.DataFrame(total_execution_time, columns=cols)


def measure(wrapped_func):
    @wraps(wrapped_func)
    def _time_it(*args, **kwargs):
        start = pc_timer()
        try:
            model_predictions = wrapped_func(*args, **kwargs)
        finally:
            end_ = pc_timer() - start
            return model_predictions, end_

    return _time_it


@measure
def train_test_model(regressor, X_train, y_train, X_test):
    regressor.fit(X_train, y_train)
    return regressor.predict(X_test)


@measure
def train_test_lstm(regressor, X_train_lstm, y_train, X_val_lstm, y_val, X_test_lstm, patience, epochs,
                    batch_size_lstm):
    regressor.compile(loss='mae', optimizer='adagrad', metrics=['mse', 'mae'])

    history = regressor.fit(
        X_train_lstm,
        y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=epochs,
        batch_size=batch_size_lstm,
        callbacks=[EarlyStopping(monitor='val_loss',
                                 mode='min',
                                 patience=patience)])

    return regressor.predict(X_test_lstm)


def define_lstm_model(x_train_lstm, layers, activations, patience):
    # Start defining the model
    input_shape = x_train_lstm.shape

    # Definining model first with LSTM n layers
    model = Sequential()
    model.add(LSTM(layers[0], input_shape=input_shape[1:], activation=activations[0], return_sequences=True))

    # Adding middle layers
    for l in range(1, len(layers) - 1):
        model.add(LSTM(layers[l], activation=activations[l], return_sequences=True))
        model.add(Dropout(0.2))

    # Add last Dense and LSTMs layers
    if len(layers) > 1:
        model.add(Dense(layers[-1], activation=activations[-1]))
        model.add(Dropout(0.2))
        model.add(LSTM(layers[-1], activation=activations[-1]))

    model.add(Dense(1))  # output layer. Since we have only 1 output value
    # End defining model

    return model


def normalize_dataset(*dataframes):
    arr = []
    for df in dataframes:
        arr.append(StandardScaler().fit_transform(df))
    return arr


def get_validation_set(df_train, batch_size=10):  # Updated Now
    '''
    lst_idx = -1
    total_batches = len(df_train) // batch_size
    train_set, val_set = [], []

    for cur_batch in range(total_batches):
        start = lst_idx + 1
        end = start + batch_size
        if cur_batch % 2 == 0:
            train_set.append(df_train.iloc[start:end])
        else:
            val_set.append(df_train.iloc[start:end])

        lst_idx = end - 1  # adjusting last index because we add 1 in starting
    '''
    train_set, val_set = [], []
    countries = df_train['country'].unique()
    for country in countries:
        train_set.append(df_train[df_train['country'] == country].iloc[:-batch_size, :])
        val_set.append(df_train[df_train['country'] == country].iloc[-batch_size:])
    return pd.concat(train_set, ignore_index=True), pd.concat(val_set, ignore_index=True)


def reset_evaluator(evaluator):
    for j in range(evaluator.n_models):
        evaluator.mean_eval_measurements[j].reset()
        evaluator.current_eval_measurements[j].reset()
    return evaluator


def update_incremental_metrics(evaluator, y, prediction):
    for j in range(evaluator.n_models):
        for i in range(len(prediction[0])):
            evaluator.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
            evaluator.current_eval_measurements[j].add_result(y[i], prediction[j][i])

        # Adding result manually causes y_true_vector to have a objects inserted like array([123.45]) in a list.
        # For calculating metrics we have to convert them into flat list.
        evaluator.mean_eval_measurements[j].y_true_vector = np.array(evaluator.mean_eval_measurements[j].y_true_vector).flatten().tolist()
        evaluator.current_eval_measurements[j].y_true_vector = np.array(evaluator.current_eval_measurements[j].y_true_vector).flatten().tolist()
    return evaluator


def unit_incremental_df(country_name, evaluator, date, milestone):  # Added Now
    frame = {}
    frame['date'] = date
    if type(country_name) == pd.Series:
        frame['Country'] = country_name
    else:
        frame['Country'] = [country_name] * len(date)
    frame['Milestone'] = [milestone] * len(date)
    frame['y_true'] = evaluator.mean_eval_measurements[0].y_true_vector
    for i in range(len(evaluator.model_names)):
        frame[f'pred_{evaluator.model_names[i]}'] = evaluator.mean_eval_measurements[i].y_pred_vector
    return pd.DataFrame(frame)


def unit_static_df(country_name, date, y_true,  milestone, model_predictions):  # Added Now
    frame = {}
    frame['date'] = date
    if type(country_name) == pd.Series:
        frame['Country'] = country_name
    else:
        frame['Country'] = [country_name] * len(date)

    frame['Milestone'] = [milestone] * len(date)
    frame['y_true'] = y_true

    for algo, y_pred in model_predictions.items():  # Updated Now
        if algo == 'LSTM':
            frame[f'pred_{algo}'] = y_pred.flatten().tolist()
        else:
            frame[f'pred_{algo}'] = y_pred

    return pd.DataFrame(frame)


def save_united_df(df, path, country=None):
    if country:
        df.to_csv(f'{path}/{country}.csv')
    else:
        df.to_csv(f'{path}/united_df.csv')

# Training all countries
results_static = []
results_incremental = []
results_runtime_static = []
results_runtime_incremental = []
max_of_pretrain_per_country = []
max_cases_per_country = []

exp1_static_united_df_path = r'content/Result/exp1/united_dataframe/static'
exp1_inc_united_df_path = r'content/Result/exp1/united_dataframe/incremental'




import sys
def get_total_size(obj):
    size = 0
    for cur_obj in obj:
        size+= sys.getsizeof(cur_obj)/1024
    return size

# Scikit Multiflow Normal
'''
def scikit_multiflow(df, pretrain_days, country):  # Added Country in parameter
    # Creating a stream from dataframe
    stream = DataStream(np.array(df.iloc[:, 4:-1]), y=np.array(df.iloc[:, -1]))  # Selecting features x=[t-89:t-39] and y=[target].

    model, model_names = instantiate_regressors()

    frames, running_time_frames = [], []

    united_dataframe = []  # Added Now

    # Setup the evaluator
    for day in pretrain_days:
        pretrain_days = day
        # max_samples = pretrain_days + 30  # Training and then testing on set one month ahead only
        max_samples = pretrain_days + 1
        testing_samples_size = 30

        evaluator = EvaluatePrequential(show_plot=True,  #False
                                        pretrain_size=pretrain_days,
                                        metrics=['mean_square_error', 'mean_absolute_error',
                                                 'mean_absolute_percentage_error'],
                                        max_samples=max_samples)

        # Run evaluation
        evaluator.evaluate(stream=stream, model=model, model_names=model_names)

        X = stream.X[pretrain_days: pretrain_days + testing_samples_size]
        y = stream.y[pretrain_days: pretrain_days + testing_samples_size]
        target_dates = df.iloc[pretrain_days: pretrain_days +testing_samples_size, 0]  # Added Now

        prediction = evaluator.predict(X)

        # Since we add one extra sample, reset the evaluator
        evaluator = reset_evaluator(evaluator)

        evaluator = update_incremental_metrics(evaluator, y, prediction)

        united_dataframe.append(unit_incremental_df(country, evaluator, target_dates, pretrain_days))  # Added now

        # Dictionary to store each iteration error scores
        mdl_evaluation_scores = {}

        # Adding Evaluation Measurements and pretraining days
        mdl_evaluation_scores['EvaluationMeasurement'] = ['RMSE', 'MAE', 'MAPE']
        mdl_evaluation_scores['PretrainDays'] = [day] * len(mdl_evaluation_scores['EvaluationMeasurement'])
        mdl_evaluation_df = get_error_scores_per_model(evaluator, mdl_evaluation_scores)

        # Errors of each model on a specific pre-train days
        frames.append(mdl_evaluation_df)

        # Run time for each algorithm
        running_time_frames.append(get_running_time_per_model_incremental_learner(evaluator, day))

    # Final Run Time DataFrame
    running_time_df = pd.concat(running_time_frames, ignore_index=True)

    # Final Evaluation Score Dataframe
    evaluation_scores_df = pd.concat(frames, ignore_index=True)

    united_dataframe = pd.concat(united_dataframe, ignore_index=True)  # Added Now
    return evaluation_scores_df, running_time_df, united_dataframe  # Added united_dataframe in return statement

for country in countries:
    # Read each country
    df_country = pd.read_csv(f'{csv_processed_path}/{country}.csv')

    # Get evaluation scores and running time for country
    evaluation_scores_df, running_time_df, united_dataframe = scikit_multiflow(df_country,pretrain_days, country)

    # save_united_df(united_dataframe, exp1_inc_united_df_path, country=country)

    # Appending evaluation scores and runtime for each country
    results_incremental.append(evaluation_scores_df)

    results_runtime_incremental.append(running_time_df)

    # Get max of each pretrain subset and for each country dataset
    max_of_pretrain_per_country.append(calc_max_of_pretrain_days(pretrain_days,df_country))
    max_cases_per_country.append(df_country['cases'].max())

'''

# Scikit Multiflow Prequential

num_selected_countries = 50

df = pd.read_csv(path)

# Pre-processing dataset
df = preprocess_dataset(df)

# Grouping records by country
df_grouped = df.groupby('country')

# Taking only those countries which have sufficient data records
#valid_countries = get_countries_with_valid_size(df_grouped)

# Sorting countries by number of cases
df_countries_sortedbycases = get_countries_sortedby_cases(valid_countries, df_grouped)

# Taking only top selected countries
top_selected_countries = df_countries_sortedbycases.iloc[0:num_selected_countries].index

# Calculating targets and lags for the above countries
result = get_dataset_with_target(top_selected_countries,df_grouped)

# Getting max of each subset in pretrain size
max_of_pretrain_days = calc_max_of_pretrain_days(pretrain_days,result)

# Mean of top selected countries
max_selected_countries = result['cases'].max()

def scikit_multiflow_alternate_batch(df, pretrain_days):

    model, model_names = instantiate_regressors()

    len_countries = len(df['country'].unique())

    # Selecting only required countries
    df = df[df['country'].isin(df['country'].unique()[0:len_countries])]  # Added Now

    frames, running_time_frames = [], []

    united_dataframe = []  # Added Now

    # Setup the evaluator
    for day in pretrain_days:

        df_subset = create_alternate_batch_subset(df, day, batch_size=10)

        # Creating a stream from dataframe
        stream = DataStream(np.array(df_subset.iloc[:, 4:-1]), y=np.array(df_subset.iloc[:, -1]))

        pretrain_size = (day//2) * len_countries
        max_samples = len(df_subset)  # (day//2 + 30) * len_countries  # Testing on set one month ahead only

        evaluator = EvaluatePrequential(show_plot=True,  # False
                                    pretrain_size=pretrain_size,
                                    metrics = ['mean_square_error', 'mean_absolute_error', 'mean_absolute_percentage_error'],
                                    max_samples=max_samples)
        # Run evaluation
        evaluator.evaluate(stream=stream, model=model, model_names=model_names)

        date_idx = list(df_subset.columns).index('date')  # Added Now
        country_idx = list(df_subset.columns).index('country')  # Added Now

        target_dates = df_subset.iloc[pretrain_size: max_samples, date_idx]  # Added Now
        subset_countries_names = df_subset.iloc[pretrain_size:max_samples, country_idx]  # Added Now

        united_dataframe.append(unit_incremental_df(subset_countries_names, evaluator, target_dates, day))  # Added now

        # Dictionary to store each iteration error scores
        mdl_evaluation_scores = {}

        # Adding Evaluation Measurements and pretraining days
        mdl_evaluation_scores['EvaluationMeasurement'] = ['RMSE', 'MAE', 'MAPE'] #,'MSE']
        mdl_evaluation_scores['PretrainDays'] = [day//2] * len(mdl_evaluation_scores['EvaluationMeasurement'])
        mdl_evaluation_df = get_error_scores_per_model(evaluator, mdl_evaluation_scores, inc_alt_batches=True)

        # Errors of each model on a specific pre-train days
        frames.append(mdl_evaluation_df)

        # Run time for each algorithm
        running_time_frames.append(get_running_time_per_model_incremental_learner(evaluator, day))

    # Final Run Time DataFrame
    running_time_df = pd.concat(running_time_frames,ignore_index=True)

    united_df = pd.concat(united_dataframe, ignore_index=True)

    # Final Evaluation Score Dataframe
    evaluation_scores_df = pd.concat(frames, ignore_index=True)
    return evaluation_scores_df, running_time_df, united_df

result_skmlflow_alternate_batch, running_time_incremental_alternate_batch, united_df = scikit_multiflow_alternate_batch(result, pretrain_days)
