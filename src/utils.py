from collections import Counter
from functools import wraps
from time import perf_counter as pc_timer
from math import sqrt

import numpy as np
import pandas as pd
import yaml
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras import Sequential
from keras.models import Sequential
from scipy import stats

# For significance tests
from scipy.stats import normaltest
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_absolute_error

# Imports for static Learner
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor

# Imports for incremental learner
from skmultiflow.trees import HoeffdingTreeRegressor


# Load all global variables from YAML file
# yaml_file_path = "vars.yaml"
yaml_file_path = "../config.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

decimal = parsed_yaml_file['decimal']  # Specify the scale of decimal places
error_metrics = parsed_yaml_file['error_metrics']
valid_countries = parsed_yaml_file['valid_countries']


# COMMON DATASET

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
            df.to_latex(f'{path}/combined_country_runtime_static.tex')
            df.to_csv(f'{path}/combined_country_runtime_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined_country_runtime_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined_country_runtime_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined_country_runtime_incremental.tex')
                df.to_csv(f'{path}/combined_country_runtime_incremental.csv')
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
            df.to_latex(f'{path}/combined_country_summary_table_static.tex')
            df.to_csv(f'{path}/combined_country_summary_table_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined_country_summary_table_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined_country_summary_table_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined_country_summary_table_incremental.tex')
                df.to_csv(f'{path}/combined_country_summary_table_incremental.csv')


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
            df.to_latex(f'{path}/combined_country_{metric_type}_static.tex')
            df.to_csv(f'{path}/combined_country_{metric_type}_static.csv')
        else:
            if alternate_batch:
                df.to_latex(f'{path}/combined_country_{metric_type}_incremental_alternate_batch.tex')
                df.to_csv(f'{path}/combined_country_{metric_type}_incremental_alternate_batch.csv')
            else:
                df.to_latex(f'{path}/combined_country_{metric_type}_incremental.tex')
                df.to_csv(f'{path}/combined_country_{metric_type}_incremental.csv')
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


def save_united_df(df, path, country=None):
    if country:
        df.to_csv(f'{path}/{country}.csv')
    else:
        df.to_csv(f'{path}/united_df.csv')


def display_runtime_per_country(results_runtime, countries):
    for i in range(len(countries)):
        print(f'_____________Running Time for {countries[i]}________________')
        print(results_runtime[i].to_string())
        print('\n')


def display_countrywise_scores(country, df_error_metric):
    print(f'_________________________________{country}____________________________________________')
    print(df_error_metric.to_string())
    print('\n\n')


def calc_save_err_metric_countrywise(countries, error_metrics, results, max_of_pretrain_per_country,
                                     max_cases_per_country, path, static_learner, transpose):
    countrywise_error_scores = {}
    for i in range(len(countries)):
        country_error_score = []
        for error_metric in error_metrics:
            df_error_metric = get_metric_with_mean(results[i], error_metric=error_metric)

            # if error_metric != 'MAPE':
            #  df_error_metric = calc_mean_to_max_error(df_error_metric, max_of_pretrain_per_country[i], max_cases_per_country[i])

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


def unit_static_df(country_name, date, y_true, milestone, model_predictions):  # Added Now
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


# COMBINING DATASET

def sortby_date_and_set_index(df):  # Updated
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df.sort_values('date', inplace=True)
    # df.set_index('date', inplace=True) #TODO:  Not setting date as idx; Might need to remove this line later
    return df


# Create lags
def create_features_with_lags(df):
    for i in range(89, 0,
                   -1):  # Loop in reverse order for creating ordered lags eg: cases_t-10, cases_t-9... cases_t-1. t=current cases
        df[f'cases_t-{i}'] = df['cases'].shift(i, axis=0)
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
        df['target'] = df.iloc[:, [idx_cases] + [i * -1 for i in range(1, 10)]].mean(
            axis=1)  # Updated: Replacing idx with idx_cases

        # Dropping null columns
        df.dropna(how='any', axis=0, inplace=True)

        # Dropping mid columns
        drop_columns = list(
            df.loc[:, 'cases_t-39':'cases_t-1'].columns)  # Updated: cases_t-38 to t-39 for exact 50 columns of lags
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


def preprocess_dataset(df):
    # Selecting required features
    df = df[['dateRep', 'cases', 'countriesAndTerritories']]

    # Rename features
    df.rename(columns={'countriesAndTerritories': 'country', 'dateRep': 'date'}, inplace=True)

    # Convert to date, sort and set index
    df = sortby_date_and_set_index(df)

    return df


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

    return (r)


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


# ALTERNATE BATCH

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

    elif total_batches % 2 == 0 and total_records % batch_size != 0:  # if few extra records present in odd batches
        extra_records = total_records % batch_size
        s = idx_list[-1] + (batch_size + 1)  # index start
        idx_list.extend([x for x in range(s, s + extra_records)])

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

    return (r)


# INCREMENTAL LEARNER

def instantiate_regressors():
    ht_reg = HoeffdingTreeRegressor()
    hat_reg = HoeffdingAdaptiveTreeRegressor()
    arf_reg = AdaptiveRandomForestRegressor()
    pa_reg = PassiveAggressiveRegressor(max_iter=1, random_state=0, tol=1e-3)

    model = [ht_reg, hat_reg, arf_reg, pa_reg]
    model_names = ['HT_Reg', 'HAT_Reg', 'ARF_Reg', 'PA_Reg']

    return model, model_names


'''
def get_error_scores_per_model(evaluator, mdl_evaluation_scores) -> pd.DataFrame:
    for i in range(len(evaluator.model_names)):
        # Desired error metrics
        mse = evaluator.mean_eval_measurements[i].get_mean_square_error()
        mae = evaluator.mean_eval_measurements[i].get_average_error()
        mape = evaluator.mean_eval_measurements[i].get_mean_absolute_percentage_error()
        rmse = sqrt(mse)

        # Dictionary of errors per model
        mdl_evaluation_scores[str(evaluator.model_names[i])] = [rmse, mae, mape]

    return (pd.DataFrame(mdl_evaluation_scores))
'''


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
    return pd.DataFrame(mdl_evaluation_scores)


def get_running_time_per_model_incremental_learner(evaluator,day):
    cols = ['PretrainDays']  # Adding pretrain as first column
    cols += evaluator.model_names  # Adding remaining columns of different algorithm
    running_time = []
    running_time.append(day)
    for i in range(len(evaluator.model_names)):
        running_time.append(evaluator.running_time_measurements[i]._total_time)

    return pd.DataFrame([running_time], columns=cols)  # Passing running_time as a list of list to insert it as a row


def reset_evaluator(evaluator):
    for j in range(evaluator.n_models):
        evaluator.mean_eval_measurements[j].reset()
        evaluator.current_eval_measurements[j].reset()
    return evaluator


def update_incremental_metrics(evaluator, y, prediction):  # Added Now
    for j in range(evaluator.n_models):
        for i in range(len(prediction[0])):
            evaluator.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
            evaluator.current_eval_measurements[j].add_result(y[i], prediction[j][i])

        # Adding result manually causes y_true_vector to have a objects inserted like array([123.45]) in a list.
        # For calculating metrics we have to convert them into flat list.
        evaluator.mean_eval_measurements[j].y_true_vector = np.array(
            evaluator.mean_eval_measurements[j].y_true_vector).flatten().tolist()
        evaluator.current_eval_measurements[j].y_true_vector = np.array(
            evaluator.current_eval_measurements[j].y_true_vector).flatten().tolist()
    return evaluator


# STATIC LEARNER

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


def get_validation_set(df_train, batch_size=10):
    train_set, val_set = [], []
    countries = df_train['country'].unique()
    for country in countries:
        train_set.append(df_train[df_train['country'] == country].iloc[:-batch_size, :])
        val_set.append(df_train[df_train['country'] == country].iloc[-batch_size:])
    return pd.concat(train_set, ignore_index=True), pd.concat(val_set, ignore_index=True)

