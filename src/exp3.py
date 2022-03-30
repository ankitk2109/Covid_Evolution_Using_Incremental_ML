# Imports

import os
import yaml
import glob
import pickle
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from src.utils import *

# from dtw import dtw
import rust_dtw  # using rust implementation of dtw as it is 10x faster
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import braycurtis

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from skmultiflow.data import DataStream
from src.skmultiflow.src.evaluate_prequential import EvaluatePrequential

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def format_result(country_dict, milestones):
    """
    convert scores to dataframe for each metric
    Parameters
    ----------
    country_dict: current source country dictionary
    milestones: list of milestone days

    Returns
    -------
    formatted dataframe score for each country
    """
    milestone_cols = milestones.copy()
    milestone_cols.append('country')
    for m in country_dict:
        country_dict[m] = pd.DataFrame(country_dict[m], columns=milestone_cols)
    return country_dict


def calc_distance(c1, c2, milestones, dist_metrics, country_dict):
    """
    find distance between country 1 and country 2
    Parameters
    ----------
    c1: first country
    c2: Second country
    milestones: list of milestones eg: 30,60,90,..,240
    dist_metrics: list of metrics
    country_dict: current source country

    Returns
    -------
    list of score for each milestone of current source country
    """
    df_c1 = pd.read_csv(f'{csv_processed_path}/{c1}.csv')
    df_c2 = pd.read_csv(f'{csv_processed_path}/{c2}.csv')
    for met in dist_metrics:
        distances = []  # saves distances for each milestone for every metric
        for m in milestones:
            lag_1 = df_c1.iloc[:m, 4:-1]
            lag_2 = df_c2.iloc[:m, 4:-1]
            if met == 'dtw':
                dist = rust_dtw.dtw(lag_1.to_numpy().flatten(), lag_2.to_numpy().flatten(), window=1, distance_mode="euclidean")
            elif met == 'euclidean':
                dist = euclidean(lag_1.to_numpy().reshape(-1, 1), lag_2.to_numpy().reshape(-1, 1))
            elif met == 'cityblock':
                dist = cityblock(lag_1.to_numpy().reshape(-1, 1), lag_2.to_numpy().reshape(-1, 1))
            elif met == 'braycurtis':
                dist = braycurtis(lag_1.to_numpy().reshape(-1, 1), lag_2.to_numpy().reshape(-1, 1))
            else:
                print(f"Metric \"{m}\" defined")
            distances.append(dist)
        distances.append(c2)
        country_dict[met].append(distances)
    return country_dict


def pairwise_country_distance(country_list, dist_metrics, num_of_country=-1):
    """
    calculate the distance among each country for every metric
    Parameters
    ----------
    country_list: all countries with valid lags
    dist_metrics: list of metrics
    num_of_country: number of countries to calculate distances

    Returns
    -------
    a nested dictionary with countries and metrics. Each metric consist of a dataframe having scores for every milestone
    """
    num_of_country = num_of_country if num_of_country > 0 else len(country_list)
    dist_country = {}
    for i in range(0, num_of_country):  # len(country_list)
        dist_country[country_list[i]] = {}
        dist_country[country_list[i]] = {metric: [] for metric in dist_metrics}  # creating empty list for all metrics
        for j in range(0, num_of_country):  # len(country_list)
            if country_list[i] != country_list[j]:
                dist_country[country_list[i]] = calc_distance(country_list[i],
                                                              country_list[j],
                                                              pretrain_days,
                                                              dist_metrics,
                                                              dist_country[country_list[i]])
        dist_country[country_list[i]] = format_result(dist_country[country_list[i]], pretrain_days)
    return dist_country


def find_distance(filepath, country_list, dist_metrics, num_of_country):
    """
    this methods load the distances if already exist else calculate the distances
    Parameters
    ----------
    filepath: path to distances file
    country_list: all countries with valid lags
    dist_metrics: list of metrics
    num_of_country: number of countries to calculate distances

    Returns
    -------
    loaded or calculated country wise dictionary of metrics and scores
    """
    if os.path.exists(filepath):
        print(f"File {filepath} exists. Loading distances!")
        with open(filepath, "rb") as f:
            distances = pickle.load(f)
    else:
        """
        This would take around 60-90 minutes to run for all 50 countries
        """
        print(
            "File doesn't exists! computing distances... \ncaution: may take a while (approx 30 minutes for 50 countries)")
        distances = pairwise_country_distance(country_list, dist_metrics, num_of_country)
        with open(filepath, "wb") as f:
            pickle.dump(distances, f)
    return distances


def sort_by_milestone(milestones, country_distances, by_metric='euclidean'):
    filename = "sorted_dist_path_" + by_metric + ".pkl"
    sorted_path = os.path.join(parsed_yaml_file['paths']['exp3_sorted_path'], filename)
    if not os.path.exists(sorted_path):
        print(f"{sorted_path} not found! sorting distances by {by_metric}")
        sorted_dict = {country: {m: [] for m in milestones} for country in country_distances}
        for country in country_distances:
            for m in milestones:
                tmp_df = country_distances[country][by_metric].loc[:, [m, 'country']]
                sorted_dict[country][m] = tmp_df.sort_values(by=m).reset_index(drop=True)
        with open(sorted_path, "wb") as f:  # saving
            pickle.dump(sorted_dict, f)
    else:
        print(f"{sorted_path} found! loading distances by {by_metric}")
        with open(sorted_path, "rb") as f:  # loading
            sorted_dict = pickle.load(f)
    return sorted_dict


def select_top_n(sorted_dist, n):
    sorted_top = sorted_dist.copy()
    for country in sorted_top:
        for m in sorted_top[country]:  # m = milestone
            sorted_top[country][m] = sorted_top[country][m].iloc[0:n]
    return sorted_top


def get_train_test_set(source_country, closest_countries, milestone, csv_country_path, features_dict, sort_by=None, remove_duplicate=False):
    frames = [pd.read_csv(f'{csv_country_path}/{c}.csv', nrows=milestone) for c in closest_countries]
    train_df = pd.concat(frames)
    test_df = pd.read_csv(f'{csv_country_path}/{source_country}.csv', nrows=milestone)

    if sort_by:
        train_df = train_df.sort_values(by=sort_by)
        test_df = test_df.sort_values(by=sort_by)

    if remove_duplicate:
        train_df = train_df.drop_duplicates()
        test_df = test_df.drop_duplicates()

    train_df = train_df.loc[:, features_dict['start_column']:features_dict['end_column']].reset_index(drop=True)
    test_df = test_df.loc[:, features_dict['start_column']:features_dict['end_column']].reset_index(drop=True)

    if features_dict['exclude_column']:
        train_df = train_df.drop(features_dict['exclude_column'], axis=1)
        test_df = test_df.drop(features_dict['exclude_column'], axis=1)

    return train_df, test_df


def split_train_test(train_df, test_df):
    x_train_df = train_df.iloc[:, :-1]
    y_train_df = train_df.iloc[:, -1]
    x_test_df = test_df.iloc[:, :-1]
    y_test_df = test_df.iloc[:, -1]
    return x_train_df, x_test_df, y_train_df, y_test_df


def initialize_static_models(lstm_params):
    lstm_model = define_lstm_model(lstm_params['x_train_lstm'],
                                   lstm_params['layers'],
                                   lstm_params['activations'],
                                   lstm_params['patience'])
    static_models = [
        RandomForestRegressor(max_depth=2, random_state=0),
        GradientBoostingRegressor(random_state=0),
        LinearSVR(random_state=0, tol=1e-5),
        DecisionTreeRegressor(random_state=0),
        BayesianRidge(),
        lstm_model
    ]
    return static_models


def get_summary_table_countrywise(df_result_dict, err_metrics, static_learner=True):  # df_runtime_result,
    """
    This method calculates the summary dataframe for exp2 for all metrics
    """
    summary_metric = []
    measure_col_name = f'Country({str(err_metrics[0])})'
    eval_measure_col = 'EvaluationMeasurement'
    start_row = 'mean'
    if static_learner:
        start_col = 'RandomForestRegressor'
    else:
        start_col = 'HT_Reg'

    for country in df_result_dict.keys():
        df_result = df_result_dict[country]

        # converting types to numeric values
        df_result = df_result.apply(pd.to_numeric, errors='ignore')

        # Setting start row and column for static and incremental learner
        for metric in err_metrics:
            df_metric = get_metric_with_mean(df_result, metric)
            df_row = pd.DataFrame([df_metric.loc[start_row][start_col:]])
            df_row[eval_measure_col] = metric
            df_row[measure_col_name] = country
            summary_metric.append(df_row)

    df_summary = pd.concat(summary_metric, ignore_index=True)
    df_summary.set_index(measure_col_name, inplace=True)

    return df_summary


def get_static_model_prediction(static_models, x_train, x_test, y_train, lstm_params):
    static_predictions = {type(m).__name__ if type(m).__name__ != 'Sequential' else 'LSTM': [] for m in static_models}
    for m in static_models:
        if type(m).__name__ == 'Sequential':  # LSTM == Sequential
            static_predictions['LSTM'], exec_time = train_test_lstm(m, lstm_params['x_train_lstm'],
                                                                              lstm_params['y_train_batch'],
                                                                              lstm_params['x_val_lstm'],
                                                                              lstm_params['y_val_batch'],
                                                                              lstm_params['x_test_lstm'],
                                                                              lstm_params['patience'],
                                                                              lstm_params['epochs'],
                                                                              lstm_params['batch_size_lstm'])
        else:
            static_predictions[type(m).__name__], exec_time = train_test_model(m, x_train, y_train, x_test)
    return static_predictions


def get_filename(metric_type, country=None, static_learner=True, alternate_batch=False):
    if country is None:
        if static_learner:
            return f'/combined_country_{metric_type}_static.csv'
        else:
            if alternate_batch:
                return f'/combined_country_{metric_type}_incremental_alternate_batch.csv'
            else:
                return f'combined_country_{metric_type}_incremental.csv'
    else:
        if static_learner:
            return f'/{country}_{metric_type}_static.csv'
        else:
            return f'/{country}_{metric_type}_incremental.csv'


def save_error_metric(src_country, scores, metrics, path, static_learner=True, alternate_batch=False, transpose=True):
    if len(scores) > 1:  # dataframe is not empty
        for m in metrics:
            metric_score_df = get_metric_with_mean(scores, m)
            save_metrics(metric_score_df,
                         path,
                         country=src_country,
                         static_learner=static_learner,
                         alternate_batch=alternate_batch,
                         transpose=transpose)
            print(f'Saved {m} score for country: {src_country}')


def add_extra_row(df):
    new_index = df[-1:].index.start + 1
    new_value = df[-1:].values
    col = df[-1:].columns
    extra_row = pd.DataFrame(new_value, index=[new_index], columns=col)
    df = df.append(extra_row)
    return df


def get_sum_table_combined_mean(countrywise_error_score, static_learner=False):
    sum_table_combined_mean = []
    measure_col_name = 'Metric'
    start_row = 'mean'
    if static_learner:
        start_col = 'RandomForestRegressor'
    else:
        start_col = 'HT_Reg'

    for metric in error_metrics:
        df_sum_cur_metric = get_summary_table_countrywise(countrywise_error_score, [metric],
                                                          static_learner=static_learner)
        df_row = pd.DataFrame([df_sum_cur_metric.describe().loc[start_row]])

        df_row[measure_col_name] = metric
        sum_table_combined_mean.append(df_row)

    # Concat results to one dataframe
    sum_table_combined_mean = pd.concat(sum_table_combined_mean, ignore_index=True)
    sum_table_combined_mean.set_index(measure_col_name, inplace=True)
    return sum_table_combined_mean


def split_train_val_lstm(train, x_test, batch_size, lstm_params):
    train_df = train.iloc[:-batch_size, :]
    val_df = train_df.iloc[-batch_size:]

    x_train_batch, y_train_batch = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    x_val_batch, y_val_batch = val_df.iloc[:, :-1], val_df.iloc[:, -1]

    # normalize
    x_train_lstm_norm, x_test_lstm_norm, x_val_lstm_norm = normalize_dataset(x_train_batch, x_test, x_val_batch)

    # Reshaping the dataframes
    x_train_lstm, x_val_lstm, x_test_lstm = reshape_dataframe(x_train_lstm_norm, x_val_lstm_norm, x_test_lstm_norm)

    lstm_params.update({'x_train_lstm': x_train_lstm,
                        'x_val_lstm': x_val_lstm,
                        'x_test_lstm': x_test_lstm,
                        'y_train_batch': y_train_batch,
                        'y_val_batch': y_val_batch})

    return lstm_params


def start_static_learning(distances, paths, features, err_metric, batch_size_lstm, save_path):
    """

    Parameters
    ----------
    distances: source country to n closest country dictionary
    paths: path to extract csv files for each source and closest country
    features: a dictionary for deciding the columns
    err_metric: list of error metric to calculate fro each country
    batch_size_lstm: batch size to be extracted from each country
    save_path: path to save the results
    -------

    """
    # iterate over all source country and closest target countries
    final_score = {}

    # params (others like epoch and batch size are also hardcoded in train_test_lstm())
    lstm_params = {'layers': [50, 30, 20, 10],
                   'activations': ['tanh', 'tanh', 'relu'],
                   'epochs': 200,
                   'patience': 20,
                   'batch_size_lstm': batch_size_lstm}

    for source_country in distances:
        files_exists, file_paths = [], []

        # check if score files exist
        for m in err_metric:
            f_name = get_filename(m, country=source_country)
            files_exists.append(os.path.exists(f"{save_path}/{f_name}"))
            file_paths.append(f"{save_path}/{f_name}")

        # scores files not present
        if not all(files_exists):
            combined_score_df, combined_score_mean_df = [], []
            # iterate over milestone and predict scores
            for milestone in distances[source_country]:
                closest_countries = distances[source_country][milestone]['country']
                train, test = get_train_test_set(source_country, closest_countries, milestone, paths, features, sort_by='date',
                                                 remove_duplicate=False)
                x_train, x_test, y_train, y_test = split_train_test(train, test)

                total_batch_size = len(closest_countries) * lstm_params['batch_size_lstm']
                lstm_params = split_train_val_lstm(train, x_test, total_batch_size, lstm_params)

                models = initialize_static_models(lstm_params)  # new iteration new model

                predictions = get_static_model_prediction(models, x_train, x_test, y_train, lstm_params)

                score_df = get_scores(y_test, predictions, milestone)

                combined_score_df.append(score_df)

            # all metric score for current source country
            combined_score_df = pd.concat(combined_score_df, ignore_index=True)

            # calculate mean and save scores
            for m in err_metric:
                metric_score_df = get_metric_with_mean(combined_score_df, m)  # calculate mean
                save_metrics(metric_score_df, save_path, country=source_country, transpose=True)  # save score
                combined_score_mean_df.append(metric_score_df)

            final_score[source_country] = pd.concat(combined_score_mean_df)

        # scores file present
        else:
            combined_score_mean_df = [pd.read_csv(filename, index_col='Unnamed: 0').transpose() for filename in file_paths]
            final_score[source_country] = pd.concat(combined_score_mean_df)

    summary_table_countrywise_static = get_summary_table_countrywise(final_score, ['RMSE'], static_learner=True)
    save_summary_table(summary_table_countrywise_static, exp3_summary_path, country=True, static_learner=True,
                       alternate_batch=False, transpose=True)

    sum_static_countrywise_mean = get_sum_table_combined_mean(final_score, static_learner=True)
    save_combined_summary_table(sum_static_countrywise_mean, exp3_summary_path, static_learner=True, transpose=True)


def start_inc_learning(distances, paths, features, err_metric, save_path):
    """

    Parameters
    ----------
    distances: source country to n closest country dictionary
    paths: path to extract csv files for each source and closest country
    features: a dictionary for deciding the columns
    err_metric: list of error metric to calculate fro each country
    save_path: path to save the results
    -------

    """
    final_score = {}
    for source_country in distances:
        files_exists, file_paths = [], []

        # check if score files exist
        for m in err_metric:
            f_name = get_filename(m, country=source_country, static_learner=False)
            files_exists.append(os.path.exists(f"{save_path}/{f_name}"))
            file_paths.append(f"{save_path}/{f_name}")

        # scores files not present
        if not all(files_exists):
            combined_score_df, combined_score_mean_df = [], []
            # iterate over milestone and predict scores
            for milestone in distances[source_country]:
                models, model_names = instantiate_regressors()
                closest_countries = distances[source_country][milestone]['country']
                train, test = get_train_test_set(source_country, closest_countries, milestone, paths, features,
                                                 sort_by='date', remove_duplicate=False)
                train = add_extra_row(train)  # add extra row for pretraining
                x_train, x_test, y_train, y_test = split_train_test(train, test)
                train_stream = DataStream(np.array(x_train), y=np.array(y_train))
                test_stream = DataStream(np.array(x_test), y=np.array(y_test))
                pretrain_size = milestone * len(closest_countries)
                max_samples = pretrain_size + 1  # One Extra Sample
                evaluator = EvaluatePrequential(show_plot=False,
                                                pretrain_size=pretrain_size,
                                                metrics=['mean_square_error',
                                                         'mean_absolute_error',
                                                         'mean_absolute_percentage_error'],
                                                max_samples=max_samples)
                evaluator.evaluate(stream=train_stream, model=models, model_names=model_names)
                predictions = evaluator.predict(test_stream.X)
                evaluator = reset_evaluator(evaluator)  # evaluated on 1 sample, reset it
                evaluator = update_incremental_metrics(evaluator, test_stream.y, predictions)
                mdl_evaluation_scores = {'EvaluationMeasurement': ['RMSE', 'MAE', 'MAPE']}
                mdl_evaluation_scores['PretrainDays'] = [milestone] * len(mdl_evaluation_scores['EvaluationMeasurement'])
                score_df = get_error_scores_per_model(evaluator, mdl_evaluation_scores)
                combined_score_df.append(score_df)

            # all metric score for current source country
            combined_score_df = pd.concat(combined_score_df, ignore_index=True)

            # calculate mean and save scores
            for m in err_metric:
                metric_score_df = get_metric_with_mean(combined_score_df, m)  # calculate mean
                save_metrics(metric_score_df, save_path, country=source_country, static_learner=False, transpose=True)  # save score
                combined_score_mean_df.append(metric_score_df)

            final_score[source_country] = pd.concat(combined_score_mean_df)

        # scores file present
        else:
            combined_score_mean_df = [pd.read_csv(filename, index_col='Unnamed: 0').transpose() for filename in
                                      file_paths]
            final_score[source_country] = pd.concat(combined_score_mean_df)

    summary_table_countrywise_inc = get_summary_table_countrywise(final_score, ['RMSE'], static_learner=False)
    save_summary_table(summary_table_countrywise_inc, exp3_summary_path, country=True, static_learner=False,
                       alternate_batch=False, transpose=True)

    sum_inc_countrywise_mean = get_sum_table_combined_mean(final_score, static_learner=False)
    save_combined_summary_table(sum_inc_countrywise_mean, exp3_summary_path, static_learner=False, transpose=True)


# YAML FILE
parsed_yaml_file = get_configs_yaml()
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
distance_metrics = parsed_yaml_file['distance_metrics']
exp3_distance_path = parsed_yaml_file['paths']['exp3_distance_path']
error_metrics = parsed_yaml_file['error_metrics']
exp3_path = parsed_yaml_file['paths']['exp3_path']
exp3_summary_path = parsed_yaml_file['paths']['exp3_summary_path']
batch_size_lstm = parsed_yaml_file['batch_size_lstm']
exp4_path = r"C:\Ankit\Personal\Github\Covid_Evolution_Using_Incremental_ML\results\running_results\content\Result\exp4"


# Find distances among countries
country_wise_distances = find_distance(exp3_distance_path, countries, distance_metrics, num_of_country=50)

# Sort the distances by metric
sorted_distances = sort_by_milestone(pretrain_days, country_wise_distances, by_metric='dtw')

# For every source country select top n closest countries
top_selected_distances = select_top_n(sorted_distances, n=9)

# a list of features to exclude from train and test
features_to_select = {'start_column': 'cases_t-89', 'end_column': 'target', 'exclude_column': None}

# static learning
start_static_learning(top_selected_distances, csv_processed_path, features_to_select, error_metrics, batch_size_lstm, exp3_path)

# incremental learning
start_inc_learning(top_selected_distances, csv_processed_path, features_to_select, error_metrics, exp3_path)


