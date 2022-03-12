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

# YAML FILE
parsed_yaml_file = get_configs_yaml()
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
distance_metrics = parsed_yaml_file['distance_metrics']
exp3_distance_path = parsed_yaml_file['paths']['exp3_distance_path']
error_metrics = parsed_yaml_file['error_metrics']
exp3_path = parsed_yaml_file['paths']['exp3_path']


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
            dist = eval(met + f"({list(lag_1.to_numpy().flatten())},{list(lag_2.to_numpy().flatten())})")
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
        This would take around 30 minutes to run for all 50 countries
        """
        print(
            "File doesn't exists! computing distances... \ncaution: may take a while (approx 30 minutes for 50 countries)")
        distances = pairwise_country_distance(country_list, dist_metrics, num_of_country)
        with open(filepath, "wb") as f:
            pickle.dump(distances, f)
    return distances


def sort_by_milestone(milestones, country_distances, by_metric='euclidean'):
    sorted_path = parsed_yaml_file['paths']["exp3_sorted_" + by_metric + "_path"]  # TODO: Remove hard coded path
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


def initialize_static_models():
    static_models = [
        RandomForestRegressor(max_depth=2, random_state=0),
        GradientBoostingRegressor(random_state=0),
        LinearSVR(random_state=0, tol=1e-5),
        DecisionTreeRegressor(random_state=0),
        BayesianRidge()
    ]
    return static_models


def get_static_model_prediction(static_models, x_train, x_test, y_train):
    static_predictions = {type(m).__name__: [] for m in static_models}
    for m in static_models:
        static_predictions[type(m).__name__], exec_time = train_test_model(m, x_train, y_train, x_test)
    return static_predictions


def save_error_metric(scores, metrics, path, static_learner=True, alternate_batch=False, transpose=True):
    for key_country in scores:
        if len(scores[key_country]) > 1:  # dataframe is not empty
            for m in metrics:
                metric_score_df = get_metric_with_mean(scores[key_country], m)
                save_metrics(metric_score_df,
                             path,
                             country=key_country,
                             static_learner=static_learner,
                             alternate_batch=alternate_batch,
                             transpose=transpose)
                print(f'Saved {m} score for country: {key_country}')


def start_static_learning(distances, paths, features):
    # iterate over all source country and closest target countries
    final_score = {source_country: [] for source_country in sorted(distances)[0:2]}  # TODO: remove sorted
    for source_country in sorted(distances)[0:2]:  # TODO: remove sorted
        combined_score_df = []
        for milestone in distances[source_country]:
            closest_countries = distances[source_country][milestone]['country']
            train, test = get_train_test_set(source_country, closest_countries, milestone, paths, features, sort_by='date',
                                             remove_duplicate=False)
            x_train, x_test, y_train, y_test = split_train_test(train, test)
            models = initialize_static_models()
            predictions = get_static_model_prediction(models, x_train, x_test, y_train)
            score_df = get_scores(y_test, predictions, milestone)
            combined_score_df.append(score_df)
        final_score[source_country] = pd.concat(combined_score_df, ignore_index=True)

    save_error_metric(final_score,
                      error_metrics,
                      path=exp3_path,
                      static_learner=True,
                      alternate_batch=False,
                      transpose=True)
    print('Done')


def add_extra_row(df):
    new_index = df[-1:].index.start + 1
    new_value = df[-1:].values
    col = df[-1:].columns
    extra_row = pd.DataFrame(new_value, index=[new_index], columns=col)
    df = df.append(extra_row)
    return df


def start_inc_learning(distances, paths, features):
    final_score = {source_country: [] for source_country in sorted(distances)[0:2]}  # TODO: remove sorted
    for source_country in sorted(distances)[0:2]:  # TODO: remove sorted
        combined_score_df = []
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

        final_score[source_country] = pd.concat(combined_score_df, ignore_index=True)

    save_error_metric(final_score,
                      error_metrics,
                      path=exp3_path,
                      static_learner=False,
                      alternate_batch=False,
                      transpose=True)
    print('Done')


# Find distances among countries
country_wise_distances = find_distance(exp3_distance_path, countries, distance_metrics, num_of_country=50)

# Sort the distances by metric
sorted_distances = sort_by_milestone(pretrain_days, country_wise_distances, by_metric='euclidean')

# For every source country select top n closest countries
top_selected_distances = select_top_n(sorted_distances, n=9)

# a list of features to exclude from train and test
features_to_select = {'start_column': 'cases_t-89', 'end_column': 'target', 'exclude_column': None}

# start_static_learning(top_selected_distances, csv_processed_path, features_to_select)

start_inc_learning(top_selected_distances, csv_processed_path, features_to_select)
