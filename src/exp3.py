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

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# YAML FILE
parsed_yaml_file = get_configs_yaml()
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
distance_metrics = parsed_yaml_file['distance_metrics']
exp3_distance_path = parsed_yaml_file['paths']['exp3_distance_path']


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
        for milestone in milestones:
            lag_1 = df_c1.iloc[:milestone, 4:-1]
            lag_2 = df_c2.iloc[:milestone, 4:-1]
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
        print("File exists. Loading distances!")
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
    sorted_dict = {country: {milestone: [] for milestone in milestones} for country in country_distances}
    for country in country_distances:
        for milestone in milestones:
            tmp_df = country_distances[country][by_metric].loc[:, [milestone, 'country']]
            sorted_dict[country][milestone] = tmp_df.sort_values(by=milestone)
    return sorted_dict


country_wise_distances = find_distance(exp3_distance_path, countries, distance_metrics, num_of_country=50)
sorted_distances = sort_by_milestone(pretrain_days, country_wise_distances, by_metric='euclidean')