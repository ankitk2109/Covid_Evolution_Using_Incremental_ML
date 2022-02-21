# Imports

import os
import yaml
import glob
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


def format_result(country_dict, milestones):
    milestone_cols = milestones.copy()
    milestone_cols.append('country')
    for m in country_dict:
        country_dict[m] = pd.DataFrame(country_dict[m], columns=milestone_cols)
    return country_dict


def calc_distance(c1, c2, milestones, dist_metrics, country_dict):
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


def pairwise_country_distance(countries, dist_metrics, num_of_country=-1):
    num_of_country = num_of_country if num_of_country > 0 else len(countries)
    dist_country = {}
    for i in range(0, num_of_country):  # len(countries)
        dist_country[countries[i]] = {}
        dist_country[countries[i]] = {metric: [] for metric in dist_metrics}  # creating empty list for all metrics
        for j in range(0, num_of_country):  # len(countries)
            if countries[i] != countries[j]:
                dist_country[countries[i]] = calc_distance(countries[i],
                                                           countries[j],
                                                           pretrain_days,
                                                           dist_metrics,
                                                           dist_country[countries[i]])
        dist_country[countries[i]] = format_result(dist_country[countries[i]], pretrain_days)
    return dist_country


country_wise_distances = pairwise_country_distance(countries, distance_metrics, num_of_country=5)
# pd.DataFrame(dist_country['Australia']['euclidean'], columns=['30','60', '90', '120', '150', '180', '210', '240', 'country'])
print(country_wise_distances)
