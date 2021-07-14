# Imports

import os
import yaml
import glob
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Create lags
def create_features_with_lags(df_2, lag_days):
    for i in range(lag_days, 0,
                   -1):  # Loop in reverse order for creating ordered lags eg: cases_t-10, cases_t-9... cases_t-1. t=current cases
        df_2[f'cases_t-{i}'] = df_2['cases'].shift(i, axis=0)
    return df_2


# Preprocess data
def preprocess_data(df_grouped, total_countries, lag_days):
    for country in total_countries:
        # process only if file does not exist already
        filepath = os.path.join(csv_processed_path, f'{country}.csv')
        filepath_null = os.path.join(csv_processed_with_null_path, f'{country}.csv')
        if not (os.path.exists(filepath) or os.path.exists(filepath_null)):
            df_1 = df_grouped.get_group(country)

            # Selecting required features
            df_1 = df_1[['dateRep', 'cases', 'countriesAndTerritories']]

            # Rename features
            df_1.rename(columns={'countriesAndTerritories': 'country', 'dateRep': 'date'}, inplace=True)

            # Convert to date, sort and set index
            df_1['date'] = pd.to_datetime(df_1['date'], format='%d/%m/%Y')
            df_1.sort_values('date', inplace=True)
            df_1.set_index('date', inplace=True)

            # Adding feature
            df_1['day_no'] = pd.Series([i for i in range(1, len(df_1) + 1)], index=df_1.index)

            # Reordering features
            df_1 = df_1[['day_no', 'country', 'cases']]

            # Adding features through lags
            df_1 = create_features_with_lags(df_1, lag_days)

            # Creating target with last 10 days cases
            df_1['target'] = df_1.iloc[:, [2] + [i * -1 for i in range(1, 10)]].mean(axis=1)

            # Dropping mid columns
            drop_columns = list(
                df_1.loc[:, 'cases_t-39':'cases_t-1'].columns)  # list(df_1.loc[:,'cases_t-38':'cases_t-1'].columns)
            df_1.drop(drop_columns, axis=1, inplace=True)

            # Country name
            filename = df_1['country'].unique()[0]

            # Saving file
            df_1.to_csv(f'{csv_processed_with_null_path}/{filename}.csv')

            # Dropping null records
            df_1.dropna(how='any', axis=0, inplace=True)

            # Valid countries that have records more than max of pretrain
            if len(df_1) > max(pretrain_days):
                valid_countries.append(country)
                df_1.to_csv(f'{csv_processed_path}/{filename}.csv')
        else:
            print(f"File already exist! Skipping country: {country} ")
    print('Processing Done!')


# Replaces underscore from country names
def format_names(list_countries):
    updated_country_list = []
    for country_name in list_countries:
        updated_country_list.append(country_name.replace("_", " "))
    return updated_country_list


# Top selected countries
def top_selected_countries(num_of_countries):
    # A dictionary of all countries
    dict_countries = Counter(valid_countries)
    for country in valid_countries:
        dict_countries[country] = df_grouped.get_group(country)['cases'].sum()

    # Select top_countries and order(Ascending/Decending)
    top_countries = sorted(dict_countries.items(), key=lambda dict_countries: dict_countries[1], reverse=True)[
                    0:num_of_countries]

    # Creating dataframe of top selected countries
    df_top_countries = pd.DataFrame.from_dict(dict(top_countries), orient='index', columns=['Total Cases'])
    return df_top_countries


# get valid countries
def get_valid_countries(val_countries, path):
    if len(val_countries) != 0:
        return val_countries
    else:
        return [x.split('\\')[-1].split('.')[0] for x in glob.glob(f"{path}/*")]


# plot top countries
def plot_top_countries(df_top_countries, num_country_plot):
    df_plot_countries = df_top_countries.iloc[0:num_country_plot]
    # Plotting graph
    sns.set_theme(style='white')
    plt.figure(figsize=(12, 16))
    top_countries_list = format_names(df_plot_countries.index)
    plt.barh(top_countries_list[::-1], df_plot_countries['Total Cases'].values[
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
    plt.show()


# Average cases of top countries
def avg_cases_top_countries(df_top_countries, total_countries):
    dict_countries_avg = Counter(total_countries)

    for country in df_top_countries.index.tolist():  # loop through the list of top countries
        dict_countries_avg[country] = df_grouped.get_group(country)['cases'].mean()

    # Average cases for all countries
    df_avg_cases_countries = pd.DataFrame.from_dict(dict_countries_avg, orient='index', columns=['Average'])

    # List of top selected countries
    top_countries = list(df_top_countries.index)

    # Average of selected top countries
    avg_df = df_avg_cases_countries[df_avg_cases_countries.index.isin(top_countries)]

    return avg_df


# Load all global variables from YAML file
# yaml_file_path = "vars.yaml"
yaml_file_path = "../config.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Get variables
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
csv_processed_with_null_path = parsed_yaml_file['paths']['csv_processed_with_null_path']
path = parsed_yaml_file['paths']['data_path']
error_metrics = parsed_yaml_file['error_metrics']
pretrain_days = parsed_yaml_file['pretrain_days']
number_of_countries = parsed_yaml_file['number_of_countries']
num_country_plot = parsed_yaml_file['num_country_plot']
decimal = parsed_yaml_file['decimal']  # Specify the scale of decimal places
bar_plot_path = parsed_yaml_file['paths']['bar_plot_path']
lag_days = int(parsed_yaml_file['lag_days'])

# read data, group countries and count countries
df = pd.read_csv(path)
total_countries = df['countriesAndTerritories'].unique()
df_grouped = df.groupby('countriesAndTerritories')
valid_countries = []

# Preprocess data
preprocess_data(df_grouped, total_countries, lag_days)

# get valid countries
valid_countries = get_valid_countries(valid_countries, csv_processed_path)

# Get top selected countries
df_top_countries = top_selected_countries(number_of_countries)

# plot top countries
plot_top_countries(df_top_countries, num_country_plot)

# Average cases of top countries
avg_df = avg_cases_top_countries(df_top_countries, total_countries)  # Not saving these details anywhere

# Update yaml file parser
parsed_yaml_file['valid_countries'] = valid_countries
parsed_yaml_file['total_countries'] = total_countries.tolist()

# Dump data to yaml file
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(parsed_yaml_file, yaml_file, default_flow_style=False)

print('Process Complete!')
