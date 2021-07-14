# TODO: test this file

# General Imports
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import yaml
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import matplotlib
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def preprocess_data(df, metric_type, learner_type, col_mapper):
    # Only pretrain days records are required not the mean row
    df.drop(['mean'], axis=1, inplace=True)

    # Renaming the Algorithm Columns
    df.rename(columns={'Unnamed: 0': 'Algorithms'}, inplace=True)

    # Dropping first two rows: "EvaluationMeasurement" & "PretrainDays"
    df.drop([0, 1], axis=0, inplace=True)

    # Renaming columns based on mapper
    df['Algorithms'].replace(col_mapper, inplace=True)

    # Melting the dataframe based on 'Algorithms'
    df_melt = df.melt(id_vars=['Algorithms'])

    # Dropping unwanted varibale column(created bcoz of index)
    df_melt.drop('variable', axis=1, inplace=True)

    # Renaming the value column by metric type
    df_melt.rename(columns={'value': metric_type}, inplace=True)

    # Converting to float value bcoz by default the values are of type object
    df_melt[metric_type] = df_melt[metric_type].astype('float64')

    df_melt['Learner Type'] = learner_type  # Adding the learner type

    return df_melt


def order_by_median(df, reverse=False):
    grouped_df = df.groupby('Algorithms')
    algo_medians = {}
    for cur_group in grouped_df.groups.keys():
        df_cur_grp = grouped_df.get_group(cur_group)
        algo_medians[cur_group] = df_cur_grp['MAPE'].median()
    sorted_algo_medians = dict(sorted(algo_medians.items(), key=lambda kv: kv[1], reverse=reverse))
    return list(sorted_algo_medians.keys())


# If value is less than zero return float value otherwise an integer value
def format_values(y_val, pos):
    if y_val < 1:
        return format(float(y_val))
    else:
        return format(int(y_val))


def draw_save_boxplot(df, hue_order_learner, save_filename, prequential=False):
    if not prequential:
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#ca0020']
    else:
        colors = ['#ca0020', '#2ca02c', '#ff7f0e']
    # Setting custom color palette
    sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(10, 6), dpi=90)
    ordered_algo_list = order_by_median(df, reverse=False)

    ax = sns.boxplot(x="Algorithms", y=metric_type, hue='Learner Type', data=df, order=ordered_algo_list, dodge=False,
                     width=0.5, hue_order=hue_order_learner)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(yscale='log')
    ax.set_ylim(top=1000)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(format_values))  # lambda x, p: format(int(x), ',')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(metric_type, fontsize=18)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{box_plot_path}/{save_filename}.pdf')
    plt.show()


def read_preprocess_plot_graph(filenames, col_mapper, save_filename, metric_type='MAPE'):
    metric_type = metric_type
    frames = []
    for filename in filenames:
        if 'static' in filename:
            learner_type = 'Static'
        elif 'alternate' in filename:
            learner_type = 'Incremental(prequential)'
        else:
            learner_type = 'Incremental'

        df = pd.read_csv(filename)
        df_melt = preprocess_data(df, metric_type, learner_type, col_mapper)
        frames.append(df_melt)

    final_df = pd.concat(frames, ignore_index=True)

    # Updating LSTM learner type as Sequential
    final_df.loc[final_df['Algorithms'] == 'LSTM', 'Learner Type'] = 'Sequential'

    # Sorting final dataframe
    final_df = final_df.sort_values(by=['MAPE'])

    hue_order_learner = sorted(final_df['Learner Type'].unique())

    prequential_flag = 'Incremental(prequential)' in hue_order_learner

    draw_save_boxplot(final_df, hue_order_learner, save_filename, prequential=prequential_flag)


col_mapper = {'HT_Reg': 'Hoeffding Trees',
              'HAT_Reg': 'Hoeffding Adapt Tr',
              'ARF_Reg': 'Adaptive RF',
              'PA_Reg': 'Pass Agg Regr',
              'RandomForest': 'Random Forest',
              'GradientBoosting': 'Gradient Boosting',
              'DecisionTree': 'Decision Trees',
              'LinearSVR': 'Linear SVR',
              'BayesianRidge': 'Bayesian Ridge'
              }

metric_type = 'MAPE'

yaml_file_path = "../config.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

box_plot_path = parsed_yaml_file['paths']['box_plot_path']
exp1_path = parsed_yaml_file['paths']['exp1_path']
exp2_path = parsed_yaml_file['paths']['exp2_path']
# exp3_path = 'content/Result/exp3'

exp1_filenames = glob.glob(f'{exp1_path}/*{metric_type}*.csv')
exp2_filenames = glob.glob(f'{exp2_path}/*{metric_type}*.csv')
# exp3_filenames = glob.glob(f'{exp3_path}/*{metric_type}*.csv')

save_filename = 'fig1'
read_preprocess_plot_graph(exp1_filenames, col_mapper, save_filename, metric_type)

save_filename = 'fig2'
read_preprocess_plot_graph(exp2_filenames, col_mapper, save_filename, metric_type)

# exp3_filenames.append(exp2_filenames[1])
# save_filename = 'fig3'
# read_preprocess_plot_graph(exp3_filenames, col_mapper, save_filename, metric_type)

print('Done')
