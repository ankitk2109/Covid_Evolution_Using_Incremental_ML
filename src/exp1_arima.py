import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima, arima
from pmdarima import ARIMA

from src.utils import *
from src.utils import get_scores


def get_arima_dataset(filepath, all_country):
    """Get dataset from day 1 with actual case numbers"""
    df = []
    for country in all_country:
        cur_df = pd.read_csv(f"{filepath}/{country}.csv")
        cur_df = cur_df.loc[:, ['date', 'day_no', 'country', 'cases']]  # drop unwanted columns
        cur_df['Target'] = cur_df['cases'].rolling(DAYS_TO_AVG).mean()
        df.append(cur_df)
    df = pd.concat(df)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    return df


def plot_cases(train_set, test_set):
    x_train, x_test = np.array(range(train_set.shape[0])), \
                      np.array(range(train_set.shape[0], test_set.shape[0] + train_set.shape[0]))
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x_train, train_set)
    ax.plot(x_test, test_set)
    plt.show()


def get_arima_order(train_set):
    model = auto_arima(train_set, start_p=1, start_q=1,
                       test='adf',
                       max_p=10, max_q=10,
                       m=1,
                       d=1,
                       seasonal=False,
                       start_P=0,
                       D=None,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    return model.order


def plot_prediction(pred, cf, test_set, train_set, days):
    pred_df = pd.DataFrame({'date': list(test_set.index), 'pred': pred}).set_index('date')
    prediction_series = pred_df['pred']
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(train_set, label='train')
    ax.plot(test_set, label='test')
    ax.plot(prediction_series, label='prediction')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.fill_between(prediction_series.index,
                    cf[0],
                    cf[1], color='grey', alpha=.3)
    ax.legend()
    plt.title(f'Pretrain Days: {days}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"../results/running_results/content/Plots/lineplot/{days}.png")
    # plt.show()


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def get_summary(df_result_dict, error_metrics, static_learner=True):  # df_runtime_result,
    """
    This method calculates the summary dataframe for exp2 for all metrics
    """
    summary_metric = []
    measure_col_name = f'Country({str(error_metrics[0])})'
    eval_measure_col = 'EvaluationMeasurement'
    start_row = 'mean'
    # if static_learner:
    #     start_col = 'RandomForest'
    # else:
    #     start_col = 'HT_Reg'
    start_col = 'ARIMA'
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


def get_metric_with_mean(result: pd.DataFrame, error_metric: str) -> pd.DataFrame:
    """
    This method calculates mean of the EvaluationMeasurement column for specified error metric (MAPE/RMSE/MAE)
    """
    df_grouped = result.groupby('EvaluationMeasurement')
    df = df_grouped.get_group(error_metric).reset_index(drop=True)
    df = df.append(df.describe().loc['mean'])
    return df


def get_predictions(df_arima_country):
    # save scores for all countries
    all_milestone_scores = []

    for milestone in PRETRAIN_DAYS:
        # get train set
        train_start = 0
        train_end = milestone + DEFAULT_SIZE
        train = df_arima_country.iloc[train_start:train_end]  # separate training size

        # get test set
        test_start = train_end + SKIP_DAYS
        test_end = test_start + TEST_SIZE
        test = df_arima_country.iloc[test_start: test_end]

        # # plot the train test cases on graph
        # plot_cases(train, test)

        # get order for arima
        order = get_arima_order(train['cases'])

        milestone_predictions = []
        for window_start in range(TEST_SIZE):
            window_end = window_start + train_end
            x_train = df_arima_country.iloc[window_start: window_end]['cases']

            model = ARIMA(order=order)
            model_fit = model.fit(x_train)

            prediction, _ = model_fit.predict(n_periods=FORECAST_DAYS, return_conf_int=True)
            average = prediction[-DAYS_TO_AVG:].mean()
            milestone_predictions.append(average)

            # plot the predictions on graph
            # plot_prediction(prediction, df_cf, test, train, p_days)

        # test and prediction length should be same
        if len(test) < len(milestone_predictions):
            milestone_predictions = milestone_predictions[0:len(test)]

        # prediction dictionary
        model_predictions = {'ARIMA': milestone_predictions}
        cur_milestone_scores = get_scores(test['Target'].values.tolist(), model_predictions, milestone)
        all_milestone_scores.append(cur_milestone_scores)

    # convert frames to dataframe
    # country_metric[valid_country] = pd.concat(all_pday_scores)
    df_score = pd.concat(all_milestone_scores)
    return df_score


def train_predict_save(df, save_path):
    all_countries_with_metrics = {}
    for valid_country in COUNTRIES:
        # check if file already exists
        csv_file_name = f"{valid_country}_['RMSE' 'MAE' 'MAPE']_static.csv"
        csv_file_path = os.path.join(save_path, csv_file_name)

        tex_file = f"{valid_country}_['RMSE' 'MAE' 'MAPE']_static.tex"
        tex_file_path = os.path.join(save_path, tex_file)

        if not (os.path.exists(csv_file_path) and os.path.exists(tex_file_path)):
            # filter valid countries
            df_arima_country = df[df['country'] == valid_country]
            # drop columns
            df_arima_country = df_arima_country.drop(['country', 'day_no'], axis=1)

            df_score = get_predictions(df_arima_country)

            # add country name
            all_countries_with_metrics[valid_country] = df_score

            # save the metric for the current country
            save_metrics(df_score, save_path, valid_country)

        else:
            print(f"csv file for country {valid_country.upper()} already exists!")

            # read scores
            df_score = pd.read_csv(csv_file_path, usecols=['EvaluationMeasurement', 'PretrainDays', 'ARIMA'])

            # add country name
            all_countries_with_metrics[valid_country] = df_score

    return all_countries_with_metrics


# YAML FILE
parsed_yaml_file = get_configs_yaml()
PRETRAIN_DAYS = parsed_yaml_file['pretrain_days']
COUNTRIES = parsed_yaml_file['valid_countries']
SKIP_DAYS = 30
DEFAULT_SIZE = 50
DAYS_TO_AVG = 10
FORECAST_DAYS = 40
TEST_SIZE = 30

exp1_inc_united_df_path = parsed_yaml_file['paths']['exp1_inc_united_df_path']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
csv_processed_with_null_path = parsed_yaml_file['paths']['csv_processed_with_null_path']
exp1_runtime_path = parsed_yaml_file['paths']['exp1_runtime_path']
exp1_path = parsed_yaml_file['paths']['exp1_path']
exp1_summary_path = parsed_yaml_file['paths']['exp1_summary_path']
exp1_static_united_df_path = parsed_yaml_file['paths']['exp1_static_united_df_path']
exp4_path = parsed_yaml_file['paths']['exp4_path']

df_arima = get_arima_dataset(csv_processed_with_null_path, COUNTRIES)
result_dict = train_predict_save(df_arima, exp4_path)
summary = get_summary(result_dict, ['RMSE', 'MAPE', 'MAE'])
grouped_summary = summary.groupby('EvaluationMeasurement')
summary_dict = {'Metric': 'ARIMA',
                'RMSE': round(grouped_summary.get_group('RMSE')['ARIMA'].mean(), 2),
                'MAE': round(grouped_summary.get_group('MAE')['ARIMA'].mean(), 2),
                'MAPE': round(grouped_summary.get_group('MAPE')['ARIMA'].mean(), 2)
                }
final_summary_df = pd.DataFrame.from_dict(summary_dict, orient='index').transpose()
