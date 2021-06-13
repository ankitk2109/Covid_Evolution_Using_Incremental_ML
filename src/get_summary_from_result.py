import glob
import pandas as pd

decimal = 3


# Return a combined dataframe for a each error statistics(MAE,RMSE,MAPE etc) along with the newly added mean row.
def get_metric_with_mean(result: pd.DataFrame, error_metric: str) -> pd.DataFrame:
    df_grouped = result.groupby('EvaluationMeasurement')
    df = df_grouped.get_group(error_metric).reset_index(drop=True)
    df = df.append(df.describe().loc['mean'])
    return df


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


def calc_save_err_metric_countrywise(countries, error_metrics, results, path, static_learner, transpose):
    countrywise_error_scores = {}
    for i in range(len(countries)):
        country_error_score = []
        for error_metric in error_metrics:
            df_error_metric = get_metric_with_mean(results[i], error_metric=error_metric)
            country_error_score.append(df_error_metric)

            # Transposing the metrics while saving
            save_metrics(df_error_metric, path=path, country=countries[i], static_learner=static_learner,
                         transpose=transpose)

        countrywise_error_scores[countries[i]] = pd.concat(country_error_score, ignore_index=True)

    return countrywise_error_scores


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


file_paths = glob.glob('content/Result/exp1/*static.csv')
error_metrics = ['MAE', 'MAPE', 'RMSE']
exp1_summary_path = 'content/Result/exp1/summary'
exp1_path = 'content/Result/exp1'


countries = []
for file in glob.glob('content/csv_files/processed/*'):
    countries.append(file.split('\\')[-1].split('.')[0])

results_static = []
results_runtime_static = []

def get_results_static(exp1_path,countries):
    for country in countries:
        cur_country_scores = []
        for cur_country_path in glob.glob(f'{exp1_path}/{country}*static.csv'):
            df = pd.read_csv(cur_country_path)
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df[1:9]
            cur_country_scores.append(df)
        df_cur_result = pd.concat(cur_country_scores, ignore_index=True)
        numeric_cols = df_cur_result.columns[1:]
        df_cur_result[numeric_cols] = df_cur_result[numeric_cols].astype('float')
        results_static.append(df_cur_result)
    return results_static


def get_results_runtime_static(exp1_path, countries):
    for country in countries:
        df = pd.read_csv(glob.glob(f'{exp1_path}/runtime/{country}*static.csv')[0])
        df.set_index('Unnamed: 0', inplace=True)
        results_runtime_static.append(df)
    return results_runtime_static


results_static = get_results_static(exp1_path,countries)
results_runtime_static = get_results_runtime_static(exp1_path, countries)

#TODO: Now find summary

# countrywise_error_scores_static = calc_save_err_metric_countrywise(countries, error_metrics, results_static,
#                                                                    path=exp1_path, static_learner=True, transpose=True)
#
# summary_table_countrywise_static = get_summary_table_countrywise(countrywise_error_scores_static, ['MAPE'],
#                                                                  static_learner=True)
#
# # Saving the transposed matrix
# save_summary_table(summary_table_countrywise_static, exp1_summary_path, country=True, static_learner=True,
#                    alternate_batch=False, transpose=True)
#
# sum_static_countrywise_mean = get_sum_table_combined_mean(countrywise_error_scores_static, results_runtime_static,
#                                                           static_learner=True)
#
# save_combined_summary_table(sum_static_countrywise_mean, exp1_summary_path, static_learner=True, transpose=True)
#
# print(sum_static_countrywise_mean)

print('done')
