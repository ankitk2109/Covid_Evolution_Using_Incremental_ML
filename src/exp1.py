from src.helper import *
from skmultiflow.data import DataStream
from src.src.evaluate_prequential import EvaluatePrequential


def reset_evaluator(evaluator):  # Added Now
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


def scikit_multiflow(df, pretrain_days, country):  # Added Country in parameter
    # Creating a stream from dataframe
    stream = DataStream(np.array(df.iloc[:, 4:-1]),
                        y=np.array(df.iloc[:, -1]))  # Selecting features x=[t-89:t-39] and y=[target].

    model, model_names = instantiate_regressors()

    frames, running_time_frames = [], []

    united_dataframe = []  # Added Now

    # Setup the evaluator
    for day in pretrain_days:
        pretrain_days = day
        # max_samples = pretrain_days + 30  # Training and then testing on set one month ahead only
        max_samples = pretrain_days + 1
        testing_samples_size = 30

        evaluator = EvaluatePrequential(show_plot=False,
                                        pretrain_size=pretrain_days,
                                        metrics=['mean_square_error', 'mean_absolute_error',
                                                 'mean_absolute_percentage_error'],
                                        max_samples=max_samples)

        # Run evaluation
        evaluator.evaluate(stream=stream, model=model, model_names=model_names)

        X = stream.X[pretrain_days: pretrain_days + testing_samples_size]
        y = stream.y[pretrain_days: pretrain_days + testing_samples_size]
        target_dates = df.iloc[pretrain_days: pretrain_days + testing_samples_size, 0]  # Added Now

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


# YAML FILE
yaml_file_path = "vars.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

exp1_inc_united_df_path = parsed_yaml_file['paths']['exp1_inc_united_df_path']
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
exp1_runtime_path = parsed_yaml_file['paths']['exp1_runtime_path']

# Training all countries
results_incremental = []
results_runtime_incremental = []
max_of_pretrain_per_country = []
max_cases_per_country = []

for country in countries:
    # Read each country
    df_country = pd.read_csv(f'{csv_processed_path}/{country}.csv')

    # Get evaluation scores and running time for country
    evaluation_scores_df, running_time_df, united_dataframe = scikit_multiflow(df_country, pretrain_days, country)

    save_united_df(united_dataframe, exp1_inc_united_df_path, country=country)

    # Appending evaluation scores and runtime for each country
    results_incremental.append(evaluation_scores_df)

    results_runtime_incremental.append(running_time_df)

    # Get max of each pretrain subset and for each country dataset
    max_of_pretrain_per_country.append(calc_max_of_pretrain_days(pretrain_days, df_country))
    max_cases_per_country.append(df_country['cases'].max())

# Save the running time for each country
for i in range(len(countries)):
    save_runtime(results_runtime_incremental[i], path=exp1_runtime_path, country=countries[i], static_learner=False)

# Display countrywise running time complexity
display_runtime_per_country(results_runtime_incremental, countries)
