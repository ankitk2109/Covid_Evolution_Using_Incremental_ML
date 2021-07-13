from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from skmultiflow.data import DataStream

from src.utils import *
from src.skmultiflow.src.evaluate_prequential import EvaluatePrequential


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


def inc_evaluate_save_results():
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


def inc_save_runtimes():
    # Save the running time for each country
    for i in range(len(countries)):
        save_runtime(results_runtime_incremental[i], path=exp1_runtime_path, country=countries[i], static_learner=False)


def start_inc_learning():
    # Note: all arguments are global vars
    inc_evaluate_save_results()

    inc_save_runtimes()

    # Display countrywise running time complexity
    display_runtime_per_country(results_runtime_incremental, countries)

    # Country-wise Error Metric
    countrywise_error_score_incremental = calc_save_err_metric_countrywise(countries, error_metrics,
                                                                           results_incremental,
                                                                           max_of_pretrain_per_country,
                                                                           max_cases_per_country, path=exp1_path,
                                                                           static_learner=False, transpose=True)

    # Get summary table for each country for specified metric
    summary_table_countrywise_incremental = get_summary_table_countrywise(countrywise_error_score_incremental,
                                                                          ['MAPE'],
                                                                          static_learner=False)

    # Saving the summary table
    save_summary_table(summary_table_countrywise_incremental,
                       exp1_summary_path,
                       country=True,
                       static_learner=False,
                       alternate_batch=False,
                       transpose=True)

    sum_inc_countrywise_mean = get_sum_table_combined_mean(countrywise_error_score_incremental,
                                                           results_runtime_incremental)

    save_combined_summary_table(sum_inc_countrywise_mean, exp1_summary_path, static_learner=False, transpose=True)


def scikit_learn(df, training_days, country):  # Added country now
    frames = []
    model_predictions = {
        'RandomForest': [],
        'GradientBoosting': [],
        'LinearSVR': [],
        'DecisionTree': [],
        'BayesianRidge': [],
        'LSTM': []
    }
    total_execution_time = []

    # params (others like epoch and batch size are also hardcoded in train_test_lstm())
    layers = [50, 30, 20, 10]  # the final net will have n_layers +  2 + 1 = n*LSTMs + Dense + LSTM + output
    activations = ['tanh', 'tanh', 'relu']
    epochs = 200  # Previously: 500
    patience = 20
    batch_size_lstm = 10
    united_dataframe = []  # Added Now

    for day in training_days:
        print(f"~~~~~~~~~~~~~~~~~~~~*Pretraining Day: {day}~~~~~~~~~~~~~~~~~~~~")
        testing_samples_size = 30  # Added Now
        cur_exec_time = [day]  # Keeping running time for each pre-train set
        target_dates = df.iloc[day: day + testing_samples_size, 0]  # Added Now
        train = df.iloc[:day, :]
        test = df.iloc[day:day + testing_samples_size, :]  # Testing on set one month ahead only, hence day+30.

        # training and test sets for all models except LSTM
        X_train, y_train = train.iloc[:, 4:-1], train.iloc[:, -1]
        X_test, y_test = test.iloc[:, 4:-1], test.iloc[:, -1]

        # Seperating validation set from train set
        train_df, val_df = get_validation_set(train, batch_size=10)

        # Splitting test and validation into dependent and independent sets
        X_train_batch, y_train_batch = train_df.iloc[:, 4:-1], train_df.iloc[:, -1]  # Consist only odd batches
        X_val_batch, y_val_batch = val_df.iloc[:, 4:-1], val_df.iloc[:, -1]

        # Normalizing dataset
        X_train_lstm_norm, X_test_lstm_norm, X_val_lstm_norm = normalize_dataset(X_train_batch, X_test, X_val_batch)

        # Reshaping the dataframes
        X_train_lstm, X_val_lstm, X_test_lstm = reshape_dataframe(X_train_lstm_norm, X_val_lstm_norm, X_test_lstm_norm)

        rf_reg = RandomForestRegressor(max_depth=2, random_state=0)
        model_predictions['RandomForest'], exec_time = train_test_model(rf_reg, X_train, y_train, X_test)
        cur_exec_time.append(exec_time)

        gb_reg = GradientBoostingRegressor(random_state=0)
        model_predictions['GradientBoosting'], exec_time = train_test_model(gb_reg, X_train, y_train, X_test)
        cur_exec_time.append(exec_time)

        lsv_reg = LinearSVR(random_state=0, tol=1e-5)
        model_predictions['LinearSVR'], exec_time = train_test_model(lsv_reg, X_train, y_train, X_test)
        cur_exec_time.append(exec_time)

        dt_reg = DecisionTreeRegressor(random_state=0)
        model_predictions['DecisionTree'], exec_time = train_test_model(dt_reg, X_train, y_train, X_test)
        cur_exec_time.append(exec_time)

        br_reg = BayesianRidge()
        model_predictions['BayesianRidge'], exec_time = train_test_model(br_reg, X_train, y_train, X_test)
        cur_exec_time.append(exec_time)

        lstm_model = define_lstm_model(X_train_lstm, layers, activations, patience)
        model_predictions['LSTM'], exec_time = train_test_lstm(lstm_model, X_train_lstm, y_train_batch, X_val_lstm,
                                                               y_val_batch, X_test_lstm, patience, epochs,
                                                               batch_size_lstm)
        cur_exec_time.append(exec_time)

        united_dataframe.append(unit_static_df(country, target_dates, y_test, day, model_predictions))  # Added now

        mdl_evaluation_df = get_scores(y_test, model_predictions, day)
        total_execution_time.append(cur_exec_time)
        frames.append(mdl_evaluation_df)

    evaluation_score_df = pd.concat(frames, ignore_index=True)
    united_dataframe = pd.concat(united_dataframe, ignore_index=True)  # Added Now
    running_time_df = get_running_time_per_model_static_learner(model_predictions, total_execution_time)
    return evaluation_score_df, running_time_df, united_dataframe


def static_evaluate_save_results():
    for country in countries:
        # Read country wise csv file
        df_country = pd.read_csv(f'{csv_processed_path}/{country}.csv')

        print(f"*******************Processing {country}************************")
        # Evaluation scores and running time of each algorithm over different pre-training days
        evaluation_scores_df, running_time_df, united_dataframe = scikit_learn(df_country, pretrain_days,
                                                                               country)  # Returning united_dataframe also

        save_united_df(united_dataframe, exp1_static_united_df_path, country=country)

        # Append result of each pretrain size in results
        results_static.append(evaluation_scores_df)

        # Appending every country runtime
        results_runtime_static.append(running_time_df)

        # Saving runtime for each country
        save_runtime(running_time_df, path=exp1_runtime_path, country=country, static_learner=True)

        # Calculating max cases per country based on pre-train size
        max_of_pretrain_per_country.append(calc_max_of_pretrain_days(pretrain_days, df_country))

        # Maximum case of each country
        max_cases_per_country.append(df_country['cases'].max())


def start_static_learning():
    # Note: all arguments are global vars
    static_evaluate_save_results()

    countrywise_error_scores_static = calc_save_err_metric_countrywise(countries, error_metrics,
                                                                       results_static, max_of_pretrain_per_country,
                                                                       max_cases_per_country, path=exp1_path,
                                                                       static_learner=True, transpose=True)

    summary_table_countrywise_static = get_summary_table_countrywise(countrywise_error_scores_static,
                                                                     ['MAPE'],
                                                                     static_learner=True)

    # Saving the transposed matrix
    save_summary_table(summary_table_countrywise_static, exp1_summary_path,
                       country=True, static_learner=True,
                       alternate_batch=False, transpose=True)

    sum_static_countrywise_mean = get_sum_table_combined_mean(countrywise_error_scores_static,
                                                              results_runtime_static,
                                                              static_learner=True)

    save_combined_summary_table(sum_static_countrywise_mean, exp1_summary_path,
                                static_learner=True, transpose=True)


# YAML FILE
# yaml_file_path = "vars.yaml"
yaml_file_path = "../config.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Fetching Vars
exp1_inc_united_df_path = parsed_yaml_file['paths']['exp1_inc_united_df_path']
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
exp1_runtime_path = parsed_yaml_file['paths']['exp1_runtime_path']
exp1_path = parsed_yaml_file['paths']['exp1_path']
exp1_summary_path = parsed_yaml_file['paths']['exp1_summary_path']
exp1_static_united_df_path = parsed_yaml_file['paths']['exp1_static_united_df_path']

# INCREMENTAL LEARNER
results_incremental = []
results_runtime_incremental = []
max_of_pretrain_per_country = []
max_cases_per_country = []
start_inc_learning()

# STATIC LEARNER
results_static = []
results_runtime_static = []
max_cases_per_country, max_of_pretrain_per_country = reset(max_cases_per_country,max_of_pretrain_per_country)
start_static_learning()

# TODO: Significance test for exp1
