from src.utils import *
from skmultiflow.data import DataStream
from src.skmultiflow.src.evaluate_prequential import EvaluatePrequential
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def scikit_multiflow(df, pretrain_days):  # Updated Now
    model, model_names = instantiate_regressors()

    len_countries = len(df['country'].unique())

    # Selecting only required countries
    df = df[df['country'].isin(df['country'].unique()[0:len_countries])]  # Added Now

    frames, running_time_frames = [], []

    united_dataframe = []  # Added Now

    # Setup the evaluator
    for day in pretrain_days:
        df_subset = create_subset(df, day)

        # Creating a stream from dataframe
        stream = DataStream(np.array(df_subset.iloc[:, 4:-1]),
                            y=np.array(df_subset.iloc[:, -1]))  # Selecting features x=[t-89:t-39] and y=[target].

        pretrain_size = day * len_countries
        max_samples = pretrain_size + 1  # One Extra Sample
        testing_samples_size = (day + 30) * len_countries

        evaluator = EvaluatePrequential(show_plot=False,
                                        pretrain_size=pretrain_size,
                                        metrics=['mean_square_error', 'mean_absolute_error',
                                                 'mean_absolute_percentage_error'],
                                        max_samples=max_samples)
        # Run evaluation
        evaluator.evaluate(stream=stream, model=model, model_names=model_names)

        # Added Now
        X = stream.X[pretrain_size: testing_samples_size]  # Updated Now
        y = stream.y[pretrain_size: testing_samples_size]  # Updated Now
        date_idx = list(df_subset.columns).index('date')  # Added Now
        target_dates = df_subset.iloc[pretrain_size: testing_samples_size, date_idx]  # Added Now

        prediction = evaluator.predict(X)

        # Since we add one extra sample, reset the evaluator
        evaluator = reset_evaluator(evaluator)
        evaluator = update_incremental_metrics(evaluator, y, prediction)

        country_idx = list(df_subset.columns).index('country')  # Added Now
        subset_countries_names = df_subset.iloc[pretrain_size:testing_samples_size, country_idx]  # Added Now
        united_dataframe.append(unit_incremental_df(subset_countries_names, evaluator, target_dates, day))  # Added now

        # Dictionary to store each iteration error scores
        mdl_evaluation_scores = {}

        # Adding Evaluation Measurements and pretraining days
        mdl_evaluation_scores['EvaluationMeasurement'] = ['RMSE', 'MAE', 'MAPE']  # ,'MSE']
        mdl_evaluation_scores['PretrainDays'] = [day] * len(mdl_evaluation_scores['EvaluationMeasurement'])
        mdl_evaluation_df = get_error_scores_per_model(evaluator, mdl_evaluation_scores)

        # Errors of each model on a specific pre-train days
        frames.append(mdl_evaluation_df)

        # Run time for each algorithm
        running_time_frames.append(get_running_time_per_model_incremental_learner(evaluator, day))

    # Final Run Time DataFrame
    running_time_df = pd.concat(running_time_frames, ignore_index=True)

    united_df = pd.concat(united_dataframe, ignore_index=True)

    # Final Evaluation Score Dataframe
    evaluation_scores_df = pd.concat(frames, ignore_index=True)

    return evaluation_scores_df, running_time_df, united_df


def scikit_learn(df, training_days):

    len_countries = len(df['country'].unique())

    # Selecting only required countries
    df = df[df['country'].isin(df['country'].unique()[0:len_countries])]  # Added Now

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

    layers = [50, 30, 20, 10]  # the final net will have n_layers +  2 + 1 = n*LSTMs + Dense + LSTM + output
    activations = ['tanh', 'tanh', 'relu']
    epochs = 500
    patience = 20 * num_selected_countries
    batch_size_lstm = 10 * num_selected_countries

    united_dataframe = []  # Added Now

    for day in training_days:
        df_subset = create_subset(df, day)

        train_end_day = day * len_countries
        test_end_day = (day + 30) * len_countries

        date_idx = list(df_subset.columns).index('date')  # Added Now
        target_dates = df_subset.iloc[train_end_day: test_end_day, date_idx]  # Added Now

        train = df_subset.iloc[:train_end_day, :]
        test = df_subset.iloc[train_end_day:test_end_day, :]  # Testing on set one month ahead only, hence day+30.
        cur_exec_time = [day]

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
        model_predictions['LSTM'], exec_time = train_test_lstm(lstm_model, X_train_lstm, y_train_batch, X_val_lstm, y_val_batch, X_test_lstm, patience, epochs,batch_size_lstm)
        cur_exec_time.append(exec_time)

        country_idx = list(df_subset.columns).index('country')  # Added Now
        subset_countries_names = df_subset.iloc[train_end_day: test_end_day, country_idx]  # Added Now
        united_dataframe.append(unit_static_df(subset_countries_names, target_dates, y_test, day, model_predictions))  # Added now

        mdl_evaluation_df = get_scores(y_test, model_predictions, day)
        total_execution_time.append(cur_exec_time)
        frames.append(mdl_evaluation_df)

    evaluation_score_df = pd.concat(frames, ignore_index=True)
    united_df = pd.concat(united_dataframe, ignore_index=True)  # Added Now
    running_time_df = get_running_time_per_model_static_learner(model_predictions, total_execution_time)
    return evaluation_score_df, running_time_df, united_df


def start_inc_learning():
    result_skmlflow, running_time_combined_incremental, united_df = scikit_multiflow(result,
                                                                                     pretrain_days)  # Updated Now

    save_united_df(united_df, exp2_inc_united_df_path)  # Added Now

    df_skmlflow = calc_save_err_metric_combined(error_metrics,
                                                result_skmlflow,
                                                max_of_pretrain_days,
                                                max_selected_countries,
                                                path=exp2_path,
                                                static_learner=False,
                                                alternate_batch=False,
                                                transpose=True)

    save_runtime(running_time_combined_incremental, path=exp2_runtime_path, static_learner=False)

    summary_table_incremental = get_summary_table(df_skmlflow,
                                                  running_time_combined_incremental,
                                                  error_metrics,
                                                  static_learner=False)

    save_summary_table(summary_table_incremental,
                       exp2_summary_path,
                       static_learner=False,
                       alternate_batch=False,
                       transpose=True)


def start_static_learning():
    result_sklearn, running_time_static, united_df = scikit_learn(result, pretrain_days)  # Updated Now

    save_united_df(united_df, exp2_static_united_df_path)  # Added Now

    df_sklearn = calc_save_err_metric_combined(error_metrics,
                                               result_sklearn,
                                               max_of_pretrain_days,
                                               max_selected_countries,
                                               path=exp2_path,
                                               static_learner=True,
                                               alternate_batch=False,
                                               transpose=True)
    # display_scores(df_sklearn)

    save_runtime(running_time_static, path=exp2_runtime_path, static_learner=True)

    summary_table_static = get_summary_table(df_sklearn,
                                             running_time_static,
                                             error_metrics,
                                             static_learner=True)

    save_summary_table(summary_table_static,
                       exp2_summary_path,
                       static_learner=True,
                       alternate_batch=False,
                       transpose=True)


yaml_file_path = "../config.yaml"
with open(yaml_file_path, 'r') as yaml_file:
    # yaml_file = open(yaml_file_path)
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Fetching Vars
exp2_inc_united_df_path = parsed_yaml_file['paths']['exp2_inc_united_df_path']
pretrain_days = parsed_yaml_file['pretrain_days']
countries = parsed_yaml_file['valid_countries']
csv_processed_path = parsed_yaml_file['paths']['csv_processed_path']
exp2_runtime_path = parsed_yaml_file['paths']['exp2_runtime_path']
exp2_path = parsed_yaml_file['paths']['exp2_path']
exp2_summary_path = parsed_yaml_file['paths']['exp2_summary_path']
exp2_static_united_df_path = parsed_yaml_file['paths']['exp2_static_united_df_path']
data_path = parsed_yaml_file['paths']['data_path']
valid_countries = parsed_yaml_file['valid_countries']
num_selected_countries = len(valid_countries)

# Get Dataset
df = pd.read_csv(data_path)
df = preprocess_dataset(df)
df_grouped = df.groupby('country')

# Calculating targets and lags for the above countries
result = get_dataset_with_target(valid_countries, df_grouped)

# Getting max of each subset in pretrain size
max_of_pretrain_days = calc_max_of_pretrain_days(pretrain_days, result)

# Mean of top selected countries
max_selected_countries = result['cases'].max()

# INCREMENTAL LEARNING
start_inc_learning()

# STATIC LEARNER
start_static_learning()
