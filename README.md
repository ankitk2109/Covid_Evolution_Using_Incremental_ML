# Applying Dynamic Time Warping for boosting the performance of Static and Incremental Machine Learning in Forecasting COVID-19 Cases
Modelling the COVID-19 virus evolution using incremental machine learning methods



## Experiments

1. **Experiment 1: Single-Country training**

This experiment trains the supervised ML models with a single-country and predicts the cases for the same single country with which the model is trained

* Mean performance of the incremental and static methods

  | Metric                        | RMSE    | MAE    | Time (Sec) |
  | ----------------------------- | ------- | ------ | ---------- |
  | Gradient Boosting             | 1,821   | 1,635  | 0.147      |
  | Decision Tree                 | 1,976   | 1,713  | 0.005      |
  | Random Forest                 | 2,187   | 2,031  | 0.197      |
  | LSTM                          | 3,311   | 3,053  | 20.525     |
  | Bayesian Ridge                | 7,165   | 5,934  | 0.011      |
  | Hoeffding Adaptive Trees*     | 15,653  | 11,502 | 0.146      |
  | Hoeffding Trees*              | 19,412  | 13,548 | 0.115      |
  | Adaptive Random Forest*       | 23,923  | 17,336 | 2.909      |
  | Linear SVR                    | 37,518  | 31,370 | 0.02       |
  | Passive Aggressive Regressor* | 111,570 | 91,809 | 0.002      |

 

2. **Experiment 2: Multi-Country training**

This second experiment predicts over the same 50 countries at the same eight points as Experiment I, but this time we
train the model with 50 countries rather than training it with a single country as in Experiment I

* Mean performance of the incremental and static methods

| Metric                        | RMSE    | MAE    | Time (Sec) |
| ----------------------------- | ------- | ------ | ---------- |
| Gradient Boosting             | 7,500   | 3,052  | 6.86       |
| Bayesian Ridge                | 7,671   | 2,943  | 0.04       |
| Random Forest                 | 7,739   | 3,197  | 3.11       |
| LSTM                          | 8,272   | 3,089  | 138.57     |
| Decision Tree                 | 8,764   | 3,663  | 0.46       |
| Linear SVR                    | 12,824  | 4,577  | 1.91       |
| Hoeffding Adaptive Trees*     | 13,946  | 4,803  | 21.23      |
| Adaptive Random Forest*       | 14,151  | 4,599  | 163.51     |
| Hoeffding Trees*              | 17,381  | 5,496  | 7.26       |
| Passive Aggressive Regressor* | 130,604 | 40,041 | 0.005      |

3. **Experiment 3: Multi-Countries Training by Similarity**

   We use time similarity measures (ED and DTW) to calculate the nine more similar countries to the predicted one for the training set of this predicted country at each milestone. And then, we create a dataset to train the models with the data from those nine countries plus the one that is predicted. Thus, training occurs over a total of ten countries where COVID-19 impacts similarly at each milestone. Predictions and evaluation of the models occur over all the milestones and countries as in Experiments I and II.

   | N    | Algorithm/RMSE                | SC      | MC      | ED     | DTW    | Mean   |
   | ---- | ----------------------------- | ------- | ------- | ------ | ------ | ------ |
   | 1    | Gradient Boosting             | 1,822   | 7,500   | 1,908  | 1,896  | 3,282  |
   | 2    | Random Forest                 | 2,188   | 7,739   | 1,811  | 1,800  | 3,385  |
   | 3    | Decision Tree                 | 1,977   | 8,764   | 2,171  | 2,155  | 3,767  |
   | 4    | LSTM                          | 3,311   | 8,272   | 2,277  | 2,253  | 4,028  |
   | 5    | Bayesian Ridge                | 7,166   | 7,671   | 1,839  | 1,845  | 4,630  |
   | 6    | Hoeffding Adaptive Trees*     | 15,654  | 13,946  | 2,594  | 2,827  | 8,755  |
   | 7    | Hoeffding Trees*              | 19,412  | 17,381  | 2,424  | 2,894  | 10,528 |
   | 8    | Adaptive Random Forest*       | 23,923  | 14,151  | 3,120  | 3,102  | 11,074 |
   | 9    | Linear SVR                    | 37,518  | 12,824  | 4,448  | 4,405  | 14,799 |
   | 10   | Passive Aggressive Regressor* | 111,570 | 130,604 | 35,319 | 33,204 | 77,674 |

   

## Steps to run

1. Before running the code make sure the following empty structure for **running_results** directory is created in **results**:

   ```bash
   `results`
   `├───final_result_files`
   `├───final_result_plots`
   `│   ├───barplot`
   `│   └───boxplots`
   `└───running_results`
       `└───content`
           `├───csv_files`
           `│   ├───processed`
           `│   └───processed_null`
           `├───Plots`
           `│   ├───barplot`
           `│   └───boxplots`
           `└───Result`
               `├───exp1`
               `│   ├───runtime`
               `│   ├───summary`
               `│   └───united_dataframe`
               `│       ├───incremental`
               `│       └───static`
               `├───exp2`
               `│   ├───runtime`
               `│   ├───summary`
               `│   └───united_dataframe`
               `│       ├───incremental`
               `│       └───static`
               `└───exp3`
                   `├───runtime`
                   `├───summary`
                   `└───united_dataframe`
                       `└───incremental`
               		`└───static`
   ```

2. Once the above folder structure is ready, make sure all the requirements are installed as mentioned in [requirement.txt](requirement.txt).

3. Run the  [preprocess.py](src/preprocess.py) file to create the data frames and files required for both experiments.

4. Make sure that  [skmultiflow](src/skmultiflow/src) folder is present in source code as we have implemented few functionalities that were not a part of skmultiflow package originally.

5. Run the  [exp1.py](src/exp1.py),  [exp2.py](src/exp2.py) and [exp3.py](src/exp3.py) files to generate results.
