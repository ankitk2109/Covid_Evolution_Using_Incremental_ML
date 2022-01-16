# Covid-19 Evolution Using Incremental ML

Modelling the COVID-19 virus evolution using incremental machine learning methods



## About the Experiment
In this work, we recreated the scene of the year, 2020, when the pandemic erupted across the world and selected top fifty countries with more COVID-19 cases reported. We performed some experiments in which we compare state-of-the-art machine learning algorithms, such as LSTM, against online incremental machine learning algorithms to adapt them to the daily changes in the spread of the disease and predict future COVID-19 cases.



## E-print of the paper

Please find the pre-print of the paper here: https://arxiv.org/abs/2104.09325



## Experiments

1. **Experiment 1: Single-Country training**

This experiment trains the supervised ML models with a single-country and predicts the cases for the same single country with which the model is trained

* Mean performance of the incremental and static methods 


| Metric            | MAPE   | MAE       | RMSE       | Time(Sec)  |
| ----------------- | ------ | --------- | ---------- | ---------- |
| LSTM              | 1.732  | 3053.884  | 3311.373   | **20.525** |
| Gradient Boosting | 1.838  | 1635.620  | 1821.918   | 0.147      |
| Decision Tree     | 1.862  | 1713.391  | 1976.576   | 0.005      |
| Random Forest     | 2.124  | 2031.514  | 2187.974   | 0.197      |
| Bayesian Ridge    | 7.501  | 5934.957  | 7165.991   | 0.011      |
| HAT*              | 17.802 | 11502.321 | 15653.924  | 0.146      |
| HT*               | 18.467 | 13548.34  | 19412.316  | 0.115      |
| ARF*              | 28.084 | 17336.117 | 23923.305  | 2.909      |
| PA*               | 43.646 | 91809.182 | 111579.093 | 0.002      |
| Linear SVR        | 51.521 | 31370.912 | 37518.076  | 0.02       |



2. **Experiment 2: Multi-Country training**

This second experiment predicts over the same 50 countries at the same eight points as Experiment I, but this time we
train the model with 50 countries rather than training it with a single country as in Experiment I

* Mean performance of the incremental and static methods

  | Metric            | MAPE    | MAE       | RMSE       | Time(Sec) |
  | ----------------- | ------- | --------- | ---------- | --------- |
  | Linear SVR        | 24.348  | 4577.227  | 12823.977  | 1.907     |
  | HT*               | 30.874  | 5495.883  | 17380.819  | 7.265     |
  | HAT*              | 36.171  | 4802.677  | 13945.722  | 21.234    |
  | ARF*              | 41.651  | 4599.165  | 14150.65   | 163.506   |
  | LSTM              | 53.74   | 3089.347  | 8271.583   | 138.567   |
  | Bayesian Ridge    | 107.16  | 2943.008  | 7671.23    | 0.036     |
  | PA*               | 136.261 | 40040.947 | 130604.297 | 0.005     |
  | Random Forest     | 138.922 | 3196.719  | 7739.082   | 3.106     |
  | Gradient Boosting | 215.72  | 3052.013  | 7499.656   | 6.858     |
  | Decision Tree     | 254.97  | 3662.684  | 8764.345   | 0.455     |



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
                       `└───incremental_alternate`
   ```

2. Once the above folder structure is ready, make sure all the requirements are installed as mentioned in [requirement.txt](requirement.txt).

3. Run the  [preprocess.py](src/preprocess.py) file to create the data frames and files required for both experiments.

4. Make sure that  [skmultiflow](src/skmultiflow/src) folder is present in source code as we have implemented few functionalities that were not a part of skmultiflow package originally.

5. Run the  [exp1.py](src/exp1.py)  and  [exp2.py](src/exp2.py)  files to generate results.
