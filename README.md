# Covid Evolution Using Incremental ML
 
Modelling the COVID-19 virus evolution using incremental machine learning methods

## Abstract
The investment of time and resources for better strategies and methodologies to tackle a potential pandemic is key to deal with potential outbreaks of new variants or other viruses in the future. In this work, we recreated the scene of a year ago, 2020, when the pandemic erupted across the world for the fifty countries with more COVID-19 cases reported. We performed some experiments in which we compare state-of-the-art machine learning algorithms, such as LSTM, against online incremental machine learning algorithms to adapt them to the daily changes in the spread of the disease and predict future COVID-19 cases.

To compare the methods, we performed three experiments: In the first one, we trained the models using only data from the country we predicted. In the second one, we use data from all fifty countries to train and predict each of them. In the first and second experiment, we used a static hold-out approach for all methods. In the third experiment, we trained the incremental methods sequentially, using a prequential evaluation. This scheme is not suitable for most state-of-the-art machine learning algorithms because they need to be retrained from scratch for every batch of predictions, causing a computational burden.


## Approach for evaluation of the applied ML methods

![Evaluation of the applied ML methods](https://user-images.githubusercontent.com/26432753/121810913-73fcc380-cc5a-11eb-9228-1f5692bdf2e4.png)


## LSTM Architecture Used

![LSTM Architecture](https://user-images.githubusercontent.com/26432753/121810971-ae666080-cc5a-11eb-94a0-e11abb9c3d21.png)


## Experiments

1. __Experiment 1:__ Single-Country training

This experiment trains the supervised ML models with a single-country and predicts the cases for the same singlecountry with which the model is trained

* Mean performance of the incremental and static methods 
![Experiment 1](https://user-images.githubusercontent.com/26432753/121811077-15841500-cc5b-11eb-900a-6f57c08c86eb.png)

* Boxplot for MAPE per algorithm
![Exp1 Box Plot](https://user-images.githubusercontent.com/26432753/121811110-3482a700-cc5b-11eb-8929-907d066638b8.png)



2. __Experiment 2:__ Multi-Country training

This second experiment predicts over the same 50 countries at the same eight points as Experiment I, but this time we
train the model with 50 countries rather than training it with a single country as in Experiment I

* Mean performance of the incremental and static methods
![Experiment 2](https://user-images.githubusercontent.com/26432753/121811276-c7bbdc80-cc5b-11eb-985e-e5f0a8f4ec73.png)


* MAPE per algorithm for the multi-country experiment
![Exp2 Box Plot](https://user-images.githubusercontent.com/26432753/121811218-a529c380-cc5b-11eb-9fbe-fbe2077d8912.png)


## Evolution of MAPE
![Evolution of MAPE](https://user-images.githubusercontent.com/26432753/121811303-e4581480-cc5b-11eb-828a-aa7d4e4507aa.png)

