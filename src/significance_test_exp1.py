from src.helper import *

# @Andres: Please run [4]
# Only run this cell if manually uploaded the results

summary_table_countrywise_incremental = pd.read_csv(
    r'../results/running_results/content/Result/exp1/summary/top_countries_MAPE_summary_table_incremental.csv',
    index_col='Unnamed: 0')
summary_table_countrywise_static = pd.read_csv(
    r'../results/running_results/content/Result/exp1/summary/top_countries_MAPE_summary_table_static.csv',
    index_col='Unnamed: 0')
summary_table_countrywise_static.index.name = None
print(summary_table_countrywise_incremental.head())
print(summary_table_countrywise_static.head())

# @Andres: Please run [5]
# EXP1
# Significance results for Experiment 1
err_metric_for_significance = 'MAPE'
significance_thresh = 0.01

# Concatenating a population of all results (as in boxplot) for experiment 1
# concated_df = pd.concat([summary_table_countrywise_incremental.transpose(), summary_table_countrywise_static.transpose()]).transpose().drop(columns=['EvaluationMeasurement'], axis=1)
concated_df = pd.concat([summary_table_countrywise_incremental, summary_table_countrywise_static]).transpose().drop(
    columns=['EvaluationMeasurement'], axis=1)

# COnverting to float type
concated_df = concated_df.astype('float64')
concated_df

# @Andres: Please run [6]
# Selecting the best algorithm for statistical comparisons
# We want to know if the best is statistically significantly better compared to the rest.
best_algo = concated_df.mean().sort_values(ascending=True).index[0]
best_algo

# @Andres: Please run [7]
print('AVG results across countries')
concated_df.mean()

# @Andres: Please run [8]
print('STDEV across countries')
concated_df.std()

# @Andres: Please run [9]
# Iterate through all the other algorithms to see if the difference in results is significant
competitors = list(concated_df.columns)
competitors.remove(best_algo)

for significance_thresh in [0.01, 0.05]:
    print(f'Running significane at: {significance_thresh}')
    for competitor in competitors:
        # print(competitor)
        pval, significant = check_significance(concated_df[best_algo], concated_df[competitor],
                                               significance_at=significance_thresh)
        print(f'Comparison of {best_algo} to {competitor} pvalue: {pval}   /  significant?: {significant}')

# @Andres: Please run [10]
# Iterate through all the other algorithms to see if the difference in results is significant
best_algo2 = concated_df.mean().sort_values(ascending=True).index[1]
competitors = list(concated_df.columns)
competitors.remove(best_algo2)
for significance_thresh in [0.01, 0.05]:
    print(f'Running significane at: {significance_thresh}')
    for competitor in competitors:
        # print(competitor)
        pval, significant = check_significance(concated_df[best_algo2], concated_df[competitor],
                                               significance_at=significance_thresh)
        print(f'Comparison of {best_algo2} to {competitor} pvalue: {pval}   /  significant?: {significant}')
