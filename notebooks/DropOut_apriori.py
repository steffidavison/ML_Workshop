# kernel: mlenv

import os
os.getcwd()

import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for ploting frequency distribution chart
from efficient_apriori import apriori # for association analysis

df = pd.read_csv('../data/DropOut_SummaryData.csv')

df = df.drop(['Unnamed: 0', 'residual_glucose'], axis=1)

# Binning into 5 equal-width bins
df['OD60'] = pd.cut(df['mean_calc_OD60'], bins = 3)
df = df.drop(['mean_calc_OD60'], axis=1)

for j in range(3, 14):
    df[df.columns[j]] = df.columns[j] + '_' + df[df.columns[j]].astype(str)

# Put all transactions into a single list
txns = df.values.reshape(-1).tolist()

# Create a dataframe using this single list and add a column for count
df_list = pd.DataFrame(txns)
df_list['Count'] = 1

# Group by items and rename columns
df_list = df_list.groupby(by = [0], as_index = False).count().sort_values(by = ['Count'], ascending = True) # count
df_list['Percentage'] = (df_list['Count'] / df_list['Count'].sum()) # percentage
df_list = df_list.rename(columns = {0 : 'Item'})

# Show dataframe
df_list

# Draw a horizontal bar chart
plt.figure(figsize = (16,20), dpi = 300)
plt.ylabel('Item Name')
plt.xlabel('Count')
plt.barh(df_list['Item'], width = df_list['Count'], color = 'black', height = 0.8)
plt.margins(0.01)   
plt.show()

# Create a list of lists from a dataframe
txns2 = df.stack().groupby(level = 0).apply(list).tolist()
# Show what it looks like
txns2

itemsets, rules = apriori(txns2, min_support = 0.5, min_confidence = 0.8, verbosity = 1)

itemsets

for item in sorted(rules, key=lambda item: (item.lift,item.conviction), reverse=True):
    print(item)
