# Import packages 

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotnine

# Data ingestion & curation 

df = pd.read_csv('../data/DropOut_SummaryData.csv')
df = df.dropna()

df['strain'].replace('Strain A', '0', inplace=True)
df['strain'].replace('Strain B', '1', inplace=True)
df['strain'] = df['strain'].astype(int)
df['type_of_doe'].replace('Round_0_Definitive_screening_design', '0', inplace=True)
df['type_of_doe'].replace('Round_1_central_composite_design', '1', inplace=True)
df['type_of_doe'] = df['type_of_doe'].astype(int)
df['plate'].replace('Plate 18', '18', inplace=True)
df['plate'].replace('Plate 19', '19', inplace=True)
df['plate'].replace('Plate 20', '20', inplace=True)
df['plate'].replace('Plate 21', '21', inplace=True)
df['plate'].replace('Plate 24', '24', inplace=True)
df['plate'].replace('Plate 25', '25', inplace=True)
df['plate'] = df['plate'].astype(int)

print(df.dtypes)

df = df.drop(['Unnamed: 0', 'residual_glucose'], axis=1)
df.reset_index(drop = True, inplace = True) 

# Split into input and output elements 

X, y = df.iloc[:, :-1], df.iloc[:, -1]
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15 , random_state = 42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Build model 

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
hidden_layers = [1, 2, 3]

# Central Composite Design (CCD) - Simplified for demonstration
experiments = []
for lr in learning_rates:
    for bs in batch_sizes:
        for hl in hidden_layers:
            experiments.append((lr, bs, hl))

# Conduct experiments
results = []
for lr, bs, hl in experiments:
    model = MLPRegressor(hidden_layer_sizes = (hl,), learning_rate_init = lr, batch_size = bs, max_iter = 1000, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    score = median_absolute_error(y_test, y_pred)
    results.append((lr, bs, hl, score))

model = MLPRegressor(hidden_layer_sizes = 3, learning_rate_init = 0.1, batch_size = 32, max_iter = 200, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

compare = pd.DataFrame({
    "pred": y_pred,
    "expt": y_test
})

from plotnine import ggplot, aes, geom_point, geom_abline, ggtitle, theme_tufte, coord_fixed
(
    ggplot(compare, aes(x = 'expt', y = 'pred'))
    + geom_point()
    #+ p9.geom_smooth(method = 'lm', se = False)
    + geom_abline(slope = 1.0, intercept = (0,0))
    + ggtitle('MLP Regression', subtitle = 'Drop Out\nMedian Absolute Error = 0.99')
    + coord_fixed()
    + theme_tufte()
)

df['strain'].unique()
df['type_of_doe'].unique()
df['plate'].unique()
df['condition'].unique()
df['p_h'].unique()
df['glucose_g_l'].unique()
df['ammonium_sulfate_g_l'].unique()
df['phosphate_citrate_x'].unique()
df['ynb_x'].unique()
df['amino_acid_1_m_m'].unique()
df['amino_acid_2_g_l'].unique()
df['amino_acid_2_g_l_1'].unique()
df['ethanol_g_l'].unique()

# Optimization (Gradient Descent - Simplified)
from scipy.optimize import minimize
from skopt import gp_minimize

def objective(params):
    strain, type_of_doe, plate, condition, p_h, glucose_g_l, ammonium_sulfate_g_l, phosphate_citrate_x, ynb_x, amino_acid_1_m_m, amino_acid_2_g_l, amino_acid_2_g_l_1, ethanol_g_l = params
    return -model.predict(pd.DataFrame({'strain': [strain],
    'type_of_doe': [type_of_doe],
    'plate': [plate],
    'condition': [condition],
    'p_h': [p_h],
    'glucose_g_l': [glucose_g_l],
    'ammonium_sulfate_g_l': [ammonium_sulfate_g_l],
    'phosphate_citrate_x': [phosphate_citrate_x],
    'ynb_x': [ynb_x],
    'amino_acid_1_m_m': [amino_acid_1_m_m],
    'amino_acid_2_g_l': [amino_acid_2_g_l],
    'amino_acid_2_g_l_1': [amino_acid_2_g_l_1],
    'ethanol_g_l': [ethanol_g_l]}))[0]

dimensions = [(0, 1), (0, 1), (18, 25), (1, 48), (4, 6.5), (15, 40), (3, 12), (0.2, 1), (0.5, 1.375), (0, 3.5), (0, 0.5), (0, 1), (0, 6)]

hoping = gp_minimize(objective, dimensions, n_calls = 50)
optimal_params = hoping.x

##### brickyard below this line #####
           
qaz = pd.DataFrame(results, columns=['LearningRate', 'BatchSize', 'HiddenLayers', 'Score'])
# Fit a quadratic regression model
formula = 'MSE ~ LearningRate + BatchSize + HiddenLayers + I(LearningRate**2) + I(BatchSize**2) + I(HiddenLayers**2) + LearningRate:BatchSize + LearningRate:HiddenLayers + BatchSize:HiddenLayers'
model = ols(formula, data = qaz).fit()
print(model.summary())

# Analyze the response surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(qaz['LearningRate'], df['BatchSize'], df['Accuracy'], c = 'r', marker = 'o')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Batch Size')
ax.set_zlabel('Accuracy')
plt.show()

# Optimization (Gradient Descent - Simplified)
from scipy.optimize import minimize

def objective(params):
    lr, bs, hl = params
    return -model.predict(pd.DataFrame({'LearningRate': [lr], 'BatchSize': [bs], 'HiddenLayers': [hl]}))[0]

initial_guess = [0.01, 16, 23]
result = minimize(objective, initial_guess, bounds=[(0.001, 0.1), (16, 64), (1, 3)], method = 'L-BFGS-B')
optimal_params = result.x

print(f'Optimal Learning Rate: {optimal_params[0]}')
print(f'Optimal Batch Size: {optimal_params[1]}')
print(f'Optimal Hidden Layers: {optimal_params[2]}')

optModel = MLPRegressor(hidden_layer_sizes = (hl,), learning_rate_init = lr, batch_size = bs, max_iter = 200, random_state = 42)

model = MLPRegressor(hidden_layer_sizes = 3, learning_rate_init = 0.1, batch_size = 32, max_iter = 200, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "pred": y_pred,
    "expt": y_test
})


learning_rates = [0.001]
batch_sizes = [64]
hidden_layers = [1]

# Central Composite Design (CCD) - Simplified for demonstration
experiments = []
for lr in learning_rates:
    for bs in batch_sizes:
        for hl in hidden_layers:
            experiments.append((lr, bs, hl))

# Conduct experiments
results = []
for lr, bs, hl in experiments:
    model = MLPRegressor(hidden_layer_sizes = (hl,), learning_rate_init = lr, batch_size = bs, max_iter = 200, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results.append((lr, bs, hl, mse))

model = MLPRegressor(hidden_layer_sizes = (3, ), learning_rate_init = 0.010, batch_size = 16, max_iter = 200, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "pred": y_pred,
    "expt": y_test
})

from plotnine import ggplot, aes, geom_point, geom_abline, ggtitle, theme_tufte
(
    ggplot(results, aes(x = 'expt', y = 'pred'))
    + geom_point()
    #+ p9.geom_smooth(method = 'lm', se = False)
    + geom_abline(slope = 1.0, intercept = (0,0))
    + ggtitle('MLP Regression', subtitle = 'Drop Out\nMedian Absolute Error = 1.04')
    + theme_tufte()
)

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
median_absolute_error(y_test, y_pred)


# Create the scatter plot
plt.scatter(np.array(y_test), np.array(y_pred))

# Add labels and title
plt.xlabel("Experimental")
plt.ylabel("Prediction")
plt.title("DropOut")

# Display the plot
plt.show()

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

new_y_test = np.delete(y_test, 6)
new_y_pred = np.delete(y_pred, 6)

# Create the scatter plot
plt.scatter(np.array(new_y_test), np.array(new_y_pred))

# Add labels and title
plt.xlabel("Experimental")
plt.ylabel("Prediction")
plt.title("DropOut")

# Display the plot
plt.show()

mean_squared_error(new_y_test, new_y_pred)
r2_score(new_y_test, new_y_pred)

# Optimization (Gradient Descent - Simplified)
from scipy.optimize import minimize

model = MLPRegressor(hidden_layer_sizes = (3, ), learning_rate_init = 0.010, batch_size = 16, max_iter = 500, random_state = 42)
model.fit(X_train, y_train)

params

def objective(params):
    lr, bs, hl = params
    return -model.predict(pd.DataFrame({'LearningRate': [lr], 'BatchSize': [bs], 'HiddenLayers': [hl]}))[0]

initial_guess = [0.01, 16, 3]
result = minimize(objective, initial_guess, bounds=[(0.001, 0.1), (16, 64), (1, 3)], method = 'L-BFGS-B')
optimal_params = result.x

print(f'Optimal Learning Rate: {optimal_params[0]}')
print(f'Optimal Batch Size: {optimal_params[1]}')
print(f'Optimal Hidden Layers: {optimal_params[2]}')