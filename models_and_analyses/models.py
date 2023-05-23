# %%
# Modelling the data

# %%
# Import Libraries
import pandas as pd
import psycopg2
import sklearn as sk
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV, train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Connect to the database
try:
    conn = psycopg2.connect("host = localhost dbname = ccbc_db user = postgres password = 123456")
    conn.autocommit = True # This automatically commits any changes to the database without having to call conn.commit() after each command.
    cur = conn.cursor()
except Exception as e:
    print("Unable to connect to the database")
    raise Exception(e)
else:
    print("Database connected")

# %% [markdown]
# I have an SQL query error, I need to break this down and figure out what is going on. It could definitely be a team_abbr issue. Start with writing a new SQL query to see if I can get the data I want.

# %%
# Query the database to get all the team batting and team pitching data for all years when the season is Fall, CCBC, or CCBC Champ
try:
    # Define the query
    query = """
    SELECT standings.team_abbr as team, standings.year, SUM(team_pitching.ip) as innings_pitched, SUM(team_pitching.r) as runs_against, 
        SUM(team_pitching.er) as earned_runs, SUM(team_pitching.h) as hits_against, SUM(team_pitching.bb) as walks_given, SUM(team_pitching.wp) as wild_pitches, 
        SUM(team_pitching.hbp) as hit_batters, SUM(team_pitching.so) as pitching_strikeouts, SUM(team_batting.r) as batting_runs, 
        SUM(team_batting.h) as batting_hits, SUM(team_batting.doubles), SUM(team_batting.triples), SUM(team_batting.hr) as homeruns, SUM(team_batting.rbi), 
        sum(team_batting.tb), sum(team_batting.bb) as batting_walks, SUM(team_batting.ab) as at_bats, SUM(standings.w) as wins, SUM(standings.l) as losses, 
        standings.pct as w_pct
    FROM standings
    LEFT JOIN team_pitching ON standings.team_abbr = team_pitching.team_abbr AND standings.year = team_pitching.year 
        AND standings.team_abbr = team_pitching.team_abbr
    LEFT JOIN team_batting ON standings.team_abbr = team_batting.team_abbr AND standings.year = team_batting.year 
        AND standings.season_type = team_pitching.season_type
    WHERE (standings.w + standings.l <> 0) AND (standings.season_type = 'CCBC' AND team_pitching.season_type = 'CCBC' AND team_batting.season_type = 'CCBC')
    GROUP BY standings.team_abbr, standings.year, standings.season_type
    ;
    """
    # Execute the query
    cur.execute(query)

    # Fetch the results
    result = cur.fetchall()
    
    # Convert the data into a pandas dataframe
    df = pd.DataFrame(result, columns=['team', 'year', 'innings_pitched', 'runs_against', 'earned_runs', 'hits_against', 'walks_given', 'wild_pitches', 'hit_batters', 'pitching_strikeouts', 'batting_runs', 'batting_hits', 'doubles', 'triples', 'homeruns', 'rbi', 'tb', 'batting_walks', 'at_bats', 'wins', 'losses', 'w_pct'])

except Exception as e:
    print("Unable to query the database")
    raise Exception(e)

# %%
df

# %%
# Show the columns team, year, w_pct, wins, and losses
df[['team', 'year', 'w_pct', 'wins', 'losses']]

# %%
df.columns

# %%
# Convert the first two columns to categorical variables
df['team'] = df['team'].astype('category')
df['year'] = df['year'].astype('category')

# Convert all the other columns to float
df['innings_pitched'] = df['innings_pitched'].astype('float')
df['runs_against'] = df['runs_against'].astype('float')
df['earned_runs'] = df['earned_runs'].astype('float')
df['hits_against'] = df['hits_against'].astype('float')
df['walks_given'] = df['walks_given'].astype('float')
df['wild_pitches'] = df['wild_pitches'].astype('float')
df['hit_batters'] = df['hit_batters'].astype('float')
df['pitching_strikeouts'] = df['pitching_strikeouts'].astype('float')
df['batting_runs'] = df['batting_runs'].astype('float')
df['batting_hits'] = df['batting_hits'].astype('float')
df['doubles'] = df['doubles'].astype('float')
df['triples'] = df['triples'].astype('float')
df['homeruns'] = df['homeruns'].astype('float')
df['rbi'] = df['rbi'].astype('float')
df['tb'] = df['tb'].astype('float')
df['batting_walks'] = df['batting_walks'].astype('float')
df['at_bats'] = df['at_bats'].astype('float')
df['wins'] = df['wins'].astype('float')
df['losses'] = df['losses'].astype('float')
df['w_pct'] = df['w_pct'].astype('float')

# %%
# With df create new baseball stats OPS, SLG, OBP, WHIP, K/9, BB/9, and BB/AB

# Note we are missing sac flies and sac hits and hit by pitch for batters

# Create singles which is hits minus doubles, triples, and homeruns
df['singles'] = df['batting_hits'] - df['doubles'] - df['triples'] - df['homeruns']

# Create the SLG column
df['SLG'] = df['tb'] / df['at_bats']

# Create the OBP column
df['OBP'] = (df['batting_hits'] + df['batting_walks']) / (df['at_bats'] + df['batting_walks'])

# Create the OPS column
df['OPS'] = df['OBP'] + df['SLG']

# Create the WHIP column
df['WHIP'] = (df['hits_against'] + df['walks_given']) / df['innings_pitched']

# Create the K/9 column
df['K/9'] = df['pitching_strikeouts'] / df['innings_pitched'] * 9

# Create the BB/9 column
df['BB/9'] = df['walks_given'] / df['innings_pitched'] * 9

# Create the AB/BB column
df['AB/BB'] = df['at_bats'] / df['batting_walks']

# Create the wOBA (weighted on base average) column
df['wOBA'] = (0.69 * (df['batting_walks'] / df['at_bats'])) + (0.89 * (df['singles'] / df['at_bats'])) + (1.27 * (df['doubles'] / df['at_bats'])) + (1.62 * (df['triples'] / df['at_bats'])) + (2.10 * (df['homeruns'] / df['at_bats']))

# Create Run Differential column which is runs scored minus runs allowed
df['run_differential'] = df['batting_runs'] - df['runs_against']

# %%
df.head()

# %%
df.columns

# %%
# Drop the rows with missing values
df = df.dropna() # One row was dropped

# %%
# Create a dataframe with the team and year columns removed
df2 = df.drop(['team', 'year'], axis=1)

# Create a correlation plot
f, ax = plt.subplots(figsize=(10, 8))
corr = df2.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)

# Save as a png file
plt.savefig('correlation_plot.png')

# %%
# See correlation between w_pct and other variables
corr['w_pct'].sort_values(ascending=False)
# OBP, wOBA, OPS have the highest correlation with w_pct
# WHIP, and BB/9 have the highest negative correlation with w_pct

# %%
# Plot run_differential vs w_pct with a regression line, title, and axis labels
sns.regplot(x='run_differential', y='w_pct', data=df)
plt.title('Win Percentage vs Run Differential')
plt.xlabel('Run Differential')
plt.ylabel('Win Percentage')

# %%
# The data is highly correlated so we will perform PCA to reduce the number of variables
# Create Features 
pca_features = df.drop(['w_pct', 'team', 'year','wins', 'losses'], axis=1)

scaler = StandardScaler()
pca_df = scaler.fit_transform(pca_features)

# Perform PCA
pca = PCA()
pca.fit(pca_df)

# Plot explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# Add a horizontal line at 0.95
plt.axhline(y=0.95, linestyle='--')
# Add a vertical line at 2
plt.axvline(x=7, linestyle='--')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Save as a jpg file
plt.savefig('explained_variance_ratio.jpg')

# %%
# Split the data into features (X) and target (y):

# Target is the w_pct column
target = df['w_pct']

# Create Features 
features = df.drop(['w_pct', 'team', 'year','wins', 'losses'], axis=1)

# Create features_rd which is the run_differential column
features_rd = df['run_differential']

# Scale the features using StandardScaler (Note we do not scale features_rd as it is not necessary):
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets with 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=37)

# Split the data into training and testing sets with 20% of the data for testing using run_differential
X_train_rd, X_test_rd, y_train_rd, y_test_rd = train_test_split(features_rd, target, test_size=0.2, random_state=37)

# Split the data into training and testing sets with 20% of the data for testing using pca
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, target, test_size=0.2, random_state=37)


# %%
# Get the shape of x_train and y_train
print(X_train.shape)

# %%
# Reshape the features of the run_differential data to fit the model

# Create a NumPy array of the features which is all columns besides w_pct, team, year, wins, losses
X_train_rd = X_train_rd.values.reshape(-1, 1)

# Create a NumPy array of the features which is all columns besides w_pct, team, year, wins, losses
X_test_rd = X_test_rd.values.reshape(-1, 1)

# %% [markdown]
# Kernel Ridge Models

# %%
# Run Differential

# Define a range of kernel and alpha values to search over
param_grid = {'kernel': ['linear', 'rbf', 'poly'],
              'alpha': np.logspace(-5, 2, num=8)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with KernelRidge and the parameter grid
model = KernelRidge()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_rd, y_train_rd)

# Print the best kernel and alpha parameter values
print('Best kernel:', grid.best_params_['kernel'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = KernelRidge(alpha=grid.best_params_['alpha'], kernel=grid.best_params_['kernel'])

# Fit the model using the training data
best_model.fit(X_train_rd, y_train_rd)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_rd)
rmse = np.sqrt(mean_squared_error(y_test_rd, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
# print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test_rd, y_pred))

# %%
# All Features

# Define a range of kernel and alpha values to search over
param_grid = {'kernel': ['linear', 'rbf', 'poly'],
              'alpha': np.logspace(-5, 2, num=8)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with KernelRidge and the parameter grid
model = KernelRidge()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train, y_train)

# Print the best kernel and alpha parameter values
print('Best kernel:', grid.best_params_['kernel'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = KernelRidge(alpha=grid.best_params_['alpha'], kernel=grid.best_params_['kernel'])

# Fit the model using the training data
best_model.fit(X_train, y_train)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
# print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test, y_pred))

# %%
# PCA

# Define a range of kernel and alpha values to search over
param_grid = {'kernel': ['linear', 'rbf', 'poly'],
              'alpha': np.logspace(-5, 2, num=8)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with KernelRidge and the parameter grid
model = KernelRidge()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_pca, y_train_pca)

# Print the best kernel and alpha parameter values
print('Best kernel:', grid.best_params_['kernel'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = KernelRidge(alpha=grid.best_params_['alpha'], kernel=grid.best_params_['kernel'])

# Fit the model using the training data
best_model.fit(X_train_pca, y_train_pca)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test_pca, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
# print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test_pca, y_pred))

# %% [markdown]
# Lasso Models

# %%
# Run Differential

# Define a range of alpha values to search over
param_grid = {'alpha': np.logspace(-5, 5, num=20)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with Lasso and the parameter grid
model = Lasso(max_iter=10000)
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_rd, y_train_rd)

# Print the best alpha parameter value
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = Lasso(alpha=grid.best_params_['alpha'])

best_model.fit(X_train_rd, y_train_rd)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_rd)
rmse = np.sqrt(mean_squared_error(y_test_rd, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test_rd, y_pred))

# %%
# All Features

# Define a range of alpha values to search over
param_grid = {'alpha': np.logspace(-5, 5, num=20)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with Lasso and the parameter grid
model = Lasso(max_iter=10000)
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train, y_train)

# Print the best alpha parameter value
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = Lasso(alpha=grid.best_params_['alpha'])

best_model.fit(X_train, y_train)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test, y_pred))

# %%
# PCA

# Define a range of alpha values to search over
param_grid = {'alpha': np.logspace(-5, 5, num=20)}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with Lasso and the parameter grid
model = Lasso(max_iter=10000)
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_pca, y_train_pca)

# Print the best alpha parameter value
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = Lasso(alpha=grid.best_params_['alpha'])

best_model.fit(X_train_pca, y_train_pca)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test_pca, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the coefficients of the best model
print(best_model.coef_)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test_pca, y_pred))

# %% [markdown]
# Gradient Boosting Regressor

# %%
# Run Differential

# Define a range of n_estimators, learning_rate, and alpha values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'learning_rate': [0.01, 0.1, 1.0],
              'alpha': [0.1, 0.5, 1.0]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with GradientBoostingRegressor and the parameter grid
model = GradientBoostingRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_rd, y_train_rd)

# Print the best n_estimators, learning_rate, and alpha parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best learning_rate:', grid.best_params_['learning_rate'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = GradientBoostingRegressor(n_estimators=grid.best_params_['n_estimators'],
                                        learning_rate=grid.best_params_['learning_rate'],
                                        alpha=grid.best_params_['alpha'])


# Fit the model using the training data
best_model.fit(X_train_rd, y_train_rd)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_rd)
rmse = np.sqrt(mean_squared_error(y_test_rd, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test_rd,y_pred))


# %%
# All Features

# Define a range of n_estimators, learning_rate, and alpha values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'learning_rate': [0.01, 0.1, 1.0],
              'alpha': [0.1, 0.5, 1.0]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with GradientBoostingRegressor and the parameter grid
model = GradientBoostingRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train, y_train)

# Print the best n_estimators, learning_rate, and alpha parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best learning_rate:', grid.best_params_['learning_rate'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = GradientBoostingRegressor(n_estimators=grid.best_params_['n_estimators'],
                                        learning_rate=grid.best_params_['learning_rate'],
                                        alpha=grid.best_params_['alpha'])


# Fit the model using the training data
best_model.fit(X_train, y_train)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the R^2 score of the best model
print("R^2: ", r2_score(y_test,y_pred))


# %%
# PCA

# Define a range of n_estimators, learning_rate, and alpha values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'learning_rate': [0.01, 0.1, 1.0],
              'alpha': [0.1, 0.5, 1.0]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with GradientBoostingRegressor and the parameter grid
model = GradientBoostingRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_pca, y_train_pca)

# Print the best n_estimators, learning_rate, and alpha parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best learning_rate:', grid.best_params_['learning_rate'])
print('Best alpha:', grid.best_params_['alpha'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = GradientBoostingRegressor(n_estimators=grid.best_params_['n_estimators'],
                                        learning_rate=grid.best_params_['learning_rate'],
                                        alpha=grid.best_params_['alpha'])


# Fit the model using the training data
best_model.fit(X_train_pca, y_train_pca)

# Get best model MSE on the test set
y_pred = best_model.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test_pca, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the R^2 score of the best model
print("Test R^2: ", r2_score(y_test_pca,y_pred))


# %% [markdown]
# Random Forest Regressor

# %%
# Run Differential

# Define a range of n_estimators and max_depth values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [2, 4, 6],
              'max_features': [0.5, 0.75, 1.0],
              'bootstrap': [True, False]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with RandomForestRegressor and the parameter grid
model = RandomForestRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_rd, y_train_rd)

# Print the best n_estimators, max_depth, max_features, and bootstrap parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best max_depth:', grid.best_params_['max_depth'])
print('Best max_features:', grid.best_params_['max_features'])
print('Best bootstrap:', grid.best_params_['bootstrap'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = RandomForestRegressor(n_estimators=grid.best_params_['n_estimators'],
                                    max_depth=grid.best_params_['max_depth'],
                                    max_features=grid.best_params_['max_features'],
                                    bootstrap=grid.best_params_['bootstrap'])
# Create the model
model.fit(X_train_rd, y_train_rd)

# Get best model MSE on the test set
y_pred = model.predict(X_test_rd)
rmse = np.sqrt(mean_squared_error(y_test_rd, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the feature importances of the best model
# print('Feature Importances:', model.feature_importances_)

# Print the R^2 score of the best model
print("Test R^2: ", r2_score(y_test,y_pred))


# %%
# All Features

# Define a range of n_estimators and max_depth values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [2, 4, 6],
              'max_features': [0.5, 0.75, 1.0],
              'bootstrap': [True, False]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with RandomForestRegressor and the parameter grid
model = RandomForestRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train, y_train)

# Print the best n_estimators, max_depth, max_features, and bootstrap parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best max_depth:', grid.best_params_['max_depth'])
print('Best max_features:', grid.best_params_['max_features'])
print('Best bootstrap:', grid.best_params_['bootstrap'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = RandomForestRegressor(n_estimators=grid.best_params_['n_estimators'],
                                    max_depth=grid.best_params_['max_depth'],
                                    max_features=grid.best_params_['max_features'],
                                    bootstrap=grid.best_params_['bootstrap'])

# Create the model
model.fit(X_train, y_train)

# Get best model MSE on the test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the feature importances of the best model
# print('Feature Importances:', model.feature_importances_)

# Print the R^2 score of the best model
print("Test R^2: ", r2_score(y_test,y_pred))


# %%
# PCA

# Define a range of n_estimators and max_depth values to search over
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [2, 4, 6],
              'max_features': [0.5, 0.75, 1.0],
              'bootstrap': [True, False]}

# Initialize the LOOCV object
loo = LeaveOneOut()

# Initialize a GridSearchCV object with RandomForestRegressor and the parameter grid
model = RandomForestRegressor()
grid = GridSearchCV(model, param_grid, cv=loo, scoring='neg_mean_squared_error')

# Fit the grid search object on the data
grid.fit(X_train_pca, y_train_pca)

# Print the best n_estimators, max_depth, max_features, and bootstrap parameter values
print('Best n_estimators:', grid.best_params_['n_estimators'])
print('Best max_depth:', grid.best_params_['max_depth'])
print('Best max_features:', grid.best_params_['max_features'])
print('Best bootstrap:', grid.best_params_['bootstrap'])
print('Best negative mean squared error:', grid.best_score_)

# Optionally, fit the model on the entire data using the best parameters
best_model = RandomForestRegressor(n_estimators=grid.best_params_['n_estimators'],
                                    max_depth=grid.best_params_['max_depth'],
                                    max_features=grid.best_params_['max_features'],
                                    bootstrap=grid.best_params_['bootstrap'])


# Create the model
model.fit(X_train_pca, y_train_pca)

# Get best model MSE on the test set
y_pred = model.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test_pca, y_pred))

# Print the test set MSE
print('Test set RMSE:', rmse)

# Print the feature importances of the best model
# print('Feature Importances:', model.feature_importances_)

# Print the R^2 score of the best model
print("Test R^2: ", r2_score(y_test_pca,y_pred))


# %% [markdown]
# Simple Linear Regression

# %%
# Simple Linear Regression

# split the data into the predictor variable and response variable
X = df['run_differential'].values.reshape(-1, 1)
y = df['w_pct'].values.reshape(-1, 1)
# Note that since we are solving a simple regression here we will directly fit the model on the entire data

# Initialize a LinearRegression object
model = LinearRegression()

# Fit the model on the data
model.fit(X, y)

# Get model MSE on the test set
y_pred = model.predict(X)

# Calculate the mean squared error
rmse = np.sqrt(mean_squared_error(y, y_pred))

# calculate the R-squared value
r_squared = model.score(X, y)

# Print the MSE
print('Linear Regression RMSE:', rmse)

# Print the R^2 score of the model
print("R^2: ", r2_score(y, y_pred))

# Print the coefficients of the model
print('Coefficients:', model.coef_)

# Print the intercept of the model
print('Intercept:', model.intercept_)

# %%
# Plot the Regression

# plot the data points
plt.scatter(X, y)

# plot the line of best fit
plt.plot(X, model.predict(X), color='red')

# add labels and title to the plot
plt.xlabel('Run Differential')
plt.ylabel('Winning Percentage')
plt.title('Linear Regression (R-squared = {:.2f}, RMSE = {:.2f})'.format(r_squared, rmse))

# show the plot
plt.show()

# Save the plot as a PNG file
plt.savefig('linear_regression.png')

# %%
# Get the teams with the top 5 run differentials
top_rd = df.sort_values('run_differential', ascending=False).head()
top_rd


