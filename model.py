import pickle

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline




df = pd.read_csv('train-data.csv')
df = df.dropna(subset = ["Power", "Mileage", "Engine", "Seats"])
#  Had to drop these null bhp strings
df = df[df.Power != 'null bhp']


# turn into CAD
df['Price'] = df['Price']*0.018*100000

# Function to remove units from numbers
def remove_words(column):
    return column.str.split(' ').str[0]

df['Mileage'] = remove_words(df['Mileage']).astype(float)
df['Engine'] = remove_words(df['Engine']).astype(float)
df['Power'] = remove_words(df['Power']).astype(float)
df['New_Price'] = remove_words(df['New_Price']).astype(float)

# Train test Split
target = 'Price'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


mapper = DataFrameMapper([
     ('Name',[CategoricalImputer(), LabelBinarizer()]),
     ('Location',[CategoricalImputer(), LabelBinarizer()]),
     (['Year'],[SimpleImputer(), StandardScaler()]),
     (['Kilometers_Driven'],[SimpleImputer(), StandardScaler()]),
     ('Fuel_Type',[CategoricalImputer(), LabelBinarizer()]),
     ('Transmission',[CategoricalImputer(), LabelBinarizer()]),
     ('Owner_Type',[CategoricalImputer(), LabelBinarizer()]),
     (['Mileage'], [SimpleImputer(), StandardScaler()]),
     (['Engine'], [SimpleImputer(), StandardScaler()]),
     (['Power'], [SimpleImputer(), StandardScaler()]),
     (['Seats'], [SimpleImputer(), StandardScaler()]),
     # (['New_Price'], [SimpleImputer(), StandardScaler()]),
     ], df_out=True)


# SelectPercentile
select = SelectPercentile(percentile=40)

## GridSearchCV to find best params for the pipe
# params = {
#     'iterations': [400],
#     'learning_rate': [0.1,0.5,0.9],
# }
# grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
# grid.fit(Z_train, y_train)
# print(grid.best_score_)
# print(grid.best_params_)

# CatBoostRegressor using the best params found above^
model = cb.CatBoostRegressor(iterations=400, learning_rate=0.5)

# pipe and pickle
pipe = make_pipeline(mapper, select, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))
