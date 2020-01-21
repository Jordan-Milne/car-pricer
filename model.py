import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
import catboost as cb

import pickle

df = pd.read_csv('train-data.csv')
df = df.dropna(subset = ["Power", "Mileage", "Engine", "Seats"])
#  Had to drop these ANNOYING null bhp strings
df = df[df.Power != 'null bhp']


# turn into CAD
df['Price'] = df['Price']*0.018*100000

# Couldnt get the FunctionTransformer to work :/
def remove_words(column):
    return column.str.split(' ').str[0]

df['Mileage'] = remove_words(df['Mileage']).astype(float)
df['Engine'] = remove_words(df['Engine']).astype(float)
df['Power'] = remove_words(df['Power']).astype(float)
df['New_Price'] = remove_words(df['New_Price']).astype(float)

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



Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

# SelectPercentile
select = SelectPercentile(percentile=40)
select.fit(Z_train, y_train)
Z_train = select.transform(Z_train)
Z_test = select.transform(Z_test)

## Plain LinearRegression

# model = LinearRegression().fit(Z_train, y_train)
# print(model.score(Z_train, y_train))
# print(model.score(Z_test, y_test))


##  Gridsearch ... Ridge beat Lasso with score of 0.76

# model = Ridge()
# params = {
#     'fit_intercept': [True, False],
#     'alpha': [0.01,0.1,1]
# }
# grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
# grid.fit(Z_train, y_train)
# print(grid.best_score_)
# print(grid.best_params_)

#  CATBOOOOOOOOST
model = cb.CatBoostRegressor(
    iterations=100,
    learning_rate=0.5,
)

model.fit(
    Z_train, y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot=False,
)

model.score(Z_test, y_test)

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))


## Below is sample predicting
# X_train.sample().to_dict(orient='list')
#
# new = pd.DataFrame({
#
#  'Name': ['Hyundai i20 Active S Diesel'],
#  'Location': ['Jaipur'],
#  'Year': [2015],
#  'Kilometers_Driven': [81310],
#  'Fuel_Type': ['Diesel'],
#  'Transmission': ['Manual'],
#  'Owner_Type': ['First'],
#  'Mileage': [21.19],
#  'Engine': [1396.0],
#  'Power': [88.76],
#  'Seats': [4.0],
# })
#
# type(pipe.predict(new)[0])
#
# prediction = float(pipe.predict(new)[0])
# prediction
