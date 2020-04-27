import pickle

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline


df = pd.read_csv('cars1.csv')
# df.info()

# Dropping fake prices
df = df[df.price > 100]

# Dropping cars older than 20yrs
df = df[df.year > 2000]

# Dropping cars with over 1,000,000 kms - most likely user added extra digit by accident
df = df[df.odometer < 1000000]

df['name'] = df['manufacturer'].str.title() + ' ' + df['model'].str.title()
df['name']


# Encode models
le = LabelEncoder()
le.fit(df['name'])
df['model'] = le.transform(df['name'])

# Train Test SPlit
target = 'price'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# DataFrame Mapper
mapper = DataFrameMapper([
#     ('region', LabelBinarizer()),
    (['year'], StandardScaler()),
    # ('manufacturer',[CategoricalImputer(), LabelBinarizer()]),
    ('model', [CategoricalImputer()]),
    ('cylinders', [CategoricalImputer(), LabelBinarizer()]),
    ('fuel', [CategoricalImputer(), LabelBinarizer()]),
    (['odometer'], [SimpleImputer(), StandardScaler()]),
    # ('title_status', [CategoricalImputer(), LabelBinarizer()]),
    ('transmission', [CategoricalImputer(), LabelBinarizer()]),
    # (['vin'], StandardScaler()),
    # ('type', [CategoricalImputer(), LabelBinarizer()]),
    ('paint_color', [CategoricalImputer(), LabelBinarizer()]),
    ('condition', [CategoricalImputer(), LabelBinarizer()]),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)



# # GridSearchCV to find best params for the pipe
# params = {
#     'iterations': [100,500],
#     'learning_rate': [0.1,0.3,0.7],
#     'depth': [4, 10],
# }
# grid = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
# grid.fit(Z_train, y_train)
# print(grid.best_score_)
# print(grid.best_params_)




# CatBoostRegressor using the best params found above^

model = cb.CatBoostRegressor(depth= 10, iterations= 500, learning_rate =0.3)
model.fit(Z_train,y_train)
print(model.score(Z_train,y_train))
print(model.score(Z_test, y_test))

# # pipe and pickle
# pipe = make_pipeline(mapper, model)
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)
# pickle.dump(pipe, open('pipe.pkl', 'wb'))
