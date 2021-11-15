# Intro to Machine Learning

[![hackmd-github-sync-badge](https://hackmd.io/Q38exfLxSXGUquksT3WbDA/badge)](https://hackmd.io/Q38exfLxSXGUquksT3WbDA)


Decision Tree

capturing patterns from data is called fitting or training the model; the data used to train the model is called training data
<p>&nbsp;</p>

**Pandas**

Pandas library - a tool used by data scientists for exploring and manipulating data, abbreviated as pd

DataFrame - holds the type of data you might think of as a table

```
import pandas as pd

#path of file to read
file_path = '../file.csv'

#read file into var home_data
home_data = pd.read_csv(file_path)

#print summary statistics of the data
home_data.describe()
```

columns property of the DataFrame e.g. `home_data.columns`

to drop variables with missing data use dropna e.g. `home_data.dropna(axis=0)`

*Aside; there is also a Pandas course on kaggle.com*

There are many ways to select a subset of your data. ..in this example we use two approaches:
1. Dot-notation, which we use to select the "prediction target"
2. Selecting with a column list, which we use to select the "features"

You can extract a column or variable that you want to predict with dot-notation - this is called the prediction target, and by convention is also called y e.g. `y = home_data.Price`

Columns input into our model are called "features" and are used to make predictions e.g. `home_features = ['Rooms', 'Bathroom', 'Landsize']` - by convention this data is called X e.g. `X = home_data[home_features]`
<p>&nbsp;</p>

**Building your model**

scikit-learn library to create models; written as sklearn

The steps to building and using a model are:

* Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
* Fit: Capture patterns from provided data. This is the heart of modeling.
* Predict: Just what it sounds like.
* Evaluate: Determine how accurate the model's predictions are.

```
from sklearn.tree import DecisionTreeRegressor

# Define model; specify a number for random_state to ensure same results each run
home_model = DecisionTreeRegressor(random_state=1)

# Fit model
home_model.fit(X, y)
```

```
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(home_model.predict(X.head()))
```
<p>&nbsp;</p>

**Model Validation**

mean_absolute_error (MAE)

```
from sklearn.metrics import mean_absolute_error

predicted_home_prices = home_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

the problem with "in-sample" scores - using a single "sample" of data for both building and predicting the model and evaluating it

a model's practical value comes from making predictions on new data; we measure performance on data that wasn't used to build the model - one way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before, this data is called *validation data*

the scikit-learn library has a function - train_test_split, to break up the data into two pieces; we'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate mean_absolute_error

```
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
home_model = DecisionTreeRegressor()

# Fit model
home_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = home_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
```
<p>&nbsp;</p>

**Underfitting and Overfitting**

Overfitting - where a model matches the training data almost perfectly but does poorly in validation and other new data

Underfitting - where a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data
<p>&nbsp;</p>

**Random Forests**

Random forest - the random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree; it generally has much better predictive accuracy than a single decision tree

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
home_preds = forest_model.predict(val_X)
print(mean_absolute_error(home_preds, val_y))
```

