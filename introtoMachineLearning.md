# Intro to Machine Learning

Decision Tree

capturing patterns from data is called fitting or training the model
the data used to train the model is called training data

**Pandas**
Pandas library - a tool used by data scientists for exploring and manipulating data, abbrev. as pd

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

There are many ways to select a subset of your data. ..two approaches:
1. Dot notation, which we use to select the "prediction target"
2. Selecting with a column list, which we use to select the "features"


