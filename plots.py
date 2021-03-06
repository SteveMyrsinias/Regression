import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import warnings
import os

# Import Data
my_path = os.path.abspath(os.path.dirname(__file__))
missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
df_train = pd.read_csv(r''+my_path+'\\data\\train.csv', sep=',', na_values=missing_values )

categorical_features = df_train.select_dtypes(include=['object', 'category']).columns # get all the categorical features
print(categorical_features.shape)

for feature in categorical_features:
    sns.countplot(data=df_train, x=df_train[feature])
    plt.title('Count the distribution of '+ feature + ' Feature')
    plt.show()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = df_train.select_dtypes(include=numerics).columns # get all the numeric features
print(numeric_features.shape)

for feature in numeric_features:
    plt.boxplot(df_train[feature])
    plt.title(feature)
    plt.show()