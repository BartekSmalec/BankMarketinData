import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(name):
    df = pd.read_csv(name, sep=';')
    print(df.head())

    print(df.shape)

    print(df.columns)
    print(df.info)
    print(df.head())
    print(df.shape)
    return df

def preprocess_data(df):
    df.columns = [col.replace('"', '') for col in df.columns]
    df.drop(columns=['day', 'poutcome'], axis=1, inplace=True)

    le = preprocessing.LabelEncoder()
    df.job = le.fit_transform(df.job)
    df.education = le.fit_transform(df.education)
    df.housing = le.fit_transform(df.housing)
    df.loan = le.fit_transform(df.loan)
    df.month = le.fit_transform(df.month)
    df.contact = le.fit_transform(df.contact)
    df.marital = le.fit_transform(df.marital)
    df.default = le.fit_transform(df.default)
    # df.poutcome = le.fit_transform(df.poutcome)
    df.y = le.fit_transform(df.y)
    print(df.shape)
    return df
def split_data(df):
    X = df.iloc[:, 0:14]
    y = df.iloc[:, 14]
    X = np.array(X, dtype="float64")
    y = np.array(y, dtype="float64")
    return X,y

def describe_data(df):
    print("Is null")
    print(pd.isnull(df).any())
    for i in df.columns:
        #print("Value counts for \"{}\" is: {}".format(i, df[i].value_counts()))
        print("Number of unique values for \"{}\"  is: {}".format(i, df[i].nunique()))
        print("==============================================================")
        print("Describe for \"{}\" is: {}".format(i, df[i].describe()))


