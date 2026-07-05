import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
sns.set_theme(style="whitegrid")

df = pd.read_csv("AI/Automobile_data.csv")
df = df.replace('?' , np.nan).dropna()

df['price'] = df["price"].astype(float)
if 'horsepower' in df.columns:
    df['horsepower'] = df["horsepower"].astype(float)
if 'curb-weight' in df.columns:
    df['curb-weight'] = df['curb-weight'].astype(float)
X = df.copy()
y = X.pop("price")

for colname in X.select_dtypes("object"):
    X[colname] , _ = X[colname].factorize()

discrete_features = X.dtypes.isin([np.int64,np.int32]).values

def make_mi_scores(X,y,discrete_features):
    mi_scores = mutual_info_regression (X,y,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores , name = "MI Scores" , index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_scores = make_mi_scores(X,y,discrete_features)
mi_scores[::3]

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending = True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width , scores)
    plt.yticks(width , ticks)
    plt.title("Mutual Information Scores")
plt.figure(dpi=100 , figsize=(8,5))
plot_mi_scores(mi_scores)
plt.show()

sns.relplot(x="curb-weight" , y="price" , data = df)
plt.show()

sns.lmplot(x="horsepower" , y = "price" , hue = "fuel-type" , data = df)
plt.show()
