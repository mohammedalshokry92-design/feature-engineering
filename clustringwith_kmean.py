import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure" , autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10
)
df = pd.read_csv('AI/housing.csv')
X = df.loc[: , ["median_income", "latitude", "longitude"]]
print (X.head())
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
X.head()
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=X, 
    x="longitude", 
    y="latitude", 
    hue="Cluster", 
    palette="viridis", 
    alpha=0.6
) 

plt.title("K-Means Clustering (6 Clusters)")
plt.show()

X["median_house_value"] = df["median_house_value"]
sns.catplot(x="median_house_value" , y="Cluster" , data=X , kind="boxen" , height=6) ;
plt.show()
