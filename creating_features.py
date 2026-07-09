import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure" , autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
accidents = pd.read_csv("AI/US_Accidents_March23.csv")
autos = pd.read_csv("AI/Automobile_data.csv")
concrete = pd.read_excel("AI/Concrete_Data.xls")
customer = pd.read_csv("AI/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")

concrete.columns = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age", "ConcreteCompressiveStrength"]

autos["stroke"] = pd.to_numeric(autos["stroke"], errors='coerce')
autos["bore"] = pd.to_numeric(autos["bore"], errors='coerce')

autos["stroke_ratio"] = autos.stroke/autos.bore
autos[["stroke","bore","stroke_ratio"]].head()

cylinder_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
autos["num-of-cylinders"] = autos["num-of-cylinders"].map(cylinder_map)

autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos["num-of-cylinders"]
)

accidents["LogWindSpeed"] = accidents["Wind_Speed(mph)"].apply(np.log1p)

# رسم البيانات بالأسماء الصحيحة
fig , axs = plt.subplots(1,2,figsize=(10,4))
sns.kdeplot(accidents["Wind_Speed(mph)"] , fill=True , ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed , fill=True , ax=axs[1])

road_features = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal"
]

accidents["RoadFeatures"]= accidents[road_features].sum(axis=1)
accidents[road_features + ["RoadFeatures"]].head(10)
components = ["Cement" , "BlastFurnaceSlag" , "FlyAsh" , "Water","Superplasticizer" , "CoarseAggregate" ,"FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)
concrete[components + ["Components"]].head(10)

customer [["Type" , "Level"]] = (
    customer["Policy"]
    .str
    .split(" " , expand=True)
)
customer[["Policy" , "Type" , "Level"]].head(10)

autos["make_and_style"] = autos["make"] + "-" + autos["body-style"]
autos[["make" , "body-style" , "make_and_style"]].head()
customer["AverageIncome"]=(
    customer.groupby("State")
    ["Income"]
    .transform("mean")
)
customer[["State","Income","AverageIncome"]].head(10)

customer["StateFreq"]=(
    customer.groupby("State")
    ["State"]
    .transform("count")
    /customer.State.count()
)
customer[["State" , "StateFreq"]].head(10)

df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

df_train["AverageClaim"] = df_train.groupby("Coverage")["Total Claim Amount"].transform("mean")

df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left"
)
df_valid["AverageClaim"] = df_valid["AverageClaim"].fillna(df_train["Total Claim Amount"].mean())
print(df_valid[["Coverage", "AverageClaim"]].head(10))
plt.show()
