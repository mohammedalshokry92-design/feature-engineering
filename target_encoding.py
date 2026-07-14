import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import MEstimateEncoder

# Ignore warnings and set up global plotting configurations
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes", labelweight="bold", labelsize="large", titlesize=14, titlepad=10
)

# --- Part 1: Automobile Dataset (Basic Encoding) ---
autos = pd.read_csv("AI/Automobile_data.csv")
autos["price"] = pd.to_numeric(autos["price"], errors="coerce")

# Calculate the mean price for each make as a basic target encoding
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
print("--- Automobile Data Sample ---")
print(autos[["make", "price", "make_encoded"]].head(10))


# --- Part 2: Movie Dataset & Rating Merge ---
# Load both movies and ratings datasets from your directory
movies_df = pd.read_csv("AI/movie.csv")
ratings_df = pd.read_csv("AI/rating.csv")

# Merge them on 'movieId' (or the correct common ID column) to get the ratings
# This creates the 'Rating' column that was missing in movie.csv
df = pd.merge(ratings_df, movies_df, on="movieId")

# Set the categorical column you want to encode (e.g., 'genres' or 'Zipcode')
# Change 'genres' to the exact column name present in your movie.csv if needed
target_col = "genres" 

print(f"\nNumber of Unique Categories in {target_col}: {df[target_col].nunique()}")

# Split features (X) and target variable (y)
X = df.copy()
y = X.pop("rating")  # 'rating' comes from rating.csv

# Split data: 25% for encoding and 75% for training
X_encode = X.sample(frac=0.25, random_state=42)
y_encode = y[X_encode.index]

X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]

# Apply M-Estimate target encoder on the selected categorical column
encoder = MEstimateEncoder(cols=[target_col], m=5.0)
encoder.fit(X_encode, y_encode)
X_train = encoder.transform(X_pretrain)


# --- Part 3: Data Visualization ---
plt.figure(dpi=90)

# Plot the distribution of actual ratings
ax = sns.histplot(
    y, kde=False, stat="density", color="skyblue", label="Actual Rating"
)
# Plot the density curve of the target encoded features
ax = sns.kdeplot(X_train[target_col], color="r", ax=ax, label="Encoded Feature")

ax.set_xlabel("Rating")
ax.legend()
plt.show()