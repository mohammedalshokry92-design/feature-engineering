import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# Plot styling configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes", 
    labelweight="bold", 
    labelsize="large", 
    titleweight="bold", 
    titlesize=14, 
    titlepad=10
)

# Function to plot individual and cumulative variance explained by PCA components
def plot_variance(pca, width=8, dpi=100):
    fig, axs = plt.subplots(1, 2, figsize=(width, 4), dpi=dpi)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    evr = pca.explained_variance_ratio_
    
    # Plot individual explained variance on the first axis [0]
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", 
        title="% Explained Variance", 
        ylim=(0.0, 1.0)
    )
    
    # Plot cumulative explained variance on the second axis [1]
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", 
        title="% Cumulative Variance", 
        ylim=(0.0, 1.0)
    )
    return axs

# Function to compute Mutual Information (MI) scores
def make_mi_scores(x, y, discrete_features):
    mi_scores = mutual_info_regression(x, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=x.columns)
    mi_scores = mi_scores.sort_values(ascending=True) 
    return mi_scores

# 1. Load and clean the dataset
df = pd.read_csv("ai/automobile_data.csv")

# Strip hidden whitespace characters from column names
df.columns = df.columns.str.strip()

# Replace string question marks '?' with actual NaN values
df = df.replace('?', np.nan)

# Define features with hyphenated formatting to match your CSV file
features = ["highway-mpg", "engine-size", "horsepower", "curb-weight"]

# Force specified features and the target variable 'price' to be numeric
for col in features + ["price"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Safely drop rows containing missing values and reset the data index
df = df.dropna(subset=features + ["price"]).reset_index(drop=True)

# Separate independent variables (x) and target variable (y)
x = df[features].copy().astype(float)
y = df['price'].copy().astype(float)

# Standardize features using Z-score calculation formula
x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)

# 2. Apply Principal Component Analysis (PCA)
pca = PCA()
x_pca = pca.fit_transform(x_scaled)
component_names = [f'pc{i+1}' for i in range(x_pca.shape[1])]

x_pca = pd.DataFrame(x_pca, columns=component_names)

print("--- First 5 rows of Principal Components (x_pca) ---")
print(x_pca.head())

# 3. Calculate and display the Loadings Matrix
loading = pd.DataFrame(
    pca.components_.T, 
    columns=component_names, 
    index=x.columns
)
print("\n--- PCA Loadings Matrix ---")
print(loading)

# 4. Generate and render the variance plots
plot_variance(pca)
plt.show()

# 5. Compute Mutual Information scores for the new PCA components
mi_scores = make_mi_scores(x_pca, y, discrete_features=False)
print("\n--- Mutual Information Scores for PCA Components ---")
print(mi_scores)

# 6. Sort by the third component (PC3) and inspect matching vehicle characteristics
idx = x_pca["pc3"].sort_values(ascending=True).index
cols = ["make", "body-style", "horsepower", "curb-weight"]
print("\n--- Vehicle characteristics sorted by PC3 alignment ---")
print(df.iloc[idx].loc[:, cols])

# 7. Perform custom feature engineering
df["sports_or_wagon"] = x["curb-weight"] / x["horsepower"]
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);
plt.show()
print("\n--- 'sports_or_wagon' feature engineered successfully! ---")
