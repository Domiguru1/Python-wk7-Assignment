# %% [markdown]
# # Iris Dataset Analysis
# 
# This notebook performs an exploratory data analysis on the famous Iris flower dataset.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style('whitegrid')

# %% [markdown]
# ## 1. Data Loading

# %%
# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=column_names)

# %%
# Display the first few rows
print("First 5 rows of the dataset:")
iris_df.head()

# %% [markdown]
# ## 2. Data Exploration

# %%
# Check dataset structure
print("\nDataset info:")
iris_df.info()

# %%
# Check for missing values
print("\nMissing values per column:")
iris_df.isnull().sum()

# %% [markdown]
# ## 3. Data Cleaning
# 
# (No missing values in this dataset, so no cleaning needed)

# %%
# Verify no missing values
assert iris_df.isnull().sum().sum() == 0, "There are missing values in the dataset"

# %% [markdown]
# ## 4. Basic Statistics

# %%
# Compute basic statistics
print("\nBasic statistics for numerical columns:")
iris_df.describe()

# %%
# Statistics by species
print("\nMean values by species:")
iris_df.groupby('species').mean()

# %% [markdown]
# ## 5. Observations and Findings
# 
# From the initial analysis:
# - The dataset contains 150 entries with no missing values
# - There are 3 species of Iris flowers: Iris-setosa, Iris-versicolor, Iris-virginica
# - Setosa flowers tend to have smaller petals but wider sepals compared to the other species
# - Virginica flowers have the largest petal dimensions on average

# %% [markdown]
# ## 6. Data Visualization

# %%
# Create a figure with multiple plots
plt.figure(figsize=(15, 10))

# %% [markdown]
# ### Visualization 1: Line Chart (Trend of Sepal Length by Species)
# 
# Since this isn't time series data, we'll use the index as a pseudo-time variable to demonstrate a line chart.

# %%
plt.subplot(2, 2, 1)
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species]
    plt.plot(species_data.index, species_data['sepal_length'], label=species)
plt.title('Trend of Sepal Length by Species')
plt.xlabel('Observation Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# %% [markdown]
# ### Visualization 2: Bar Chart (Average Petal Length by Species)

# %%
plt.subplot(2, 2, 2)
avg_petal_length = iris_df.groupby('species')['petal_length'].mean()
avg_petal_length.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)

# %% [markdown]
# ### Visualization 3: Histogram (Distribution of Sepal Width)

# %%
plt.subplot(2, 2, 3)
sns.histplot(iris_df['sepal_width'], bins=15, kde=True, color='purple')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# %% [markdown]
# ### Visualization 4: Scatter Plot (Sepal Length vs Petal Length)

# %%
plt.subplot(2, 2, 4)
sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species', 
                palette=['blue', 'orange', 'green'], s=100)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# %%
# Adjust layout and display all plots
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Additional Findings from Visualizations
# 
# 1. The line chart shows that Setosa flowers consistently have shorter sepals than the other species.
# 2. The bar chart confirms that Setosa has the shortest petals on average, while Virginica has the longest.
# 3. The sepal width distribution appears approximately normal with most values between 2.5cm and 3.5cm.
# 4. The scatter plot reveals a strong positive correlation between sepal and petal length, with clear separation between species.