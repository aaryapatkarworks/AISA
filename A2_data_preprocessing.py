import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("AISA LAB 2: DATA COLLECTION AND PREPROCESSING")
print("="*60)

# Task 1 & 2: Load and Read the dataset
print("\n1. LOADING THE DATASET")
print("-" * 30)

# Load the CSV file
try:
    df = pd.read_csv('sample_dataset.csv')
    print("Dataset loaded successfully from 'sample_dataset.csv'!")
except FileNotFoundError:
    print("CSV file not found. Please ensure 'sample_dataset.csv' is in the same directory.")
    exit()

# Task 3: Display the dataset
print("\n2. DISPLAYING THE DATASET")
print("-" * 30)
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# Task 4: Identify missing values
print("\n3. IDENTIFYING MISSING VALUES")
print("-" * 35)

print("Missing values count per column:")
missing_count = df.isnull().sum()
print(missing_count)

print("\nMissing values percentage per column:")
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percent.round(2)
})
print(missing_info)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

print("\nRows with missing values:")
rows_with_missing = df[df.isnull().any(axis=1)]
print(rows_with_missing)

# Task 5: Handle missing values by dropping rows
print("\n4. HANDLING MISSING VALUES - DROPPING ROWS")
print("-" * 45)

# Original dataset size
print(f"Original dataset size: {df.shape}")

# Scenario 1: Drop all rows with any NaN values
df_drop_any = df.dropna()
print(f"After dropping rows with any NaN: {df_drop_any.shape}")
print("Rows remaining:")
print(df_drop_any)

# Scenario 2: Drop only if entire row has NaN values
df_drop_all = df.dropna(how='all')
print(f"\nAfter dropping rows where all values are NaN: {df_drop_all.shape}")

# Scenario 3: Drop rows with more than 2 NaN values
df_drop_thresh = df.dropna(thresh=len(df.columns)-2)  # Keep rows with at most 2 NaN values
print(f"After dropping rows with more than 2 NaN values: {df_drop_thresh.shape}")

# Scenario 4: Drop NaN in specific column (e.g., 'Name')
df_drop_specific = df.dropna(subset=['Name'])
print(f"After dropping rows with NaN in 'Name' column: {df_drop_specific.shape}")

# Task 6: Use default values to handle missing data
print("\n5. HANDLING MISSING VALUES - DEFAULT VALUES")
print("-" * 45)

df_default = df.copy()

# Fill missing values with default values
df_default['Name'].fillna('Unknown', inplace=True)
df_default['Age'].fillna(0, inplace=True)
df_default['Salary'].fillna(0, inplace=True)
df_default['Department'].fillna('Not Specified', inplace=True)
df_default['Experience'].fillna(0, inplace=True)

print("After filling with default values:")
print("Missing values count:")
print(df_default.isnull().sum())
print("\nDataset with default values:")
print(df_default)

# Task 7: Impute values using mean, median, etc.
print("\n6. HANDLING MISSING VALUES - IMPUTATION")
print("-" * 42)

df_imputed = df.copy()

# For numerical columns - use mean/median imputation
numerical_cols = ['Age', 'Salary', 'Experience']
categorical_cols = ['Name', 'Department']

print("Before imputation:")
for col in numerical_cols:
    print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

# Mean imputation for numerical columns
imputer_mean = SimpleImputer(strategy='mean')
df_imputed[numerical_cols] = imputer_mean.fit_transform(df_imputed[numerical_cols])

# Most frequent imputation for categorical columns
imputer_mode = SimpleImputer(strategy='most_frequent')
df_imputed[categorical_cols] = imputer_mode.fit_transform(df_imputed[categorical_cols])

print("\nAfter imputation with mean/mode:")
print("Missing values count:")
print(df_imputed.isnull().sum())

print("\nDataset after imputation:")
print(df_imputed)

print("\nComparison of imputation methods:")
for col in numerical_cols:
    print(f"{col} - After Imputation Mean: {df_imputed[col].mean():.2f}")

# Task 8: Identify Duplicates
print("\n7. IDENTIFYING DUPLICATES")
print("-" * 30)

print("Checking for duplicate rows:")
duplicates = df_imputed.duplicated()
print(f"Number of exact duplicate rows: {duplicates.sum()}")

if duplicates.sum() > 0:
    print("\nExact duplicate rows:")
    print(df_imputed[duplicates])
    
    print("\nAll instances of duplicated data:")
    duplicate_rows = df_imputed[df_imputed.duplicated(keep=False)]
    print(duplicate_rows.sort_values('ID'))

# Check for duplicates based on specific columns
print("\nChecking duplicates based on 'Name' and 'Age':")
name_age_duplicates = df_imputed.duplicated(subset=['Name', 'Age'], keep=False)
print(f"Number of rows with duplicate Name-Age combinations: {name_age_duplicates.sum()}")

if name_age_duplicates.sum() > 0:
    print("Rows with duplicate Name-Age combinations:")
    print(df_imputed[name_age_duplicates].sort_values(['Name', 'Age']))

# Task 9: Remove duplicates
print("\n8. REMOVING DUPLICATES")
print("-" * 25)

print(f"Shape before removing duplicates: {df_imputed.shape}")

# Remove exact duplicates
df_no_duplicates = df_imputed.drop_duplicates()
print(f"Shape after removing exact duplicates: {df_no_duplicates.shape}")

print("\nDataset after removing exact duplicates:")
print(df_no_duplicates)

# Remove duplicates based on specific columns (keep first occurrence)
df_no_name_age_duplicates = df_imputed.drop_duplicates(subset=['Name', 'Age'], keep='first')
print(f"\nShape after removing Name-Age duplicates: {df_no_name_age_duplicates.shape}")

print("Dataset after removing Name-Age duplicates:")
print(df_no_name_age_duplicates)

# Task 10: Handle data redundancy
print("\n9. HANDLING DATA REDUNDANCY")
print("-" * 32)

# Final cleaned dataset
df_final = df_no_name_age_duplicates.copy()

# Remove any remaining redundancy
df_final = df_final.drop_duplicates().reset_index(drop=True)

print("Final cleaned dataset:")
print(f"Shape: {df_final.shape}")
print("\nFinal dataset:")
print(df_final)

print("\nData cleaning summary:")
print(f"Original dataset size: {df.shape}")
print(f"Final cleaned dataset size: {df_final.shape}")
print(f"Rows removed: {df.shape[0] - df_final.shape[0]}")
print(f"Data retention: {(df_final.shape[0] / df.shape[0]) * 100:.2f}%")

# Task 11: Additional analysis
print("\n10. ADDITIONAL ANALYSIS")
print("-" * 25)

print("Data types:")
print(df_final.dtypes)

print("\nFinal dataset statistics:")
print(df_final.describe())

print("\nValue counts for categorical columns:")
for col in ['Name', 'Department']:
    print(f"\n{col}:")
    print(df_final[col].value_counts())

# Save the cleaned dataset
df_final.to_csv('cleaned_dataset.csv', index=False)
print(f"\nCleaned dataset saved as 'cleaned_dataset.csv'")
print("Data cleaning completed successfully!")

# Optional: Create comparison plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df['Age'].hist(bins=10, alpha=0.7, label='Original')
df_final['Age'].hist(bins=10, alpha=0.7, label='Cleaned')
plt.title('Age Distribution: Before vs After')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 2)
df['Salary'].hist(bins=10, alpha=0.7, label='Original')
df_final['Salary'].hist(bins=10, alpha=0.7, label='Cleaned')
plt.title('Salary Distribution: Before vs After')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 3)
missing_before = df.isnull().sum()
missing_after = df_final.isnull().sum()
columns = df.columns
x = np.arange(len(columns))
width = 0.35

plt.bar(x - width/2, missing_before, width, label='Before Cleaning', alpha=0.7)
plt.bar(x + width/2, missing_after, width, label='After Cleaning', alpha=0.7)
plt.title('Missing Values: Before vs After')
plt.xlabel('Columns')
plt.ylabel('Missing Count')
plt.xticks(x, columns, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

print("\nVisualization plots created showing before/after comparison!")