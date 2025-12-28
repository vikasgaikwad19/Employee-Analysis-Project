import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==================================================
# 1️⃣ CREATE DATA & SAVE CSV AUTOMATICALLY
# ==================================================

# Create folder "data" if not exists
os.makedirs("data", exist_ok=True)

# Sample dataset
data = {
    "Name": ["John", "Emma", "David", "Sophia", "Liam", "Olivia", "Noah", "Ava", "Mason", "Mia"],
    "Age": [28, 32, 45, 26, 38, 29, 41, 30, 36, 27],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "Department": ["IT", "HR", "Finance", "IT", "Marketing", "HR", "Finance", "IT", "Marketing", "Finance"],
    "YearsExperience": [3, 6, 15, 2, 10, 5, 18, 4, 12, 7],
    "Salary": [50000, 62000, 95000, 48000, 78000, 59000, 102000, 54000, 85000, 68000]
}

df = pd.DataFrame(data)

# Save CSV
csv_path = "data/employees.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file created successfully at: {csv_path}")

# ==================================================
# 2️⃣ LOAD THE DATA FOR ANALYSIS
# ==================================================
df = pd.read_csv(csv_path)

# View first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Shape and column names
print("\nDataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Handle missing values (if any)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
print("After Removing Duplicates:", df.shape)

# Data types
print("\nData Types Before Conversion:")
print(df.dtypes)

df['YearsExperience'] = df['YearsExperience'].astype(int)
df['Salary'] = df['Salary'].astype(float)

print("\nData Types After Conversion:")
print(df.dtypes)

# Summary stats
print(df.describe())

print("Mean Salary:", df['Salary'].mean())
print("Median Salary:", df['Salary'].median())
print("Mode Salary:", df['Salary'].mode()[0])

# ==================================================
# 3️⃣ VISUALIZATIONS
# ==================================================

sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

sns.countplot(x='Gender', data=df)
plt.title("Gender Count")
plt.show()

sns.countplot(x='Department', data=df)
plt.title("Department Count")
plt.show()

sns.scatterplot(x='Age', y='Salary', hue='Gender', data=df)
plt.title("Age vs Salary")
plt.show()

sns.boxplot(x='Department', y='Salary', data=df)
plt.title("Salary by Department")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Detect outliers
sns.boxplot(x=df['Salary'])
plt.title("Salary Outlier Detection")
plt.show()

# Remove outliers using IQR
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Salary'] >= Q1 - 1.5 * IQR) & (df['Salary'] <= Q3 + 1.5 * IQR)]
print("\nAfter removing outliers, Dataset Shape:", df.shape)

sns.barplot(x='Department', y='Salary', data=df)
plt.title("Average Salary by Department")
plt.show()

sns.lineplot(x='YearsExperience', y='Salary', data=df, marker='o')
plt.title("Experience vs Salary")
plt.show()

print("\nAverage Salary by Department:\n", df.groupby('Department')['Salary'].mean())
print("\nAverage Salary by Gender:\n", df.groupby('Gender')['Salary'].mean())