#---packages---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

#Pandas review

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

#---Pandas Question 1----
print("Pandas Question 1:")
print(f"First 3 rows:\n{df.head(3)}")
print(df.head(3))
print(f"Shape: {df.shape}")
print(df.shape)
print(f"Data types")
print(df.dtypes)

#---Pandas Question 2----
print ( f"Pandas Question 2: {len(df[(df.grade > 80) & (df.passed == True)])}" )
print(df[(df.grade > 80) & (df.passed == True)])

#---Pandas Question 3----
df['grade_curved'] = df['grade'] + 5
print(f"Pandas Question 3: DataFrame with curved grades:\n{df}")

#---Pandas Question 4----

df['name_upper'] = df['name'].str.upper()
print('Pandas Question 4:')
print(df[["name", "name_upper"]])

#---Pandas Question 5----
print('Pandas Question 5:')
print(df.groupby('city')["grade"].mean())

#---Pandas Question 6---
print("Pandas Question 6:")
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

#----Pandas Question 7---
print("Pandas Question 7:")
df_sorted = df.sort_values(by= "grade", ascending=False)
print(df_sorted.head(3))

#NumPy REVIEW

#---NumPy Question 1----
print("NumPy Question 1")
arr = np.array([10, 20, 30, 40, 50])
print(f"The array shape is {arr.shape}")
print(f"The array dtype is {arr.dtype}")
print(f"The array ndim is {arr.ndim}")

#---NumPy Question 2 ----
print("NumPy Question 2")
arr2= np.array([[1, 2, 3],
                [4, 5, 6], 
                [7, 8, 9]])
print(f"The array shape is {arr2.shape}")
print(f"The array size is {arr2.size}")

#--- NumPy Question 3----
print ("NumPy Question 3: Slicing")
print(arr2[0:2, 0:2])

#--- NumPy Question 4----
print("NumPy Question 4") 
zeros= np.zeros((3, 4))
ones = np.ones((2,5))
print(f"3x4 array: {zeros}")
print(f"2x5 array: {ones}")

#--- NumPy Question 5----
print("NumPy Question 5")
arr3 = np.arange(0,50,5)
print(f"Array: {arr3}")
print(f"Shape: {arr3.shape}")
print(f"Mean: {arr3.mean()}")
print(f"Sum: {arr3.sum()}")
print(f"Standard Deviation: {arr3.std()}")

#--- NumPy Question 6----
print("NumPy Question 6")
arr200= np.random.normal(0,1,200)
print(f"Array of Random Numbers: {arr200}")
print(f"Mean: {arr200.mean()}")
print(f"Standard Deviation: {arr200.std()}")

#Matplotlib Review

#---Matplotlib Question 1----
print("Matplotlib Question 1")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#---Matplotlib Question 2----
print("Matplotlib Question 2")
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.show()

#---Matplotlib Question 3---
print("Matplotlib Question 3")

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='tab:blue', label='series 1')
plt.scatter(x2, y2, color='tab:orange', label='series 2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()    
plt.show()

#---Matplotlib Question 4---
# Matplotlib Q4

print("Matplotlib Question 4")
fig, axes = plt.subplots(1, 2)          # 1 row, 2 columns

axes[0].plot(x, y)                       # left subplot: line
axes[0].set_title("Squares")

axes[1].bar(subjects, scores)            # right subplot: bars
axes[1].set_title("Subject vs Scores")

plt.tight_layout()
plt.show()

#---Descriptive Statistics Review\
    
#---Descriptive Statistics Question 1 ---
print("Descriptive Statistics Review Question 1")    
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print(f"Mean:{np.mean(data)}")
print(f"Median:{np.median(data)}")
print(f"Variance:{np.var(data)}")
print(f"Standard Deviation:{np.std(data)}")

#--- Descriptive Statistics Question 2---
print("Descriptive Stats Question 2")
rnd500= np.random.normal(65,10,500)
plt.hist(rnd500, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

#--- Descriptive Statistics Question 3---

print("Descriptive Stats Question 3")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"]) #Tick_labels bcs labels is now outdated
plt.title("Score Comparison")
plt.show()

#--- Descriptive Statistics Question 4---
print("Descriptive Stats Question 4")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].boxplot(normal_data)
axes[0].set_title("Normal Distribution")
axes[0].set_xlabel("Distribution")
axes[0].set_ylabel("Value")

axes[1].boxplot(skewed_data)
axes[1].set_title("Exponential Distribution")
axes[1].set_xlabel("Distribution")
axes[1].set_ylabel("Value")

plt.tight_layout()
plt.show()
# It seems like the exponential distribution is more skewed because of the long tail of the data, while the normal distribution is more symmetric and centered around the mean.

#--- Descriptive Statistics Question 5---
print("Descriptive Stats Question 5")

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Mean of data1: {np.mean(data1)}")
print(f"Mean of data2: {np.mean(data2)}")
print(f"Median of data1: {np.median(data1)}")
print(f"Median of data2: {np.median(data2)}")
print(f"Mode of data1: {pd.Series(data1).mode()[0]}")
print(f"Mode of data2: {pd.Series(data2).mode()[0]}")

# The mean is different from the median because the mean will be sensitive to outliers while the median is more robust. In data2, the outlier value of 150 skews the mean upwards, while the median remains unaffected. 

#Hypothesis Testing Review

#---Hypothesis Question 1---

print("Hypothesis Testing Question 1")

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)  #ind =independent samples t-test
print(f"T-stat and P-value: t-statistic = {t_stat}, p-value = {p_value}")

#---Hypothesis Question 2---
print("Hypothesis Testing Question 2")
alpha = 0.05
if p_value < alpha:
    print(f"P-value ({p_value:.4f}) is less than alpha ({alpha}), so the result is statistically significant.")
else:
    print(f"P-value ({p_value:.4f}) is greater than alpha ({alpha}), so the result is not statistically significant.")

#---Hypothesis Question 3---
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]


t_stat, p_value = stats.ttest_rel(before, after)  #rel = paired samples t-test
print("Hypothesis Testing Question 3")
print(f"T-stat and P-value: t-statistic = {t_stat}, p-value = {p_value}")
#---Hypothesis Question 4---
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_value = stats.ttest_1samp(scores, 70)  #1samp = one-sample t-test
print("Hypothesis Testing Question 4")
print(f"T-stat and P-value: t-statistic = {t_stat}, p-value = {p_value}")

#---Hypothesis Question 5---
t_stat, p_value= stats.ttest_ind(group_a, group_b, alternative="less")  
print(f"Hypothesis Testing Question 5 T-stat and P-value: t-statistic = {t_stat}, p-value = {p_value}")
#---Hypothesis Question 6---
print("Hypothesis Testing Question 6")
t_stat_ab, p_value_ab = stats.ttest_ind(group_a, group_b)
diff = np.mean(group_b) - np.mean(group_a)
print(f"Group B scored about {diff:.1f} points higher than Group A on average (t={t_stat_ab:.2f}, p={p_value_ab:.4f}). Since p < 0.05, this difference is unlikely to be due to chance.")
#Correlation Review

#---Correlation Question 1---

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

correlation = np.corrcoef(x, y)
print(correlation)
print(f"Correlation Question 1: {correlation[0, 1]}")
# The correlation coefficient is 1, indicating a perfect positive linear relationship between x and y. As x increases, y increases proportionally.

#---Correlation Question 2---


x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

correlation, p_value = pearsonr(x, y)
print(f"Correlation Question 2: Correlation = {correlation},")
print(f"Correlation Question 2: P-value = {p_value}")

#---Correlation Question 3---

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(f"Correlation Question 3:")
print(df.corr())

#---Correlation Question 4---

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#---Correlation Question 5---
heat = sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
heat.set_title("Correlation Heatmap")
plt.show()
#Pipelines

#---Pipeline Question 1---

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

print("Pipeline Question 1")
results = data_pipeline(arr)
for key, value in results.items():
    print(f"{key}: {value}")