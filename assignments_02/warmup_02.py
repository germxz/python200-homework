#--- scikit-learn api ---
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
#Q1

print("Q1")

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])


model = LinearRegression()
model.fit(years, salary) # (x, y)

year4= model.predict([[4]])[0] # x = 4
year8= model.predict([[8]])[0] # x = 8
print(f"The expected salary for someone working 4 years is ${year4:,.2f}") # --- :,.2ft = thousands and decimal place separator
print(f"The expected salary for someone working 8 years is ${year8:,.2f}")
print(f" The model coefficient is: {model.coef_[0]:,.2f}")
print(f"The model intercept is: {model.intercept_:.2f}")

# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)
x = x.reshape(-1, 1)
print("New shape:", x.shape)
# sklearn expects the data to be 2D in order to prevent ambiguity. It organizes our data so it is all processed correctly when runnning our models and not let it get mistaken as a feature or as a sample.

# Q3
print("Q3")
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure()
plt.scatter(X_clusters[:,0], X_clusters[:,1] ,c=labels)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker="X",s=100, c="black")
plt.title("K-means Clusters")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.savefig(os.path.join(OUT, "kmeans_clusters.png"))
plt.close()

# In the scatter plot, I see three color coded clusters of plotted points with an "X" in the center.

# --- linear regression ---
print("linear regression")


np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

#Q1
print("Q1")
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig(os.path.join(OUT, "cost_vs_age.png"))
plt.close()
# I see in this scatter plot that it is easy to form a boundary between smokers and non smokers. Without looking at a legend, it is safe to assume that the smokers are the ones who pay more for healthcare and the red plotted points prove it.

#Q2 
print("Q2")

X= age.reshape(-1,1)
Y= cost
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"X_train {X_train.shape}")
print(f"X_test {X_test.shape}")
print(f"Y_test {Y_test.shape}")
print(f"Y_train {Y_train.shape}")

#Q3

Question3 = LinearRegression()  
Question3.fit(X_train, Y_train)
Y_pred = Question3.predict(X_test)

print(f"Q3 Slope: {Question3.coef_[0]}") 
print(f"Q3 Intercept: {Question3.intercept_}")
print(f"Q3 RMSE: {np.sqrt(np.mean((Y_pred - Y_test) ** 2))}")
print(f"Q3 R² on the test set: { Question3.score(X_test, Y_test)}")

# Slope is the rate of change in something so when it comes to rising medical costs, you can calculate the slope to forecast out when it will get too expensive to buy. 

# Q4

X_full = np.column_stack([age, smoker])

X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(
    X_full, Y, test_size=0.2, random_state=42
)
Question4 = LinearRegression()
Question4.fit(X_full_train, Y_full_train)
Y_full_pred = Question4.predict(X_full_test)
print(f"Q4 R^2: {Question4.score(X_full_test, Y_full_test)}")

print("age coefficient:    ", Question4.coef_[0])
print("smoker coefficient: ", Question4.coef_[1])

# R^2 jumped from ~0.55 (age only) to ~0.97 with smoker added, so yes, it helps a lot.
# Smoker coefficient: holding age constant, being a smoker adds roughly $15,000
# to predicted annual medical cost.

# Q5

plt.figure()
plt.scatter(Y_full_pred, Y_full_test)

lims = [min(Y_full_pred.min(), Y_full_test.min()),
        max(Y_full_pred.max(), Y_full_test.max())]
plt.plot(lims, lims, "k--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig(os.path.join(OUT, "predicted_vs_actual.png"))
plt.close()

# Above the diagonal: actual cost was higher than predicted, the model underestimated.
# Below the diagonal: actual cost was lower than predicted, the model overestimated.

