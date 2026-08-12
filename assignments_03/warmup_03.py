import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


iris = load_iris(as_frame=True)
X = iris.data
y = iris.target




#--- Preprocessing Section ----

#- --Preprocessing Q1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
print(f"Q1 X_train array: {X_train.shape}")
print(f"Q1 X_test array: {X_test.shape}")
print(f"Q1 y_test: {y_test.shape}")
print(f"Q1 y_train array: {y_train.shape}")

#---preprocessing Q2
scaler = StandardScaler()
scaler.fit(X_train)                        
X_train_scaled = scaler.transform(X_train) 
X_test_scaled  = scaler.transform(X_test)  

print(f"Q2: Means of Each Column in X_train_scaled:" , X_train_scaled.mean(axis=0))

# We only need to scale the training data to keep the model from seeing the test data and inflating the performance metrics during training,

#---KNN--

#KNN Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


print(f"KNN Q1| Accuracy Score: {knn.score(X_test, y_test)} ")
print("classification report")
print(classification_report(y_test, y_pred))


#KNN Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train )
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("KNN Q2| Accuracy Score: ", knn_scaled.score(X_test_scaled, y_test))

# Scaling the data can improve the performance of KNN because it is a distance-based algorithm. When features are on different scales, the distance calculations can be dominated by features with larger ranges, leading to biased predictions. By scaling this data, we ensure that each feature contributes equally to the distance calculations, which can result in better model performance and more accurate predictions.

#KNN Q3
scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Cross_val_score:\n", scores)
 
for i, score in enumerate(scores, start=1):
    print(f"KNN Q3 Fold {i} Score: {score:.4f}")
print(f"KNN Q3 Mean CV Score: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")

# This test is more trustworthy than a single train-test split because it evaluates the model's performance across multiple subsets of the data, providing a more robust estimate of its generalization ability. It helps to mitigate the risk of overfitting and gives a better understanding of how the model will perform on unseen data.
#KNN Q4
nums = [1, 3, 5, 7, 9, 11, 13, 15]
for k in nums:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    k_scores = cross_val_score(knn_k, X_train, y_train, cv=5)
    print(f"KNN Q4: k={k}: Mean CV Score: {k_scores.mean():.4f}")
# I would choose k=5 for the KNN model because it has the highest mean cross-validation score among the tested values of k. A higher mean CV score indicates better generalization performance on unseen data


#---Classifier Evaluation

#CEQ1 


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

disp.plot()
print(f"Confusion Matrix:\n{cm}")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

#My model did not make any mistakes in classifying the Iris Setosa species, as indicated by the confusion matrix. All instances of Iris Setosa were correctly classified, resulting in a perfect score for that class.

#---The sklearn API: Decision Trees

#DTQ1
DecisionTree = DecisionTreeClassifier(max_depth=3, random_state=42)
DecisionTree.fit(X_train, y_train)
y_pred_dt = DecisionTree.predict(X_test)

print(f"DT Q1| Accuracy Score: {DecisionTree.score(X_test, y_test)} ")
print("DT Q1| classification report\n", classification_report(y_test, y_pred_dt))

#The decision tree model achieved a high accuracy score of 0.9667 on the test set, while the KNN model achieved a slightly lower accuracy score of 0.9667. 
# Scaled  data would not have a significant impact on the decision tree model's performance because decision trees are not sensitive to the scale of the features. Decision trees make splits based on feature values, and the scale of the features does not affect the tree's ability to find optimal splits as it just compares the values of the features to determine the best split.

#---Logistic Regression and Regularization

#LRQ1
for C_val in [0.01, 1.0, 100]:
    model = OneVsRestClassifier(LogisticRegression(C=C_val, max_iter=1000, solver='liblinear'))
    model.fit(X_train_scaled, y_train)
   
    total = sum(np.abs(est.coef_).sum() for est in model.estimators_)
    print(f"LR Q1| C={C_val}: total coefficient magnitude = {total:.4f}")

# As the value C increases, the regularization strength decreases, allowing the model to fit the training data more closely, which can lead to overfitting.

#PCA 
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

#PCAQ1


print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for digit in range(10):
    idx = np.where(digits.target == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(digit)
    axes[digit].axis('off')
plt.savefig("outputs/sample_digits.png")
plt.close()


#PCAQ2

pca = PCA()
pca.fit(X_digits)
scores= pca.transform(X_digits)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.title("PCA 2D projection")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# Yes, same digits are clustered together in the PCA 2D projection plot. This indicates that PCA has effectively captured the underlying structure of the data, allowing similar digits to be grouped together in the reduced dimensional space.
#PCAQ3

cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

#Approximately 15 components are needed to explain 80% of the variance in the digits dataset. 

#PCAQ4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)



n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5, figsize=(8, 8))

for i in range(5):                          # original row
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].axis('off')
axes[0, 0].set_ylabel("Original")

for row, n in enumerate(n_values, start=1):   # one row per n
    for col in range(5):
        axes[row, col].imshow(reconstruct_digit(col, scores, pca, n), cmap='gray_r')
        axes[row, col].axis('off')

plt.savefig("outputs/pca_reconstructions.png")
plt.close()

#It seems that as the number of principal components increases, the quality of the reconstructed images improves. With only 2 components, the reconstructions are quite blurry and lack detail. And as we increase components it exponentially improves the quality of the reconstructed images, capturing more details and features of the original digits. By the time we reach 40 components, the reconstructions are very close to the original images, indicating that a significant amount of information is retained with more components but it plateaus after a certain point.
