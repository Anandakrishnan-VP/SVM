
# svm_breast_cancer_full.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =======================
# Step 1: Load dataset
# =======================
# Using sklearn's built-in breast cancer dataset so I don't have to mess with CSV reading
# X = features, y = target (0 = malignant, 1 = benign)
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Train-test split (keeping class ratio same with stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)


# Step 2: PCA for plotting

# The dataset has 30 features → I can’t plot 30D space.
# So using PCA to reduce to 2D, just for visualization purposes.
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d, y, test_size=0.25, stratify=y, random_state=0
)


# Train SVM (2D)

# Trying both linear and RBF kernels on the 2D PCA data to see decision boundaries
kernels = ['linear', 'rbf']
for kernel in kernels:
    # Pipeline ensures scaling happens before SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=1, gamma='scale'))
    ])
    model.fit(X2_train, y2_train)
    

    # Step 4: Plot boundaries

    # Create a meshgrid over the plot space
    h = 0.02  # step size in the grid
    x_min, x_max = X2_train[:, 0].min() - 1, X2_train[:, 0].max() + 1
    y_min, y_max = X2_train[:, 1].min() - 1, X2_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict over the grid to know class regions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X2_train[:, 0], X2_train[:, 1], c=y2_train, edgecolors='k')
    plt.title(f"SVM Decision Boundary ({kernel} kernel, 2D PCA data)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()




# Now using all features (not PCA) with RBF kernel and tuning C & gamma
pipe_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel='rbf'))
])

# Trying a few values for C and gamma
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, 1]
}

# StratifiedKFold keeps class ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid = GridSearchCV(pipe_rbf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best RBF parameters:", grid.best_params_)
print("Best CV accuracy: {:.3f}".format(grid.best_score_))


# Step 6: Cross-validation (Linear SVM)

# Just to compare, running CV on a linear SVM
pipe_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel='linear', C=1))
])
linear_scores = cross_val_score(pipe_linear, X_train, y_train, cv=cv, scoring='accuracy')
print("Linear SVM CV Accuracy: {:.3f} ± {:.3f}".format(linear_scores.mean(), linear_scores.std()))

# Testing the best RBF model on unseen data
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("RBF SVM Test accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=cancer.target_names))
