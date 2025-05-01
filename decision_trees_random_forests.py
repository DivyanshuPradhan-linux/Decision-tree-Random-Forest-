import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz # You might need to install this and add to your system's PATH

# 1. Load the Dataset
# Replace 'heart_disease_dataset.csv' with the actual path to your dataset file
try:
    df = pd.read_csv('heart_disease_dataset.csv')
except FileNotFoundError:
    print("Error: dataset file not found. Make sure 'heart_disease_dataset.csv' is in the correct directory.")
    exit()

# Assuming the target variable is named 'target' and features are the rest
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict and evaluate (initial)
y_pred_dt = dt_classifier.predict(X_test)
print("Decision Tree Classifier Accuracy (Initial):", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report (Initial):\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree (Requires Graphviz)
# Export the tree to a .dot file
# dot_data = export_graphviz(dt_classifier, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree_initial") # This will save a file like decision_tree_initial.pdf or .png

# 3. Analyze Overfitting and Control Tree Depth
# You can analyze overfitting by comparing training and testing accuracy
y_pred_train_dt = dt_classifier.predict(X_train)
print("Decision Tree Classifier Training Accuracy (Initial):", accuracy_score(y_train, y_pred_train_dt))

# Example of controlling depth to mitigate overfitting
dt_classifier_controlled = DecisionTreeClassifier(max_depth=5, random_state=42) # Experiment with max_depth
dt_classifier_controlled.fit(X_train, y_train)
y_pred_dt_controlled = dt_classifier_controlled.predict(X_test)
print("\nDecision Tree Classifier Accuracy (Controlled Depth):", accuracy_score(y_test, y_pred_dt_controlled))
print("Decision Tree Classification Report (Controlled Depth):\n", classification_report(y_test, y_pred_dt_controlled))

# Visualize the Controlled Depth Decision Tree (Optional)
# dot_data_controlled = export_graphviz(dt_classifier_controlled, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
# graph_controlled = graphviz.Source(dot_data_controlled)
# graph_controlled.render("decision_tree_controlled")

# 4. Train a Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # You can tune n_estimators
rf_classifier.fit(X_train, y_test)

# Predict and evaluate
y_pred_rf = rf_classifier.predict(X_test)
print("\nRandom Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 5. Interpret Feature Importances (from Random Forest)
importances = rf_classifier.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))
sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

print("\nRandom Forest Feature Importances:")
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance:.4f}")

# 6. Evaluate using Cross-Validation
cv_scores_dt = cross_val_score(dt_classifier_controlled, X, y, cv=5) # Using controlled depth DT
print("\nCross-Validation Accuracy Scores (Decision Tree):", cv_scores_dt)
print("Mean Cross-Validation Accuracy (Decision Tree):", cv_scores_scores_dt.mean())

cv_scores_rf = cross_val_score(rf_classifier, X, y, cv=5)
print("\nCross-Validation Accuracy Scores (Random Forest):", cv_scores_rf)
print("Mean Cross-Validation Accuracy (Random Forest):", cv_scores_rf.mean())
