import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import ast

# Helper function to extract genres
def extract_genres(genres_data):
    try:
        genres = ast.literal_eval(genres_data)
        return [genre['name'] for genre in genres]
    except (ValueError, SyntaxError):
        return []

# Load the dataset
movies_df = pd.read_csv('D:\\ML\\Kaggle_DS\\tmdb_5000_movies.csv')

# Drop rows with critical missing data
movies_df = movies_df.dropna(subset=['release_date', 'overview'])

# Extract genres and convert to strings
movies_df['genre_list'] = movies_df['genres'].apply(extract_genres)
movies_df['genre_list_str'] = movies_df['genre_list'].apply(lambda x: ','.join(x) if x else 'Unknown')

# Define numeric columns
numeric_columns = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']

# Split into features and target
X_numeric = movies_df[numeric_columns].copy()
X_genres = movies_df[['genre_list_str']].copy()
y = (movies_df['revenue'] > 100_000_000).astype(int)

# Split the data
X_numeric_train, X_numeric_test, X_genres_train, X_genres_test, y_train, y_test = train_test_split(
    X_numeric, X_genres, y, test_size=0.25, random_state=42
)

# Handle missing values in numeric features
imputer = SimpleImputer(strategy='mean')
X_numeric_train_imputed = imputer.fit_transform(X_numeric_train)
X_numeric_test_imputed = imputer.transform(X_numeric_test)

# Scale numeric features
scaler = StandardScaler()
X_numeric_train_scaled = scaler.fit_transform(X_numeric_train_imputed)
X_numeric_test_scaled = scaler.transform(X_numeric_test_imputed)

# One-hot encode genres
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_genres_train_encoded = encoder.fit_transform(X_genres_train)
X_genres_test_encoded = encoder.transform(X_genres_test)

# Combine numeric and genre features
X_train = np.hstack([X_numeric_train_scaled, X_genres_train_encoded])
X_test = np.hstack([X_numeric_test_scaled, X_genres_test_encoded])

# Get feature names for reference
numeric_features = numeric_columns
genre_features = encoder.get_feature_names_out(['genre_list_str'])
all_features = np.concatenate([numeric_features, genre_features])

# Train SVM model
print("Training linear SVM classifier...")
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions and evaluation for SVM
y_pred_svm = svm_classifier.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))


print("Training rbf SVM classifier...")
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions and evaluation for SVM
y_pred_svm = svm_classifier.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))


print("Training polynomial SVM classifier...")
svm_classifier = SVC(kernel='poly',degree=3, random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions and evaluation for SVM
y_pred_svm = svm_classifier.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))


print("Training sigmoid SVM classifier...")
svm_classifier = SVC(kernel='sigmoid', coef0=1.0,random_state=42)
svm_classifier.fit(X_train, y_train)

# Predictions and evaluation for SVM
y_pred_svm = svm_classifier.predict(X_test)
print("\nSVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Train Naive Bayes model
print("\nTraining Naive Bayes classifier...")
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predictions and evaluation for Naive Bayes
y_pred_nb = nb_classifier.predict(X_test)
print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))