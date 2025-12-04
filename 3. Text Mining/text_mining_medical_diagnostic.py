# 0. Context
print(
    "******************************************************************\n"
    "0. Context\n"
    "------------------------------------------------------------------\n"
    "Using the dataset of medical reviews where each review is labeled according to the described medical diagnosis: mild, moderate, or severe. \
    I want to classify these reviews based on the severity of the diagnosis using different classification models. \
    Dataset link: https://www.kaggle.com/code/ohseokkim/fake-news-easy-nlp-text-classification/input?select=fake_or_real_news.csv\n"
    "******************************************************************\n"
)

# 1. Environment Setup
print(
    "******************************************************************\n"
    "1. Environment Setup\n"
    "------------------------------------------------------------------\n"
)
# 1.1. Load requered libraries
import os 

import zipfile
import pandas as pd
import numpy as np
from IPython.display import display

import string
import matplotlib.pyplot as plt
import seaborn as sn

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer        #note 1
from sklearn.model_selection import train_test_split                                #note 2                       
from sklearn.tree import DecisionTreeClassifier                                     #note 3
from sklearn.neighbors import KNeighborsClassifier                                  #note 4
from sklearn.ensemble import RandomForestClassifier                                 #note 5

from sklearn.metrics import roc_curve, roc_auc_score,auc                            #note 6
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1.2. Set working directory
HOME = os.getcwd()
print("Current working directory:", HOME)

print("******************************************************************\n")

# 1.3. Load dataset 
print(
    "------------------------------------------------------------------\n"
    "1.3. Load dataset\n"
    "------------------------------------------------------------------\n"
)

# * Enzip the dataset
zipfile_path = os.path.join(HOME, "data", "raw", "fake_or_real_news.csv.zip")
destination_path = os.path.join(HOME, "data", "raw")

with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
    zip_ref.extractall(destination_path)
    print(f"Dataset extracted to: {destination_path}")

# * Load the dataset into a pandas DataFrame
data_path = os.path.join(HOME, "data", "raw", "fake_or_real_news.csv")
df = pd.read_csv(data_path)
np.random.seed(0)
display(df.sample(10))

# * Mapping: conversion of 'label' modalities to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# 2. Data Preprocessing
print(
    "******************************************************************\n"
    "2. Data Preprocessing\n")

# 2.1. Data cleaning : Dropping stopwords + special character + lemmatization
print(
    "------------------------------------------------------------------\n"
    "2.1. Data cleaning : Dropping stopwords + special character + lemmatization\n"
    "------------------------------------------------------------------\n"
)
# * Define pre-processing function
def pre_process_text(text):
    # Droping stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Droping stopwords and ponctuation  
    tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    
    # Lemmatisation
    Lemmatizer = WordNetLemmatizer()
    
    return ' '.join(tokens)

# * Quick check
df['cleaned_text'] = df['text'].apply(pre_process_text)
display(df.head(5))

# 2.2. Text to Numeric Conversion
print(
    "------------------------------------------------------------------\n"
    "2.2. Text to Numeric Conversion\n"
    "------------------------------------------------------------------\n"
)
"""Output: I will have the occurrence name of each token and their frequencies."""

# * Vectorization
vectorizer = CountVectorizer()

# * TD-IDF
tfidf_vectorizer = TfidfVectorizer()
x_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Quick check of the matrix
frequency_matrix = x_tfidf.toarray()
print(f"View of the matrix: \n{frequency_matrix}")

# 3. Model Training and Evaluation
print(
    "******************************************************************\n"
    "3. Model Training and Evaluation\n"
)
"""Note: Here are models to run (Decision Tree, KNN, Random Forest)"""

# 3.1. Train-Test Split
print(
    "------------------------------------------------------------------\n"
    "3.1. Train-Test Split\n"
    "------------------------------------------------------------------\n"
)
X_train, X_test, y_train, y_test = train_test_split(x_tfidf, df['label'], test_size=0.2)
print(
    f"|X_train shape: {X_train.shape}, | X_test shape: {X_test.shape}|\n"
    f"|y_train shape: {y_train.shape}, | y_test shape: {y_test.shape}|\n"
)

# 3.2. Model training
print(
    "------------------------------------------------------------------\n"
    "3.2. Model training\n"
    "------------------------------------------------------------------\n")

#  * Model 1: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# * Model 2: K-Nearest Neighbors (KNN) Classifier
knn_model = KNeighborsClassifier(n_neighbors=12) 
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred_knn)

# * Model 3: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=120, random_state=42) 
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# 3.3. Model evaluation
print(
    "------------------------------------------------------------------\n"
    "3.3. Model evaluation\n"
    "------------------------------------------------------------------\n")

print(f"Decision Tree Classification Report:\n{classification_report(y_test, y_pred_dt)}\n")
print(f"K-Nearest Neighbors Classification Report:\n{classification_report(y_test, y_pred_knn)}\n")
print(f"Random Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}")

# 4. Model prediction
print(
    "******************************************************************\n"
    "4. Model prediction\n"
    "------------------------------------------------------------------\n"
)

# * Model 1: Decision Tree Classifier
# ** Prediction
pred_y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# ** ROC and AUC Curves
fpr_dt, tpr_dt, _ = roc_curve(y_test, pred_y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
auc_scor_dt = roc_auc_score(y_test,pred_y_proba_dt)

# * Model 2: K-Nearest Neighbors
# ** Prediction
pred_y_proba_knn = knn_model.predict_proba(X_test)[:, 1]

## Curves
fpr_knn, tpr_knn, _ = roc_curve(y_test, pred_y_proba_knn)
roc_auc_knn =auc(fpr_knn, tpr_knn)
auc_scor_knn = roc_auc_score(y_test,pred_y_proba_knn)

# * Model 3: Random Forest Classifier
# ** Prediction
pred_y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

## Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
auc_scor_rf = roc_auc_score(y_test,pred_y_proba_rf)

# * Print AUC Scores
print(
    "------------------------------------------------------------------\n"
    "AUC Scores\n"
    "------------------------------------------------------------------\n"
    f"AUC Score_dt: {roc_auc_dt: .2f}\n"
    f"AUC Score_knn: {roc_auc_knn: .2f}\n"
    f"AUC Score_rf: {roc_auc_rf: .2f}\n"
)

# 5. Display our ROC curves
plt.figure(figsize=(10, 8))

plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# * Add details to the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Best model based on AUC score
print(
    "------------------------------------------------------------------\n"
    "Best model based on AUC score, F1-recall, precision: Random Forest\n"
    "------------------------------------------------------------------\n"
)

# 6. Confusion matrix Analysis
# * Confusion matrice of each model 
conf_matrice_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrice_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrice_rf = confusion_matrix(y_test, y_pred_rf)

conf_matrices = {
    "Decision Tree": conf_matrice_dt,
    "K-Nearest Neighbors": conf_matrice_knn,
    "Random Forest": conf_matrice_rf
}

for model_name, conf_matrice in conf_matrices.items():
    print(f"{model_name}:\n{conf_matrice}\n")
    
# End of script