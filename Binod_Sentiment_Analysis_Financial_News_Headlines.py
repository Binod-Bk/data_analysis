

# Importing necessary libraries
import pandas as pd

# Loading the dataset with specified encoding
file_path = 'financial_news_headlines_sentiment.csv'  # Adjust the file path if necessary
try:
    data = pd.read_csv(file_path, encoding='latin1')  # Try 'latin1' encoding
    print("Dataset loaded successfully!")
except UnicodeDecodeError:
    print("Trying an alternative encoding...")
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to 'ISO-8859-1'

# Rename columns to ensure consistency
data.columns = ['sentiment', 'headline']  # Rename to meaningful column names

# Display the first few rows of the dataset
print("First few rows of the dataset after renaming:")
print(data.head())



# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display column names
print("\nColumn Names in the Dataset:")
print(data.columns)

# Display the number of rows and columns
print("\nShape of the Dataset:")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Display basic information about each column
print("\nDataset Information:")
print(data.info())

# Display summary statistics of numerical columns
print("\nSummary Statistics (Numerical Columns):")
print(data.describe())

# Checking for unique values in each column
print("\nUnique Values in Each Column:")
for column in data.columns:
    unique_values = data[column].nunique()
    print(f"{column}: {unique_values} unique values")


# ## 4. Clean the data [5 points]


# Step 1: Checking and handling missing values
print("Missing values before cleaning:")
print(data.isnull().sum())

# Dropping rows with missing values (optional)
data = data.dropna()

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Step 2: Removing duplicate rows
print(f"\nNumber of duplicate rows before cleaning: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Number of duplicate rows after cleaning: {data.duplicated().sum()}")

# Step 3: Normalizing text data (assuming text column is named 'headline')
if 'headline' in data.columns:
    import re
    data['headline'] = data['headline'].str.lower()  # Convert to lowercase
    data['headline'] = data['headline'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))  # Remove special characters
    data['headline'] = data['headline'].str.strip()  # Remove leading/trailing whitespace

    print("\nSample normalized headlines:")
    print(data['headline'].head())
else:
    print("\nText column 'headline' not found. Please specify the text column.")

# Step 4: Display the cleaned dataset shape
print("\nShape of the dataset after cleaning:")
print(data.shape)


# ## 5. SMOTE (Imbalanced dataset) [OPTIONAL] BONUS [20 points]
# Hint: Use **imblearn** library

# In[4]:


from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features if needed
X = vectorizer.fit_transform(data['headline'])  # Convert headlines to numerical representation
y = data['sentiment']  # Target column

# Check class distribution before applying SMOTE
print("Class distribution before SMOTE:")
print(Counter(y))

# Step 2: Split the data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after applying SMOTE
print("\nClass distribution after SMOTE:")
print(Counter(y_train_smote))



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Vectorize the text data using Bag of Words
vectorizer = CountVectorizer(max_features=5000)  # Use the most frequent 5000 words
X_bow = vectorizer.fit_transform(data['headline'])  # Convert text to BoW representation
y = data['sentiment']  # Target column

# Step 2: Split the data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# Step 3: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Evaluate accuracy and display classification report
print("Accuracy of the Logistic Regression model with BoW features:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ## 7. Tf-idf model [15 points]

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use the most frequent 5000 words
X_tfidf = tfidf_vectorizer.fit_transform(data['headline'])  # Convert text to TF-IDF representation
y = data['sentiment']  # Target column

# Step 2: Split the data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 3: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Evaluate accuracy and display classification report
print("Accuracy of the Logistic Regression model with TF-IDF features:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ## 8. Split train test data [3 points]

# In[7]:


from sklearn.model_selection import train_test_split

# Assuming 'X' contains features and 'y' contains the target
# Replace `X` with your feature matrix (e.g., X_bow or X_tfidf)
# Replace `y` with your target column (e.g., data['sentiment'])

# Splitting the data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the sizes of the splits
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# ## 9. Classification Algorithm [10 points]
# - Train
# - Predict

# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Train the model
model = LogisticRegression(max_iter=1000, random_state=42)  # You can replace this with another classifier
model.fit(X_train, y_train)

# Step 2: Predict on the test set
y_pred = model.predict(X_test)

# Step 3: Evaluate the model
print("Classification Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ## 10. Another Classification Algorithm [10 points]
# - Train
# - Predict

# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can tune n_estimators
rf_model.fit(X_train, y_train)

# Step 2: Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Step 3: Evaluate the model
print("Classification Accuracy (Random Forest):")
print(accuracy_score(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# ## 11. Confusion Matrixes for two classification algorithms and two feature extractor methods [10 points]

# In[10]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neutral', 'negative', 'positive'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

# Logistic Regression with BoW
print("Logistic Regression with BoW:")
y_pred_lr_bow = model.predict(X_test)  # Already trained on BoW
plot_confusion_matrix(y_test, y_pred_lr_bow, "Logistic Regression (BoW)")

# Random Forest with BoW
print("Random Forest with BoW:")
y_pred_rf_bow = rf_model.predict(X_test)  # Already trained on BoW
plot_confusion_matrix(y_test, y_pred_rf_bow, "Random Forest (BoW)")

# Logistic Regression with TF-IDF
print("Logistic Regression with TF-IDF:")
y_pred_lr_tfidf = model.predict(X_test)  # Replace X_test with TF-IDF if using a separate feature extractor
plot_confusion_matrix(y_test, y_pred_lr_tfidf, "Logistic Regression (TF-IDF)")

# Random Forest with TF-IDF
print("Random Forest with TF-IDF:")
y_pred_rf_tfidf = rf_model.predict(X_test)  # Replace X_test with TF-IDF if using a separate feature extractor
plot_confusion_matrix(y_test, y_pred_rf_tfidf, "Random Forest (TF-IDF)")

