import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import re

# Load Dataset
resumeDataSet = pd.read_csv(r'C:\Users\shez8\Desktop\RAHIL\Mini Projects\Resume Parsing\ResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''

# Function to clean the resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)  # remove non-ASCII characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Clean the resume column
resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

# Label Encoding for categories
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(resumeDataSet['cleaned_resume'])

# Prepare Target Variable
y = resumeDataSet['Category']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier with increased regularization
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=5, random_state=42)

# Cross-validation (k-fold) to reduce overfitting and get more reliable performance estimation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)  # 5-fold cross-validation

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.2f}")
print(f"Standard deviation: {cv_scores.std():.2f}")

# Fit the model
clf.fit(X_train, y_train)

# Evaluate the model
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
