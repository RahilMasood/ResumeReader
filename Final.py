import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

resumeDataSet = pd.read_csv(r'C:\Users\shez8\Desktop\RAHIL\Mini Projects\Resume Parsing\ResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)  # remove non-ASCII characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(resumeDataSet['cleaned_resume'])
y = resumeDataSet['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=5, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

print(f"Mean Accuracy: {cv_scores.mean():.2f}")

clf.fit(X_train, y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print(f'Training: {train_accuracy:.2f}')
print(f'Testing: {test_accuracy:.2f}')
