# Import necessary libraries
import matplotlib
matplotlib.use('Agg')
import nltk
nltk.data.path.append("C:\\Users\\abhiv\\AppData\\Roaming\\nltk_data")
import flask
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
from flask import request, jsonify, render_template
from textblob import TextBlob
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")


app = flask.Flask(__name__)
from waitress import serve

# Load the datasets
path = 'E:/dissertation/GPT_Dataset/'
train_data = pd.read_csv(path + 'medium-345M.test.csv').head(3000)
test_data = pd.read_csv(path + 'medium-345M-k40.test.csv').head(3000)
small_train_data = pd.read_csv(path + 'small-117M.test.csv').head(3000)
small_test_data = pd.read_csv(path + 'small-117M-k40.test.csv').head(3000)
xl_train_data = pd.read_csv(path + 'xl-1542M.test.csv').head(3000)
xl_test_data = pd.read_csv(path + 'xl-1542M-k40.test.csv').head(3000)
large_train_data = pd.read_csv(path + 'large-762M.test.csv').head(3000)
large_test_data = pd.read_csv(path + 'large-762M-k40.test.csv').head(3000)
medium_valid_data = pd.read_csv(path + 'medium-345M.valid.csv').head(3000)
small_valid_data = pd.read_csv(path + 'small-117M.valid.csv').head(3000)

# Concatenate the datasets
train_data = pd.concat([train_data, small_train_data, xl_train_data, large_train_data, medium_valid_data, small_valid_data])
test_data = pd.concat([test_data, small_test_data, xl_test_data, large_test_data])

# Remove NA values and preprocess
train_data = train_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['text'])

# Function to get part-of-speech tags
def pos_tagging(text):
    return " ".join([pos for word, pos in TextBlob(text).tags])

# Function to get sentiment score
def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Apply preprocessing, including POS tagging and sentiment scores
train_data['pos_tags'] = train_data['text'].apply(pos_tagging)
test_data['pos_tags'] = test_data['text'].apply(pos_tagging)
train_data['sentiment'] = train_data['text'].apply(sentiment_score)
test_data['sentiment'] = test_data['text'].apply(sentiment_score)

# Use TF-IDF with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_data['text'])
X_test_tfidf = vectorizer.transform(test_data['text'])

# Vectorize POS tags
pos_vectorizer = TfidfVectorizer()
X_train_pos = pos_vectorizer.fit_transform(train_data['pos_tags'])
X_test_pos = pos_vectorizer.transform(test_data['pos_tags'])

# Combine features
X_train_combined = hstack([X_train_tfidf, X_train_pos, train_data['sentiment'].values.reshape(-1, 1)])
X_test_combined = hstack([X_test_tfidf, X_test_pos, test_data['sentiment'].values.reshape(-1, 1)])


# Split data
y_train = train_data['ended']
y_test = test_data['ended']

X_train, X_val, y_train, y_val = train_test_split(X_train_combined, y_train, test_size=0.2, random_state=42)


# Apply SMOTE for handling imbalanced dataset
print("Class distribution before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", pd.Series(y_train_resampled).value_counts())

# Function to plot class distribution before and after applying SMOTE
def plot_class_distribution(y_before, y_after):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    y_before.value_counts().plot(kind='bar', ax=ax[0])
    ax[0].set_title("Class Distribution Before SMOTE")
    y_after.value_counts().plot(kind='bar', ax=ax[1])
    ax[1].set_title("Class Distribution After SMOTE")
    plt.tight_layout()
    plt.savefig('static/class_distribution.png')
    plt.close()



plot_class_distribution(y_train, y_train_resampled)

# Define a parameter grid for Naive Bayes with different binarize values
param_grid = {'binarize': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]}
nb_clf = BernoulliNB()
grid_search = GridSearchCV(nb_clf, param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best binarize value
best_binarize = grid_search.best_params_['binarize']
print(f"Best binarize value: {best_binarize}")

# Train Naive Bayes with the best binarize value
nb_clf = BernoulliNB(binarize=best_binarize)
nb_clf.fit(X_train_resampled, y_train_resampled)

# Calibrate Naive Bayes
calibrated_nb = CalibratedClassifierCV(nb_clf, method='sigmoid', cv=5)
calibrated_nb.fit(X_train_resampled, y_train_resampled)

# Predictions and Metrics for Naive Bayes (Calibrated)
nb_val_predictions = calibrated_nb.predict(X_val)
print("Calibrated Naive Bayes Metrics on Validation Data:")
print(classification_report(y_val, nb_val_predictions))

nb_test_predictions = calibrated_nb.predict(X_test_combined)
print("\nCalibrated Naive Bayes Metrics on Test Data:")
print(classification_report(y_test, nb_test_predictions))

# Training XGBoost Classifier
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train_resampled, y_train_resampled)

# Predictions and Metrics for XGBoost
xgb_val_predictions = xgb_clf.predict(X_val)
print("\nXGBoost Metrics on Validation Data:")
print(classification_report(y_val, xgb_val_predictions))

xgb_test_predictions = xgb_clf.predict(X_test_combined)
print("\nXGBoost Metrics on Test Data:")
print(classification_report(y_test, xgb_test_predictions))

# Training SVM Classifier
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train_resampled, y_train_resampled)

# Predictions and Metrics for SVM
svm_val_predictions = svm_clf.predict(X_val)
print("\nSVM Metrics on Validation Data:")
print(classification_report(y_val, svm_val_predictions))

svm_test_predictions = svm_clf.predict(X_test_combined)
print("\nSVM Metrics on Test Data:")
print(classification_report(y_test, svm_test_predictions))


# Function to plot the classification report as a heatmap
def plot_classification_report(report, title):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    sns.heatmap(df[['precision', 'recall', 'f1_score']].set_index(df['class']), annot=True)
    plt.title(title)
    plt.savefig(f'static/{title}.png')
    plt.close()


report = classification_report(y_val, nb_val_predictions)
plot_classification_report(report, "Naive Bayes Metrics on Validation Data")

report = classification_report(y_test, nb_test_predictions)
plot_classification_report(report, "Naive Bayes Metrics on Test Data")

report = classification_report(y_val, xgb_val_predictions)
plot_classification_report(report, "XGBoost Metrics on Validation Data")

report = classification_report(y_test, xgb_test_predictions)
plot_classification_report(report, "XGBoost Metrics on Test Data")

report = classification_report(y_val, svm_val_predictions)
plot_classification_report(report, "SVM Metrics on Validation Data")

report = classification_report(y_test, svm_test_predictions)
plot_classification_report(report, "SVM Metrics on Test Data")


def plot_feature_importance(model, title):
    feature_names = vectorizer.get_feature_names_out().tolist() + pos_vectorizer.get_feature_names_out().tolist() + [
        'sentiment']
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[-10:]  # We will show top 10 features

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.savefig("static/feature_importance.png", bbox_inches='tight')
    plt.close()


plot_feature_importance(xgb_clf, "XGBoost Feature Importance")

def plot_roc_auc(y_true, y_score, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'static/{title}.png')
    plt.close()

# Call plot_roc_auc for Naive Bayes and XGBoost
nb_val_probabilities = calibrated_nb.predict_proba(X_val)[:, 1]
xgb_val_probabilities = xgb_clf.predict_proba(X_val)[:, 1]
nb_test_probabilities = calibrated_nb.predict_proba(X_test_combined)[:, 1]
xgb_test_probabilities = xgb_clf.predict_proba(X_test_combined)[:, 1]
svm_val_probabilities = svm_clf.predict_proba(X_val)[:, 1]
svm_test_probabilities = svm_clf.predict_proba(X_test_combined)[:, 1]

plot_roc_auc(y_val, nb_val_probabilities, 'Naive Bayes ROC on Validation Data')
plot_roc_auc(y_val, xgb_val_probabilities, 'XGBoost ROC on Validation Data')
plot_roc_auc(y_test, nb_test_probabilities, 'Naive Bayes ROC on Test Data')
plot_roc_auc(y_test, xgb_test_probabilities, 'XGBoost ROC on Test Data')
plot_roc_auc(y_val, svm_val_probabilities, 'SVM ROC on Validation Data')
plot_roc_auc(y_test, svm_test_probabilities, 'SVM ROC on Test Data')


def predict_label(text):
    pos_tags = pos_tagging(text)
    sentiment = sentiment_score(text)
    X_test_tfidf_single = vectorizer.transform([text])
    X_test_pos_single = pos_vectorizer.transform([pos_tags])
    X_test_combined_single = hstack([X_test_tfidf_single, X_test_pos_single, [[sentiment]]])

    nb_prediction = nb_clf.predict(X_test_combined_single)[0]
    xgb_prediction = xgb_clf.predict(X_test_combined_single)[0]
    svm_prediction = svm_clf.predict(X_test_combined_single)[0]

    nb_prediction_label = "Human" if nb_prediction == 1 else "Machine"
    xgb_prediction_label = "Human" if xgb_prediction == 1 else "Machine"
    svm_prediction_label = "Human" if svm_prediction == 1 else "Machine"

    return nb_prediction_label, xgb_prediction_label, svm_prediction_label


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    nb_prediction, xgb_prediction, svm_prediction = predict_label(text)
    return jsonify({'nb_prediction': nb_prediction, 'xgb_prediction': xgb_prediction, 'svm_prediction': svm_prediction})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        nb_prediction, xgb_prediction, svm_prediction = predict_label(input_text)
        return render_template('index.html', input_text=input_text, prediction={'nb_prediction': nb_prediction, 'xgb_prediction': xgb_prediction, 'svm_prediction': svm_prediction})
    return render_template('index.html')


if __name__ == '__main__':
    host = 'localhost'
    port = 5000
    print(f'Serving on http://{host}:{port}/')
    serve(app, host=host, port=port, threads=10)