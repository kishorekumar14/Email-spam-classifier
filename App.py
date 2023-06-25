from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import pymongo
    client = pymongo.MongoClient('mongodb://localhost:27017')
    database = client['email_spam']
    db = database.get_collection("spam")
    df = pd.DataFrame(list(db.find()))
        # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        client = pymongo.MongoClient('mongodb://localhost:27017')
        database = client['email_spam']
        if(my_prediction == 0):
            prediction = 'ham'
        else:
            prediction='spam'
        df=database['In_out']
        
        df.insert_one({'class':prediction,'message': data})
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()