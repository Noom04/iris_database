from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/iris_database'
db = SQLAlchemy(app)

class IrisData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sepal_length = db.Column(db.Float)
    sepal_width = db.Column(db.Float)
    petal_length = db.Column(db.Float)
    petal_width = db.Column(db.Float)
    species = db.Column(db.String)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Make prediction
        prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        species = iris.target_names[prediction[0]]
        
        # Save data to database
        new_data = IrisData(sepal_length=sepal_length, sepal_width=sepal_width,
                            petal_length=petal_length, petal_width=petal_width,
                            species=species)
        db.session.add(new_data)
        db.session.commit()
        
        return render_template('result.html', species=species)

# Route for viewing saved data
@app.route('/view_data')
def view_data():
    data = IrisData.query.all()
    return render_template('view_data.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
