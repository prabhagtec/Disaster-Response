## Disaster-Response

# Project Description:
This project analyze disaster data from Appen to build a model for an API that classifies disaster messages.

# 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that loads the messages and categories datasets
Merges the two datasets and cleans the data to stores it in a SQLite database

# 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that loads data from the SQLite database
Splits the dataset into training and test sets and builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

# 3. Flask Web App
The web app uses the trained model to input text and return classification results.

![](https://github.com/prabhagtec/Disaster-Response/blob/main/DS.PNG)

Using Plotly in the web app, below visualizations are derived from the data
![](https://github.com/prabhagtec/Disaster-Response/blob/main/Category.PNG)
![](https://github.com/prabhagtec/Disaster-Response/blob/main/Genre.PNG)
# 4. How to run the Python scripts and web app

1. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2.To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage


# 5. Conclusion

From above visualization, data are highly imbalanced  by category which affects model prediction. 
