# Credit Card Risk Analyzer

Predicting the probability with which customers default on their bills.

<!--more-->

In this project, I leveraged the power of Logistic Regression to predict if a customer’s credit card application will be approved. Most banks have systems like this, albeit much more complex, to make a prediction on the application’s approval. The risk associated with lending credit is immense and hence, banks rely heavily on such systems to mitigate it.

## 1. Exploratory Data Analysis
I looked through the dataset to understand the various kinds of features present and their data types and statistics. There were columsn with missing and non-numeric type data. For the logistic regression model to perform well, the dataset needed some cleaning and pre-processing.

## 2. Data Cleaning
There were columns with ‘?’ in place of correct data. To address this, I replaced the missing values with ‘NaN’. Next, the ‘NaN’s had to be treated. There are various ways to deal with such values. I chose to impute these values with the mean of the column. Lastly, to fill in the missing values in the columns with non-numeric data types, I replaced it with the most frequent observation in the column for that predictor.

## 3. Data Pre-Processing
Firstly, the non-numeric data was encoded using LabelEncoding. Next, the data was scaled to bring it down to the [0-1] range. Normalizing the data improves model performance and it is computationally less expensive and since the data points are in the same range, each feature is weighted the same. Finally, the data was split into the training and testing dataset.

### Code
```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit(X_train).transform(X_train)
    rescaledX_test = scaler.fit(X_test).transform(X_test)
```

## 4. Training and Testing
Since, this is a binary classification problem, I chose to use the Logistic Regression algorithm for the prediction. The accuracy of the vanilla logistic regression model (without any tuning) was 83.7%

### Code
```python
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    y_pred = logreg.predict(rescaledX_test)
    print("Accuracy of logistic regression classifier: ", accuracy_score(y_test, y_pred))
    confusion_matrix(y_test, y_pred)
    Accuracy of logistic regression classifier:  0.8377192982456141
    array([[93, 10],
        [27, 98]])
```

## 5. Hyperparameter Tuning
To improve the accuracy of the model, I used GridSearchCV with cross-validation to find the optimum parameters for the model. Cross-validation works by splitting the data into k folds and fitting the data to the model. (k-1) sets are used for training and the last set is used for validation. 

### Code
```python
     grid_model = GridSearchCV(logreg, param_grid=param_grid, cv=5)
    rescaledX = scaler.fit(X_train).transform(X_train)
    grid_model_result = grid_model.fit(rescaledX, y_train)

    # Summarize results
    best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
    print("Best: %f using %s" % (best_score, best_params))
    Best: 0.863651 using {'max_iter': 100, 'tol': 0.01}
```

## 6. Results
With an accuracy of 86.3%, this model can be used as a tool to identify customers who could default on their Credit Card bills.
