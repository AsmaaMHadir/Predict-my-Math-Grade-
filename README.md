# Outline

1 - Application
2 - Dataset Trained on
3 - Model 

## 1) Application
Machine learning based web app that predicts the Math exam score for a student based on their input feature values. You may access the web app here.
Snippet of the web interface:

<img width="632" alt="webs" src="https://github.com/AsmaaMHadir/Students-Performance-Prediction/assets/46932156/ac2b70d0-2984-4a58-9377-3edd57a662a5">
<img width="536" alt="pred" src="https://github.com/AsmaaMHadir/Students-Performance-Prediction/assets/46932156/f68fef5e-7c60-42e2-8ae6-a072a13362e9">
- Stack: the Model's API was built with **Flask**. 

## 2) Dataset Trained on

The dataset `stud.csv` contains 1001 rows and 8 columns representing students records of their:

- Gender: female or male
- Race/ethnicity: denoted by groups
- Parental level of education: bachelor's degree, associate degree, some college, high school, master's degree
- Lunch: wether the student received standard or free/reduced lunch
- Test preparation course: wether the student completed a test prep course or not
- Math score: numeric value of the student's grade in Math ( the feature to be predicted)
- Reading score: numeric value of the student's grade in reading
- Writing score: numeric value of the student's grade in writing

## 3) Model

- The model trainer runs the following set of algorithms from the `sklearn` library on the preprocessed features to observe which performs the best:
    - Random forest
    - Decision Tree
    - Gradient Boosting
    - Linear Regression
    - K-Neighbors
    - XGBRegressor
    - AdaBoost Regressor
  
Then pickles the best model which in this case is acheived by the `Ridge` linear model.
