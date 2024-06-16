# diabetes_test_case
**Bold**
## Purpose of this repo
This is a repository to test and learn running API's on an AWS EC2 instace. the following will be done here:
>1. A small network will be trained on diabetes data from scikit-learn
>2. The network will be trained and saved to be used in the EC2 instace
>3. An EC2 instace is setup with python requiremetns installed.
>4. fastapi is setup to deploy

---
**Bold**
## Regression problem

the Diabetes dataset from Scikit-learn is used for the regression network. this database consists of 10 features and one target that we are trying to predict. in total 442 records. the last 20 of which will be assigned to a test dataset to test the accuracy of the network.
'''
    sklearn.datasets.load_diabetes(*, return_X_y=False, as_frame=False, scaled=True)
'''
set as_frame to true to return a pandas dataframe. also set return_X_y to true to get x and y datasets

---