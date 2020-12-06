import pandas as pd
import numpy as np 
import streamlit as st
from sklearn.model_selection import train_test_split

st.title('Predicting Customer Category | Recommendation Engine')
st.header('Hackerearth Challenge')

@st.cache(allow_output_mutation=True)
def load_train_data():
    train_data = pd.read_csv("train.csv")
    return train_data

@st.cache(allow_output_mutation=True)
def load_test_data():
    test_data = pd.read_csv('test.csv')
    return test_data


train_data = load_train_data()
test_data  = load_test_data()
y = train_data.iloc[:,11].values
st.text('Training Data')
st.write(train_data)

if st.button('View test data'):
    st.text('Testing Data')
    st.write(test_data)

y = train_data.iloc[:,-1]
st.write(y)
train_data = train_data.drop(['customer_category', 'customer_id'],axis = 1)
train_data = train_data.fillna(train_data.mean()) 
train_data['customer_active_segment'] = train_data['customer_active_segment'].fillna('B')
train_data['customer_active_segment'] = train_data['X1'].fillna('BA')

train_data = pd.get_dummies(train_data, columns=["X1","customer_active_segment"],drop_first = True) 
st.write(train_data)

X = train_data.iloc[:,:]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)


st.write('Splitting the data set into train and validation set to check the accuracy.')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_valid)
st.subheader('Predicted Value using Gradient Boosting Classifier')
st.write(y_pred_gb)

st.subheader('Confusion Matrix Gradient Boosting Classifier')
st.write(confusion_matrix(y_valid, y_pred_gb))

st.subheader('Accuracy using Gradient Boosting Classifier')
st.write(accuracy_score(y_valid, y_pred_gb))

from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
random.fit(X_train, y_train)


y_pred_randomF = random.predict(X_valid)
st.subheader('Predicted Value using Random Forest Classifier')
st.write(y_pred_randomF)
st.subheader('Confusion Matrix using Random Forest Classifier')
st.write(confusion_matrix(y_valid, y_pred_randomF))

st.subheader('Accuracy using Random Forest Classifier')
st.write(accuracy_score(y_valid, y_pred_randomF))


st.header('Importing the test dataset and then predicting values based on the above two algorithms')
st.subheader('All the preprocessing of the data set is done similar to training dataset')


test_data = test_data.drop(['customer_id'],axis = 1)
test_data = test_data.fillna(train_data.mean()) 
test_data['customer_active_segment'] = test_data['customer_active_segment'].fillna('B')
test_data['customer_active_segment'] = test_data['X1'].fillna('BA')
test_data = pd.get_dummies(test_data, columns=["X1","customer_active_segment"],drop_first = True)

y_pred_randomForest = random.predict(test_data)
st.subheader('Predicted Value using Random Forest Classifier')
st.write(y_pred_randomForest)
 





