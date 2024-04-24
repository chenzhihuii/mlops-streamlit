import streamlit as st
st.set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

# Load image
st.sidebar.image("health_care.png", use_column_width=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA

start_time = time.time()
# Sidebar options
st.sidebar.markdown("""
<div style="text-align: justify;">
    <p>The healthcare application is designed to predict heart attacks and acute lymphoblastic leukemia (a blood cancer).</p>
    <p>It utilizes machine learning models such as Logistic Regression and KNN to perform prediction.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("By: Firmanda Wahyunita")
st.sidebar.title("Dataset and Classifier")
dataset_name = st.sidebar.selectbox("Select Dataset:", ("Heart Attack", "Leukimia"))
classifier_name = st.sidebar.selectbox("Select Classifier:", ("Logistic Regression", "KNN"))

LE=LabelEncoder()
def get_dataset(dataset_name):
    if dataset_name=="Heart Attack":
        data=pd.read_csv("heart.csv")
        st.header("Heart Attack Prediction")
        return data

    else:
        data=pd.read_csv("ALLcancer.csv")
        
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        st.header("Leukimia Prediction")
        return data

data = get_dataset(dataset_name)

def selected_dataset(dataset_name):
    if dataset_name == "Heart Attack":
        X=data.drop(["output"],axis=1)
        Y=data.output
        return X,Y

    elif dataset_name == "Leukimia":
        X = data.drop(["id","diagnosis"], axis=1)
        Y = data.diagnosis
        return X,Y

X,Y=selected_dataset(dataset_name)

#Plot output variable
def plot_op(dataset_name):
    col1, col2 = st.columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Classes in 'Y'")
    if dataset_name == "Heart Attack":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()

    elif dataset_name == "Leukimia":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()


def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select values: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
        MI = st.sidebar.slider("max_iter",50,400,step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,20)
        params["K"] = K

    RS=st.sidebar.slider("Random State",0,100)
    params["RS"] = RS
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    global clf
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["R"],max_iter=params["MI"])

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    return clf

clf = get_classifier(classifier_name,params)

#Build Model
def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65)

    #MinMax Scaling / Normalization of data
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)

    return Y_pred,Y_test

Y_pred,Y_test=model()

#Plot Output
def compute(Y_pred,Y_test):
    c1, c2 = st.columns((4,3))
    #Output plot
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(Y_pred)),Y_pred,color="yellow",lw=5,label="Predictions")
    plt.scatter(range(len(Y_test)),Y_test,color="red",label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()
    plt.grid(True)
    c1.pyplot()

    #Confusion Matrix
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    c2.pyplot()

    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
    st.subheader("Metrics of the model: ")
    st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
        round(precision, 2), round(recall, 2), round(fscore,2), round((acc*100),1), round((mse),2)))

st.markdown("<hr>",unsafe_allow_html=True)
st.header(f"1) Model for Prediction of {dataset_name}")
st.subheader(f"Classifier Used: {classifier_name}")
compute(Y_pred,Y_test)

#Execution Time
end_time=time.time()
st.info(f"Total execution time: {round((end_time - start_time),4)} seconds")

#Get user values
def user_inputs_ui(dataset_name,data):
    user_val = {}
    if dataset_name == "Leukimia":
        X = data.drop(["id","diagnosis"], axis=1)
        for col in X.columns:
            name=col
            col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            user_val[name] = round((col),4)

    elif dataset_name == "Heart Attack":
        X = data.drop(["output"], axis=1)
        for col in X.columns:
            name=col
            col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            user_val[name] = col

    return user_val

#User values
st.markdown("<hr>",unsafe_allow_html=True)
st.header("2) User Values")
with st.expander("See more"):
    st.markdown("""
    In this section you can use your own values to predict the target variable. 
    Input the required values below and you will get your status based on the values. <br>
    <b style='color: red;'>1 - High Risk</b><br>
    <b style='color: green;'>0 - Low Risk</b>
    """, unsafe_allow_html=True)


user_val=user_inputs_ui(dataset_name,data)

#@st.cache(suppress_st_warning=True)
def user_predict():
    global U_pred
    if dataset_name == "Leukimia":
        X = data.drop(["id","diagnosis"], axis=1)
        U_pred = clf.predict([[user_val[col] for col in X.columns]])

    elif dataset_name == "Heart Attack":
        X = data.drop(["output"], axis=1)
        U_pred = clf.predict([[user_val[col] for col in X.columns]])

    st.subheader("Your Status: ")
    if U_pred == 0:
        st.markdown("""
        <div style='background-color: green; padding: 7px; border-radius: 3px;'>
            <b style='color: black; font-size: 18px;'>[0] - You are not at high risk 😉</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: red; padding: 7px; border-radius: 3px;'>
            <b style='color: white; font-size: 18px;'>[1] - You are at high risk 😢</b>
        </div>
        """, unsafe_allow_html=True)

user_predict()

