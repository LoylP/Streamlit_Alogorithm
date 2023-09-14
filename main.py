import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from time import sleep
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, metrics

st.set_page_config(
    page_title = 'LoylP',
    page_icon = '🤘',
)
#Sidebar
st.sidebar.success("# Welcome to ...", icon = "👋")
st.sidebar.markdown('----')
st.sidebar.success('# Show dataframe từ file csv', icon = "📄")
st.sidebar.markdown('----')
st.sidebar.success('# Choose input feature', icon = "📩")
st.sidebar.markdown('----')
st.sidebar.success('# Choose Algorithm', icon = "🧮")
st.sidebar.markdown('## Decision tree')
st.sidebar.markdown('## Linear regression')
#st.sidebar.markdown('## XGBoost')
st.sidebar.markdown('----')
st.sidebar.success('# Choose ratio of train/test split', icon = "🎚")
st.sidebar.markdown('----')
st.sidebar.success('# Drawexplicitly chart', icon = "📊")
st.sidebar.markdown('----')
#Title
#Tên chương trình
st.title(' :orange[Chương trình] :green[kiểm tra] :violet[MAE và MSE]')
#Thêm file csv
st.title(" :red[Hãy thêm file csv vào đây]")
sleep(0.5)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:   
    file = pd.read_csv(uploaded_file)
    st.write(file)
    my_data_1 = pd.DataFrame(file).to_numpy()
    label_encoder = preprocessing.LabelEncoder()
    my_data_1[:, 3] = label_encoder.fit_transform(my_data_1[:, 3])
#Chọn input cho bài
    st.title(" :blue[Choose input feature]")
    click1 = st.checkbox('R&D Spend')
    click2 = st.checkbox('Administration')
    click3 = st.checkbox('Marketing Spend')
    click4 = st.checkbox('State')
    del file['Profit']
    if not click1:
        del file['R&D Spend']
    if not click2:
        del file['Administration']
    if not click3:
        del file['Marketing Spend']
    if not click4:
        del file['State']
    my_data_2 = pd.DataFrame(file).to_numpy()
    label_encoder = preprocessing.LabelEncoder()
    my_data_2[:, -1] = label_encoder.fit_transform(my_data_2[:, -1])
    st.write(my_data_2)
# Kéo thanh tỉ lệ
    def keothanhtile():
        st.title(" :green[Choose ratio of train/test split]")
        ti_le = st.slider('Chọn tỉ lệ', 0.0, 1.0, 0.1)
        st.write("Bạn đã chọn tỉ lệ là: ", ti_le)
        X = my_data_1[:, :4]
        Y = my_data_1[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        my_prof = DecisionTreeRegressor(min_samples_leaf=4, min_samples_split=4, random_state=0).fit(X_train, Y_train)
        Y_predict = my_prof.predict(X_test)
        #MSE1 = mean_squared_error(Y_predict, Y_train)
        MSE2 = mean_squared_error(Y_predict, Y_test)
        #MAE1 = mean_absolute_error(Y_predict, Y_train)
        MAE2 = mean_absolute_error(Y_predict, Y_test)
        # Show biểu đồ cột
        st.title(" :violet[Drawexplicitly chart]")
        df = {"Name": ["MSE_test", "MAE_test"], "Score": [MSE2, MAE2]}
        df = pd.DataFrame(df)
        st.dataframe(df)
        fig = px.bar(df, x="Name", y="Score", color="Name", text="Score", )
        st.plotly_chart(fig)

#Chọn thuật toán
    sleep(2)
    st.title(" :orange[Choose Algorithm]")
    option = st.selectbox('Hãy chọn thuật toán bạn muốn', ('Choose','Decision Tree Regression', 'Linear Regression'))
#option = st.selectbox('Hãy chọn thuật toán bạn muốn', ('Choose','Decision Tree Regression', 'Linear Regression', 'XGBoost'))
    if option == 'Decision Tree Regression':
        sleep(0.5)
        st.caption('## Bạn đã chọn Decision Tree Regression')
        keothanhtile()
    else:
        sleep(0.5)
        st.caption('## Bạn đã chọn Linear Regression')
    #code cho phần linear tương tự 
        st.title(" :green[Choose ratio of train/test split]")
        ti_le = st.slider('Chọn tỉ lệ', 0.0, 1.0, 0.1)
        st.write("Bạn đã chọn tỉ lệ là: ", ti_le)
        X = my_data_1[:, :4]
        Y = my_data_1[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        my_prof = LinearRegression().fit(X_train, Y_train)
        Y_predict = my_prof.predict(X_test)
        MSE2 = mean_squared_error(Y_predict, Y_test)
        MAE2 = mean_absolute_error(Y_predict, Y_test)
    # Show biểu đồ cột
        st.title(" :violet[Drawexplicitly chart]")
        df = {"Name": ["MSE_test", "MAE_test"], "Score": [MSE2, MAE2]}
        df = pd.DataFrame(df)
        st.dataframe(df)
        fig = px.bar(df, x="Name", y="Score", color="Name", text="Score", )
        st.plotly_chart(fig)