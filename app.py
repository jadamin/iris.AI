import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from streamlit.elements.button import ButtonSerde


df = pd.read_csv("irisdata.csv")


st.title("Ma premiere application IA")
st.header("Bienvenue ")


X= df.drop("species",axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=12)
print('train set :' , X_train.shape)
print( 'test set :' , X_test.shape)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
print('train score:', model.score (X_test,y_test))


sns.set_theme(style="ticks")
sns.pairplot(df, hue="species")

df_graphic = (df)



sepal_lengthst = st.slider('sepal_length', min_value=0.00, max_value=8.00)


sepal_width = st.slider('sepal_width', min_value=0.00, max_value=5.00)


petal_length = st.slider('petal_length', min_value=0.00, max_value=8.00)


petal_width = st.slider('petal_width', min_value=0.00, max_value=3.00)


if st.button('valider'):
    st.success (model.predict([[sepal_lengthst,sepal_width,petal_length,petal_width]]))
    st.pyplot(sns.pairplot(df, hue="species"))

test = model.predict([[sepal_lengthst,sepal_width,petal_length,petal_width]])

















