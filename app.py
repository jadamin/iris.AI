import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




st.title("Ma premiere application IA")
st.header("Bienvenue ")


df = pd.read_csv("irisdata.csv")
st.slider('sepal_length', min_value=1, max_value=150)
sepal_length = st.slider('', 0, 130, 25)























