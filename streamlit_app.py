import streamlit as st
import pandas as pd

titanic = pd.read_csv('https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv', index_col='PassengerId')

def sibsp_group(n):
    if n == 0:
        return '0'
    elif 1 <= n <= 2:
        return '1–2'
    else:
        return '>2'

titanic['SibSp_group'] = titanic['SibSp'].apply(sibsp_group)

survival_share = titanic.groupby(['Sex', 'SibSp_group'])['Survived'].mean().reset_index()

survival_share['Survived (%)'] = (survival_share['Survived'] * 100).round(1)


st.title("Лаб3 в5 Руденко")
st.write(
    survival_share
)
