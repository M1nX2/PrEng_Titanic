import pandas as pd
import ipywidgets as widgets
from IPython.display import display

titanic = pd.read_csv(
    'https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv',
    index_col='PassengerId'
)

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

dropdown = widgets.Dropdown(
    options=['0', '1–2', '>2'],
    value='0',
    description='SibSp:',
    style={'description_width': '70px'}
)


def update_table(selected_group):
    filtered = survival_share[survival_share['SibSp_group'] == selected_group]
    display(filtered)

st.title("Лаб3 в5 Руденко")
widgets.interact(update_table, selected_group=dropdown)
