import streamlit as st
import pandas as pd


def sibsp_group(n: int) -> str:
    """Категоризация по количеству братьев/сестёр."""
    if n == 0:
        return '0'
    elif 1 <= n <= 2:
        return '1–2'
    return '>2'


def calc_survival_share(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет долю выживших по полу и группам SibSp."""
    df = df.copy()
    df['SibSp_group'] = df['SibSp'].apply(sibsp_group)

    survival_share = (
        df.groupby(['Sex', 'SibSp_group'])['Survived']
        .mean()
        .reset_index()
    )
    survival_share['Survived (%)'] = (
        survival_share['Survived'] * 100
    ).round(1)
    return survival_share


def load_data() -> pd.DataFrame:
    """Загружает набор данных Titanic."""
    return pd.read_csv(
        'https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/'
        'titanic_train.csv',
        index_col='PassengerId'
    )


def run_app():
    """Основная функция Streamlit-приложения."""
    titanic = load_data()
    survival_share = calc_survival_share(titanic)

    st.title("Лаб3(+4) в5 Руденко")

    selected_group = st.selectbox(
        "Выберите количество братьев/сестер:",
        ['0', '1–2', '>2']
    )

    filtered = survival_share[
        survival_share['SibSp_group'] == selected_group
    ]

    st.write(f"### Доля выживших для SibSp = {selected_group}")
    st.dataframe(filtered)


if __name__ == "__main__":
    run_app()
