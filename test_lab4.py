import pytest
import pandas as pd
from streamlit_app import sibsp_group, calc_survival_share 

#Тест 1: категория 
@pytest.mark.parametrize("sibsp, expected", [
    (0, '0'),
    (1, '1–2'),
    (2, '1–2'),
    (3, '>2'),
    (10, '>2')
])
def test_sibsp_group(sibsp, expected):
    assert sibsp_group(sibsp) == expected


#Тест 2: calc_survival_share
def test_calc_survival_share_basic():
    data = pd.DataFrame({
        'Sex': ['male', 'female', 'male', 'female'],
        'SibSp': [0, 0, 1, 1],
        'Survived': [0, 1, 1, 1]
    })
    result = calc_survival_share(data)

    male_0 = result.query("Sex == 'male' and SibSp_group == '0'")['Survived'].iloc[0]
    female_0 = result.query("Sex == 'female' and SibSp_group == '0'")['Survived'].iloc[0]
    
    assert male_0 == 0.0
    assert female_0 == 1.0


#Тест 3: округление %
def test_survived_percent_rounding():
    data = pd.DataFrame({
        'Sex': ['male', 'female'],
        'SibSp': [1, 1],
        'Survived': [0.333, 0.666]
    })
    result = calc_survival_share(data)
    
    male_val = result.loc[result['Sex'] == 'male', 'Survived (%)'].iloc[0]
    female_val = result.loc[result['Sex'] == 'female', 'Survived (%)'].iloc[0]

    assert male_val == 33.3
    assert female_val == 66.6


#Тест 4: группы '>2'
def test_calc_survival_share_many_siblings():
    data = pd.DataFrame({
        'Sex': ['male', 'female'],
        'SibSp': [3, 4],
        'Survived': [0, 1]
    })
    result = calc_survival_share(data)
    assert all(result['SibSp_group'] == '>2')
