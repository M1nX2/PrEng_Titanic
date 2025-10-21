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

    male_1_2 = result.query("Sex == 'male' and SibSp_group == '1–2'")['Survived'].iloc[0]
    female_1_2 = result.query("Sex == 'female' and SibSp_group == '1–2'")['Survived'].iloc[0]
    
    assert male_1_2 == 1.0
    assert female_1_2 == 1.0


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

#Тест 5: проверка всех групп
def test_calc_survival_share_all_groups():
    data = pd.DataFrame({
        'Sex': ['male', 'male', 'female', 'female'],
        'SibSp': [0, 3, 1, 5],
        'Survived': [1, 0, 1, 0]
    })
    result = calc_survival_share(data)

    for sex, sibsp, expected_group in [('male', 0, '0'), ('male', 3, '>2'), 
                                       ('female', 1, '1–2'), ('female', 5, '>2')]:
        val = result.query(f"Sex == '{sex}' and SibSp_group == '{expected_group}'")['Survived'].iloc[0]
        expected = data.query(f"Sex == '{sex}' and SibSp == {sibsp}")['Survived'].mean()
        assert val == expected