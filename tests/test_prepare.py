import pytest

from prepare import *
import pandas as pd


def test_prepare_data__correct_columns(dummy_prepared_data):
    columns = set(dummy_prepared_data.columns)
    must_include = {'PCR_01', 'PCR_02', 'PCR_03', 'PCR_07', 'PCR_10', 'num_of_siblings', 'sugar_levels',
                    'household_income'}
    assert must_include.issubset(columns)
    assert 'job' not in columns


def test_prepare_data__binary_features(dummy_prepared_data):
    binary_features = ['risk', 'spread', 'covid', 'cough', 'fever', 'shortness_of_breath', 'A+', 'A-', 'AB+', 'AB-',
                       'B+', 'B-', 'O+', 'O-']
    binary_data = change_binary_feature_format(dummy_prepared_data)
    assert isinstance(binary_data, pd.DataFrame)
    covid_ = dummy_prepared_data['covid']
    # print(type(covid_), covid_)
    for feat in binary_features:
        unique = binary_data[feat].unique()
        assert len(unique) == 2
        assert set(unique) == {1, -1}

#No longer needed
@pytest.mark.skip
def test_prepare_vs_collab(untrained_set, collab_trained_set):
    missing_columns = must_include = {'PCR_01', 'PCR_02', 'PCR_03', 'PCR_07', 'PCR_10', 'num_of_siblings', 'sugar_levels',
                    'household_income'}
    trained_data = prepare_data(untrained_set, untrained_set)
    col_list = list(collab_trained_set.columns)
    col_list.remove('Unnamed: 0')
    trained_final = trained_data[col_list].round(6)
    collab_final = collab_trained_set[col_list].round(6)
    empty = trained_final.compare(collab_final)
    assert empty.empty

    trained_data.to_csv('../train_set.csv')



def test_prepare_normalized(untrained_set):
    trained_data = prepare_data(untrained_set, untrained_set)
    normalization_dict = {"z_score": ["PCR_03", "PCR_05", "PCR_07", "PCR_10", "sugar_levels"],
                          "minmax": ["age", "num_of_siblings", "household_income", "PCR_01", "PCR_02", ]}
    for col in normalization_dict['minmax']:
        assert trained_data[col].between(0, 1).all()

@pytest.fixture
def dummy_prepared_data():
    data_set = pd.read_csv('../data_sets/virus_data.csv')
    prepared_data = prepare_data(data_set, data_set)
    print("HEY")
    return prepared_data


@pytest.fixture
def untrained_set():
    data_set = pd.read_csv('./train_set.csv')
    return data_set

@pytest.fixture
def collab_trained_set():
    data_set = pd.read_csv('./collab_train_set.csv')
    return data_set