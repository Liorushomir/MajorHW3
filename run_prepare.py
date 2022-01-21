from sklearn.model_selection import train_test_split
from prepare import prepare_data
import pandas as pd
from pprint import pprint
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    dataset = pd.read_csv('original_data/virus_labeled.csv')
    old_raw_train_set = pd.read_csv('previous_data_for_testing/raw_train_set.csv').set_index('Unnamed: 0')
    old_train_set = pd.read_csv('previous_data_for_testing/train_set.csv')
    raw_train_set, raw_test_set = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=4)

    raw_train_set = raw_train_set.round(5)
    old_raw_train_set = old_raw_train_set.round(5)
    columns = raw_train_set.columns.to_list()
    old_columns = old_raw_train_set.columns.to_list()
    [old_columns.remove(col) for col in ['risk', 'covid', 'spread']]
    columns.remove('VirusScore')
    old_raw_train_set['address'].replace('\\r\\n', '\\n', regex=True, inplace=True)

    assert columns == old_columns
    assert raw_train_set.index.equals(old_raw_train_set.index)
    assert raw_train_set[columns].compare(old_raw_train_set[old_columns]).empty

    train_set = prepare_data(raw_train_set, raw_train_set)
    test_set = prepare_data(raw_train_set, raw_test_set)

    assert all([col in train_set.columns for col in ['PCR_01', 'PCR_02', 'PCR_03',  'PCR_07', 'PCR_10',
                                                     'num_of_siblings', 'sugar_levels', 'VirusScore']])
    assert not any([col in train_set.columns for col in ['job', 'risk', 'covid', 'spread']])
    # assert train_set.index.equals(old_train_set.index)
    # print(train_set[columns].compare(old_train_set[old_columns]))
    # assert train_set[columns].compare(old_train_set[old_columns]).empty
    # print(raw_train_set)
    print(raw_train_set['VirusScore'].compare(train_set['VirusScore']))
    train_set.to_csv('prepared_data/train_set.csv')
    test_set.to_csv('prepared_data/test_set.csv')


