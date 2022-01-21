import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import os


def prepare_data(data: pd.DataFrame, train_data: pd.DataFrame) -> pd.DataFrame:
    # 1. Data validation
    assert isinstance(data, pd.DataFrame) and isinstance(train_data, pd.DataFrame)
    modified_df = data.copy(deep=True)
    random_seed = 4
    train_df = train_data.copy(deep=True)

    # 2. Feature selection

    # 2.1 Set categories
    def prepare_dataframe_for_analasys(df):
        # Drop unused features
        df.drop(['patient_id', 'job', 'current_location', 'sport_activity', 'happiness_score',
                 'conversations_per_day'], axis=1, inplace=True)

        # Converting the date to an appropriate pandas dtype
        df['pcr_date'] = pd.to_datetime(df['pcr_date'], infer_datetime_format=True, errors='coerce')

        # 2.2 OHE conversions

        symptoms_ohe_vec = df['symptoms'].str.get_dummies(sep=';')
        df = pd.concat([df, symptoms_ohe_vec], axis=1)
        df = df.drop('symptoms', axis=1)

        return df

    train_df = prepare_dataframe_for_analasys(train_df)
    modified_df = prepare_dataframe_for_analasys(modified_df)
    # 3. Outlier handling
    # 3.1 Global outliers
    distributions = {
        'normal_dist': ['sugar_levels', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_10'],
        'other_dist': ['PCR_01', 'PCR_02', 'PCR_08', 'PCR_09']}

    def get_global_outlier_caps_from_dataframe(df):
        col_clip_vars = {}

        def normal_dist_outlier_removal(normal_col, std_modefier=3.0):
            sup = df[normal_col].mean() + std_modefier * df[normal_col].std()
            inf = df[normal_col].mean() - std_modefier * df[normal_col].std()
            return sup, inf

        def precntile_outlier_removal(percentile_col, quantile_perecntage=0.75, iqr_modifier=1.5):
            # quantile_percentage = 0.75 for normal skewed dist, for precntile standard is 0.99
            q3 = df[percentile_col].quantile(quantile_perecntage)
            q1 = df[percentile_col].quantile(1 - quantile_perecntage)
            iqr = q3 - q1
            sup = q3 + iqr_modifier * iqr
            inf = q1 - iqr_modifier * iqr
            return sup, inf

        for outlier_col in df.columns:
            if outlier_col in distributions['normal_dist']:
                col_clip_vars[outlier_col] = normal_dist_outlier_removal(outlier_col, std_modefier=3)
            elif outlier_col in distributions['other_dist']:
                col_clip_vars[outlier_col] = precntile_outlier_removal(outlier_col, quantile_perecntage=0.85)
            else:
                continue
        return col_clip_vars

    # Capping global outliers
    global_column_clip_vars = get_global_outlier_caps_from_dataframe(train_df)
    # get list of keys in dictionary
    clipping_cols = list(distributions['normal_dist']) + list(
        distributions['other_dist'])
    for col in clipping_cols:
        train_df[col].clip(lower=global_column_clip_vars[col][1], upper=global_column_clip_vars[col][0], inplace=True)
        modified_df[col].clip(lower=global_column_clip_vars[col][1], upper=global_column_clip_vars[col][0],
                              inplace=True)

    # 3.2 Contextual outliers
    # Binning age categories
    bins = [0, 2, 5, 9, 13, 20, 120]
    # According to results:
    # https://greatist.com/health/how-much-should-i-weigh#based-on-age-and-sex
    weight_range_by_age_group = {
        # Weight in kgs. [0] is lower bound, [1] is upper bound, [2] is "regular weight" for the age group.
        'infant': (10.4, 19, 12.25),
        'Toddler': (10.4, 22.2, 16.8),
        'Kid': (14.5, 36.75, 25.4),
        'Tween': (22.2, 60.33, 41.75),
        'Teen': (37.65, 88.45, 63.0),
        'Adult': (45.8, 118.0, 79.37)}





    age_groups = list(weight_range_by_age_group.keys())

    # 4. Data imputing
    # Imputating age first, as it will be needed to impute weight.
    def random_sample_imputatation(column_name: str):
        # Creates vectors with random smaples from non-imputed train_data (no NaN values)
        # Each vector has size of num of NaN entries in the corresponding dataframe
        sample_df = train_df[column_name].dropna().copy(deep=True)
        rnd_train_data_vals = sample_df.sample(train_df[column_name].isnull().sum(), random_state=random_seed)
        rnd_modified_data_vals = sample_df.sample(modified_df[column_name].isnull().sum(), random_state=random_seed)

        # Give each entery an index to fit a NaN entry in the original dataframe
        rnd_train_data_vals.index = train_df[train_df[column_name].isnull()].index
        rnd_modified_data_vals.index = modified_df[modified_df[column_name].isnull()].index

        # Fill series with random values that we sampled
        train_df.loc[train_df[column_name].isnull(), column_name] = rnd_train_data_vals
        modified_df.loc[modified_df[column_name].isnull(), column_name] = rnd_modified_data_vals

        return train_df[column_name], modified_df[column_name]

    train_df['age'], modified_df['age'] = random_sample_imputatation('age')

    assert not train_df.age.isna().any()
    assert not modified_df.age.isna().any()

    train_df['age_group'] = pd.cut(train_df['age'], bins=bins, labels=list(weight_range_by_age_group.keys()),
                                   right=False)
    mean_weights_by_age_group = train_df.groupby('age_group', as_index=True)['weight'].mean()

    train_df['weight'] = train_df.apply(
        lambda x: mean_weights_by_age_group.at[x, 'age_group'] if pd.isnull(x['age']) else x['weight'], axis=1)
    modified_df['weight'] = modified_df.apply(
        lambda x: mean_weights_by_age_group.at[x, 'age_group'] if pd.isnull(x['age']) else x['weight'], axis=1)
    train_df.drop('age_group', axis=1, inplace=True)

    # in case the age group doesn't exist, which should never happen and has never happened.
    train_df = train_df.fillna({'weight': train_df.weight.median()})
    modified_df = modified_df.fillna({'weight': modified_df.weight.median()})

    # Impute the rest of the features
    for col in train_df.columns:
        train_df[col], modified_df[col] = random_sample_imputatation(col)

    # 5. Return data
    modified_df.drop(
        ['pcr_date', 'PCR_08', 'sex', 'headache', 'PCR_04', 'PCR_06', 'low_appetite', 'PCR_09',
         'weight'], axis=1, inplace=True)
    col_order = ['age', 'blood_type', 'sugar_levels', 'num_of_siblings', 'household_income', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_05',
                 'PCR_07', 'PCR_10', 'cough', 'fever', 'shortness_of_breath', 'VirusScore']
    modified_df = change_binary_feature_format(modified_df)
    modified_df = modified_df.reindex(columns=col_order)
    modified_df = normilize_data(modified_df)
    return modified_df


def change_binary_feature_format(df: pd.DataFrame):
    # binary_feautres = ['risk', 'spread', 'covid', 'cough', 'fever', 'shortness_of_breath',
    #                    'A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    # df['covid'] = df['covid'].astype(int)
    binary_feautres = ['cough', 'fever', 'shortness_of_breath']
    replace_dict = {1: 1, 0: -1, 'true': 1, 'false': -1, 'High': 1, 'Low': -1, True: 1, False: -1}
    df[binary_feautres] = df[binary_feautres].replace(replace_dict)
    return df


def normilize_data(df):
    minmax_scaler = MinMaxScaler()
    z_score_Scaler = StandardScaler()
    normalization_dict = {"z_score": ["PCR_03", "PCR_05", "PCR_07", "PCR_10", "sugar_levels"],
                          "minmax": ["age", "num_of_siblings", "household_income", "PCR_01", "PCR_02", ]}

    df[normalization_dict['minmax']] = minmax_scaler.fit_transform(df[normalization_dict['minmax']])
    df[normalization_dict['z_score']] = z_score_Scaler.fit_transform(df[normalization_dict['z_score']])
    return df
