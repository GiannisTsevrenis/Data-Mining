import pandas as pd
import numpy as np
from scipy import spatial


def input_matrix():  # given table
    data = {'Movie1': [5, 4, 3, 2],
            'Movie2': [3, 2, None, 5],
            'Movie3': [None, 1, None, 1],
            'Movie4': [1, None, 1, 5],
            'Movie5': [None, None, 3, 3],
            'Movie6': [None, 1, 3, 4]}

    df = pd.DataFrame(data, index=['User X',
                                   'User Y',
                                   'User Z',
                                   'User W'])
    # print(df)
    return df


def cos(df):
    df = df.replace(np.nan, 0)
    print(df)
    y = 1 - spatial.distance.cosine(df.iloc[0], df.iloc[1])  # User X with User Y
    z = 1 - spatial.distance.cosine(df.iloc[0], df.iloc[2])  # User X with User Z
    w = 1 - spatial.distance.cosine(df.iloc[0], df.iloc[3])  # User X with User W
    users = {
        "Y": y,
        "Z": z,
        "W": w
    }
    for key in users:
        print(f'Cosine similarity between User X and User {key}: {users[key]}')

    first, second = find_most_similar(users)
    print(f'Most similar users are to X: {first} and {second}')
    first_user = 'User ' + first
    second_user = 'User ' + second
    first_weight = df.loc[first_user, 'Movie6']
    second_weight = df.loc[second_user, 'Movie6']

    prediction = (users[first] * first_weight + users[second] * second_weight) / (users[first] + users[second])
    return prediction


def find_most_similar(dict):
    my_dict = dict.copy()
    first = max(my_dict, key=my_dict.get)
    del my_dict[first]
    second = max(my_dict, key=my_dict.get)
    return first, second


def ass_3_1(df):  # assignment 3 subquery 1
    prediction = cos(df)
    print(f'prediction for User X, Movie 6: {prediction}')


def ass_3_2(df):  # assignment 3 subquery 2
    mean = df.loc['User X'].mean()
    df = df.sub(df.mean(axis=1), axis=0)
    prediction = cos(df)
    print(f'prediction for User X, Movie 6: {prediction + mean}')


if __name__ == '__main__':
    matrix = input_matrix()
    ass_3_1(matrix)
    ass_3_2(matrix)
