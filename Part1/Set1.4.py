import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
from datetime import datetime
import numpy as np
import warnings
from itertools import combinations

indexes = ['total_cases_per_million', 'total_deaths_per_million', 'mortality_rate']
chars = ['gdp_per_capita', 'hospital_beds_per_thousand', 'population_density']
continents = ['North America', 'Asia', 'Africa', 'Europe', 'South America', 'Oceania']
assign2 = ['new_cases', 'new_deaths', 'new_mortality']


def init_data_frames():
    df = pd.read_csv('owid-covid-data.csv')
    df_ass_c = df.copy()

    # keep only columns of interest for assignment A. and B.
    df = df[['continent', 'date', 'total_cases', 'total_deaths', 'total_cases_per_million', 'total_deaths_per_million',
             'population_density', 'gdp_per_capita', 'hospital_beds_per_thousand']]

    df_full = df.copy()
    # date of interest assignment A.
    df = df.loc[df['date'] == '2020-11-01']

    # add new column with mortality rate
    df['mortality_rate'] = df.apply(lambda row: row.total_deaths / row.total_cases, axis=1)
    df_full['mortality_rate'] = df_full.apply(lambda row: row.total_deaths / row.total_cases, axis=1)

    df_ass_c = df_ass_c[['continent', 'date', 'new_cases', 'new_deaths']]

    return df, df_full, df_ass_c  # return data frames for specific subquery


def plot(df):
    # plot 3x3 grid
    fig, axs = plt.subplots(ncols=3, nrows=3)

    for i in range(3):
        for y in range(3):
            sns.scatterplot(data=df, x=indexes[i], y=chars[y], hue='continent', ax=axs[i][y])
    # fig.canvas.set_window_title('Window 3D')
    plt.show()


def correlations(df):
    # calculate correlation and p-value
    # test = df.corr(method='pearson')
    # test = test.drop(columns=['total_cases', 'total_deaths'])
    # test = test.drop(index=['total_cases', 'total_deaths'])

    test = df.drop(columns=['total_cases', 'total_deaths'])
    test = test.dropna()

    # print(stats.pearsonr(test['total_cases_per_million'], test['gdp_per_capita']))
    table = pd.DataFrame(columns=['gdp_per_capita', 'hospital_beds_per_thousand', 'population_density'],
                         index=['total_cases_per_million', 'total_deaths_per_million', 'mortality_rate'])
    for i in range(3):
        for y in range(3):
            pearson, value = stats.pearsonr(test[indexes[i]], test[chars[y]])
            # print(f'{indexes[i]} with {chars[y]}: {pearson} and {round(value, 3)}')  # TODO: make it table
            val = [pearson, value]
            table.loc[indexes[i], chars[y]] = val
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 400):
        print(table)


def ass_a(d1):  # assignment 4 subquery A.
    plot(d1)
    correlations(d1)

    # plot with log of population density
    d1['population_density'] = d1.apply(lambda row: math.log2(row.population_density), axis=1)
    plot(d1)

    # drop africa
    df_no_africa = d1.copy()
    df_no_africa = df_no_africa.loc[df_no_africa['continent'] != 'Africa']
    plot(df_no_africa)
    correlations(df_no_africa)

    # europeans only
    df_eu = d1.copy()
    df_eu = df_eu.loc[df_eu['continent'] == 'Europe']
    plot(df_eu)
    correlations(df_eu)


def ass_b(df):  # assignment 4 subquery B.
    df = df[['continent', 'total_cases_per_million', 'total_deaths_per_million', 'mortality_rate']]
    data = pd.DataFrame(columns=['total_cases_per_million', 'total_deaths_per_million', 'mortality_rate'],
                        index=['North America', 'Asia', 'Africa', 'Europe', 'South America', 'Oceania'])
    for i in continents:
        for y in indexes:
            temp = df.loc[df['continent'] == i]
            x = temp[y].mean(skipna=True)
            data.loc[i][y] = x
    # print(data)

    fig, axs = plt.subplots(ncols=3, nrows=1)
    for i in range(3):
        sns.barplot(x='continent', y=indexes[i], data=df, ax=axs[i])
        fig.autofmt_xdate()
    plt.show()

    # t1 = df.loc[df['continent'] == 'Africa'] # TODO: for the whole matrix 3 x combos
    # europe_total_cases_per_million = t1['total_cases_per_million']
    # t2 = df.loc[df['continent'] == 'Europe']
    # Asia_total_cases_per_million = t2['total_cases_per_million']
    #
    # print(stats.ttest_ind(europe_total_cases_per_million, Asia_total_cases_per_million, equal_var=False,
    #                       nan_policy='omit'))

    combos = list(combinations(continents, 2))  # find all possible country combinations
    for i in range(3):
        print(f'\n {indexes[i]}:')
        for y in combos:
            country_1, country_2 = y
            print(f'{country_1} with {country_2}:')
            t1 = df.loc[df['continent'] == country_1]
            aaa = t1[indexes[i]]
            t2 = df.loc[df['continent'] == country_2]
            bbb = t2[indexes[i]]

            print(stats.ttest_ind(aaa, bbb, equal_var=False,
                                  nan_policy='omit'))

    # sns.barplot(x='continent', y='total_deaths_per_million', data=df)
    # sns.barplot(x='continent', y='mortality_rate', data=df)
    # print(data)


def get_month(row):
    return row.date.month


def ass_c(df):  # assignment 4 subquery C.
    # convert date column to datetime objects
    df.date = df.date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))

    # filter date
    df = df[~(df['date'] < '2020-01-01')]

    # df.index = df.date
    # df = df.drop(columns=['date'])
    df = df.sort_values(by='date')
    # df['month'] = df.apply(get_month, axis=1)
    # print(df['month'])

    # add new mortality column
    df['new_mortality'] = df.apply(lambda row: row.new_deaths / row.new_cases if row.new_cases != 0 else 0, axis=1)
    # daily = df.groupby(pd.Grouper(key='date', freq='d')).sum()

    # add new month column for plotting
    df['month'] = df.apply(get_month, axis=1)

    fig, axs = plt.subplots(ncols=3, nrows=6)
    count = 0
    colors = ['Set2', 'rocket', 'magma', 'mako', 'viridis', 'cubehelix']
    for y in continents:
        temp = df.loc[df['continent'] == y]
        for i in range(3):
            sns.barplot(x='month', y=assign2[i], data=temp, ax=axs[count][i], hue='continent',
                        ci=40, dodge=True, palette=colors[count])  # TODO alla3e ta xrwmata sta countries
        count += 1
    plt.show()


def ass_d():  # assignment 4 subquery D.
    # read files
    voting = pd.read_excel('state-voting.xlsx')
    covid = pd.read_csv('united_states_covid19_cases_and_deaths_by_state.csv', skiprows=3, index_col=False)

    # outer join files
    covid = covid[['State/Territory', 'Case Rate per 100000']]
    covid.rename(columns={'State/Territory': 'State'}, inplace=True)

    # add new York city cases to the state's and remove city
    x1 = covid.iloc[38][1] + covid.iloc[39][1]
    covid.iloc[[38], [1]] = x1
    covid = covid.loc[covid['State'] != 'New York City']

    voting = voting[['State', 'Vote']]
    covid = covid.merge(voting, how='outer')
    covid.dropna(inplace=True)

    sns.scatterplot(data=covid, x='Vote', y='Case Rate per 100000', hue='State', legend=False)
    # pearson, value = stats.pearsonr(covid['State'], covid['Vote'])
    # print(f'pearson: {pearson} and p-value: {value}')
    plt.show()


if __name__ == '__main__':
    d1, d2, d3 = init_data_frames()
    ass_a(d1)
    ass_b(d2)
    ass_c(d3)
    ass_d()

    # print(list(df))  # all columns
    # for col in df: # unique dates
    #     print(df['date'].unique())
    # print(df.tail(10))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df.loc[df['date'] == '2020-11-01'])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df[['total_cases_per_million', 'total_deaths_per_million']].dropna(how='all'))  # drop if both null

    # sns.scatterplot(data=df, x="total_cases_per_million", y="gdp_per_capita")
