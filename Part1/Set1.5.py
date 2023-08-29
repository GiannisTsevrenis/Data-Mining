import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# date min 2019-12-31
# date max 2020-11-27
def get_week(row):
    return row.date.week


def get_covid_data():
    # vacation data frame
    df_vacation = pd.read_csv('vacations.csv', skiprows=2, index_col=False)
    df_vacation.rename(columns={'vacations: (Worldwide)': 'vacation searches'}, inplace=True)
    df_vacation.rename(columns={'Week': 'date'}, inplace=True)
    df_vacation.date = df_vacation.date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
    df_vacation['week'] = df_vacation.apply(get_week, axis=1)
    df_vacation = df_vacation.drop(columns=['date'])
    df_vacation = df_vacation[['week', 'vacation searches']]

    # covid data frame
    df_covid = pd.read_csv('owid-covid-data.csv')
    df_covid = df_covid[['date', 'total_cases_per_million', 'total_deaths_per_million']]
    df_covid.date = df_covid.date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
    df_covid['week'] = df_covid.apply(get_week, axis=1)
    df_covid = df_covid.drop(columns=['date'])
    df_covid = df_covid[['week', 'total_cases_per_million', 'total_deaths_per_million']]
    df_covid['total_cases_per_million'] = df_covid['total_cases_per_million'].fillna(0)
    df_covid['total_deaths_per_million'] = df_covid['total_deaths_per_million'].fillna(0)
    df_covid = df_covid.groupby(['week'], as_index=False).sum()

    # outter join
    df_covid = df_covid.merge(df_vacation, how='outer')

    # plot
    sns.lineplot(data=df_covid, x='total_cases_per_million', y='vacation searches')
    plt.show()

    pearson, value = stats.pearsonr(df_covid['vacation searches'], df_covid['total_cases_per_million'])
    print(f'pearson: {pearson} and p-value: {value}')


if __name__ == '__main__':
    get_covid_data()
