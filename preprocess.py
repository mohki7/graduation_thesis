import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier


class Att_Analysis:
    def __init__(self, is_remove_covid=True, is_addup=True):
        """
        Args:
            is_remove_covid (bool, optional): コロナの影響で中止になった2021年を除外するかどうか. Defaults to True
            is_addup (bool, optional): 3月と10月を4月と9月に合算するかどうか. Defaults to True

        params: 
            START_YEAR: データの最初の年
            END_YEAR: データの最後の年
            df: すべてのデータフレームを結合したもの（START_YEARからEND_YEARまでのデータを単純結合）
            info_df: 各年のデータフレームの情報をまとめたデータフレーム
            df_yearly_att: すべての年の年ごとの合計観客数
            df_monthly_att_all: すべての年の月ごとの合計観客数(前処理なし)
            df_monthly_att_all_covied_removed: コロナの影響で中止になった2021年を除外したデータフレーム
            df_monthly_att_all_addup: すべての年の月ごとの合計観客数(3月と10月を4月と9月に合算)
            df_monthly_att_all_addup_covid_removed: コロナの影響で中止になった2021年を除外したデータフレームの3月と10月を4月と9月に合算したもの

        Functions:
            ダウンロード関係
                download_data(): すべてのcsvファイルを読み込む。重複行は日付の新しいものを残し、欠損値は削除する
                data_info(): 各年のデータフレームの情報をまとめたデータフレームを返す

            データフレーム関係
                年ごと
                get_yearly_att():すべての年の年ごとの合計観客数を作成
                月ごと
                    すべての年を返す
                    get_monthly_att_all(): すべての年の月ごとの合計観客数を返す
                    get_monthly_all_df(): covid_removedやaddupに応じたデータフレームを返す。すべての年数を返す。とりあえずこれを呼び出しておけば良い

                    指定した年のみを返す
                    get_monthly_att(year): 指定した年の月ごとの合計観客数を返す。これがすべての起点
                    get_monthly_df(year): covid_removedやaddupに応じたデータフレームを返す。指定した年のみを返す

            データフレーム加工関係
                get_covid_removed_df(): コロナの影響で中止になった2021年を除外したデータフレームを作成
                add_mar_oct_to_apr_sep(): 各年の3月と10月を4月と9月に合算する

            可視化関係
                年ごと
                plot_monthly_att_all(): すべての年の月ごとの平均観客数をプロットする。折れ線グラフ
                plot_monthly_att_bar_all(): すべての年の月ごとの合計観客数をプロットする。棒グラフ
                plot_yearly_att_bar(): すべての年の年ごとの合計観客数をプロットする。棒グラフ

                月ごと
                plot_monthly_att(year): 指定した年の月ごとの合計観客数をプロットする
                plot_monthly_att_bar(year): 指定した年の月ごとの合計観客数をプロットする。棒グラフ
                plot_monthly_att_only_plot(year): 指定した年の月ごとの合計観客数をプロットする。データ点のみ

            試合時間関係
                get_game_time(): すべての年の平均試合時間を返す
        """
        self.START_YEAR = 2013
        self.END_YEAR = 2023

        self.IS_REMOVE_COVID = is_remove_covid
        self.IS_ADD_UP = is_addup

        self.df = None
        self.info_df = None
        self.df_yearly_att = None
        self.df_monthly_att_all = None
        self.df_monthly_att_all_covid_removed = None
        self.df_monthly_att_all_addup = None
        self.df_monthly_att_all_addup_covid_removed = None
        self.df_game_time_addup= None

        self.fig_size = (12, 8)

        # self.df_2012 = None
        self.df_2013 = None
        self.df_2014 = None
        self.df_2015 = None
        self.df_2016 = None
        self.df_2017 = None
        self.df_2018 = None
        self.df_2019 = None
        self.df_2020 = None
        self.df_2021 = None
        self.df_2022 = None
        self.df_2023 = None

        self.download_data()
        self.get_monthly_att_all()
        self.get_yearly_att()
        if self.IS_REMOVE_COVID:
            self.get_covid_removed_df()
        if self.IS_ADD_UP:
            self.add_mar_oct_to_apr_sep()

    ## 基本機能
    def download_data(self):
        """すべてのcsvファイルを読み込む
            重複行は日付の新しいものを残し、欠損値は削除する
            →重複しているのは順延になった試合が両方とも記録されているため。日付が古い方は中止になった試合なので削除
            →欠損値は試合が中止になった、またはその順延試合が翌日のダブルヘッダーの1試合目の場合。そのため削除
        """
        # self.df_2012 = pd.read_csv('./data/mlb_2012.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2013 = pd.read_csv('./data/mlb_2013.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2014 = pd.read_csv('./data/mlb_2014.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2015 = pd.read_csv('./data/mlb_2015.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2016 = pd.read_csv('./data/mlb_2016.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2017 = pd.read_csv('./data/mlb_2017.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2018 = pd.read_csv('./data/mlb_2018.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2019 = pd.read_csv('./data/mlb_2019.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2020 = pd.read_csv('./data/mlb_2020.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2021 = pd.read_csv('./data/mlb_2021.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2022 = pd.read_csv('./data/mlb_2022.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()
        self.df_2023 = pd.read_csv('./data/mlb_2023.csv').sort_values(by='Date', ascending=False).drop_duplicates(subset=['Game ID'], keep='first').dropna()

        # すべてのデータフレームを結合
        self.df = pd.concat([self.df_2013, self.df_2014, self.df_2015, self.df_2016, self.df_2017, self.df_2018,
                             self.df_2019, self.df_2020, self.df_2021, self.df_2022, self.df_2023], ignore_index=True)

        # Date列をdatetime型に変換
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d')
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month

    def data_info(self):
        """各年のデータフレームの情報をまとめたデータフレームを返す
        Rows:
            year: 年
            n_game: 試合数
            n_NaN: 欠損値の数
            n_lack: 理論上の試合数との差(162*30/2=2430との差)

        Returns:
            DataFrame: 各年のデータフレームの情報をまとめたデータフレーム
        """
        df = pd.DataFrame()
        df['year'] = [year for year in range(self.START_YEAR, self.END_YEAR + 1)]

        df['n_game'] = [self.df_2013.shape[0], self.df_2014.shape[0], self.df_2015.shape[0], self.df_2016.shape[0], self.df_2017.shape[0], self.df_2018.shape[0],
                        self.df_2019.shape[0], self.df_2020.shape[0], self.df_2021.shape[0], self.df_2022.shape[0], self.df_2023.shape[0]]

        df['n_NaN'] = [self.df_2013.isnull().sum().sum(), self.df_2014.isnull().sum().sum(), self.df_2015.isnull().sum().sum(), self.df_2016.isnull().sum().sum(), self.df_2017.isnull().sum().sum(), self.df_2018.isnull().sum().sum(),
                        self.df_2019.isnull().sum().sum(), self.df_2020.isnull().sum().sum(), self.df_2021.isnull().sum().sum(), self.df_2022.isnull().sum().sum(), self.df_2023.isnull().sum().sum()]


        df['n_lack'] = df['n_game'] - 2430

        self.info_df = df
        return df

    def get_monthly_all_df(self):
        """covid_removedやaddupに応じたデータフレームを返す。すべての年数を返す

        Returns:
            DataFrame: covid_removed, addupに応じたデータフレーム
        """
        if self.IS_REMOVE_COVID:
            if self.IS_ADD_UP:
                return self.df_monthly_att_all_addup_covid_removed
            else:
                return self.df_monthly_att_all_covid_removed
        else:
            if self.IS_ADD_UP:
                return self.df_monthly_att_all_addup
            else:
                return self.df_monthly_att_all

    def get_monthly_df(self, year):
        """covid_removedやaddupに応じたデータフレームを返す。指定した年のみを返す

        Args:
            year (int): 観客者数を取得したい年

        Returns:
            DataFrame: その年のcovid_removed, addupに応じたデータフレーム。
        """
        if self.IS_REMOVE_COVID:
            if self.IS_ADD_UP:
                df = self.df_monthly_att_all_addup_covid_removed
            else:
                df = self.df_monthly_att_all_covid_removed
        else:
            if self.IS_ADD_UP:
                df = self.df_monthly_att_all_addup
            else:
                df = self.df_monthly_att_all

        df = df[df.index.str.contains(str(year))]
        return df

    def get_covid_removed_df(self):
        """コロナの影響で中止になった2021年を除外したデータフレームを作成
        """
        self.df_monthly_att_all_covid_removed = self.df_monthly_att_all.drop([f'2021-0{month}' for month in range(4, 10)], axis=0).drop(['2021-10'])

    def add_mar_oct_to_apr_sep(self):
        """各年の3月と10月を4月と9月に合算する。
        Returns:
            DataFrame: 3月と10月を4月と9月に合算したデータフレーム
        """
        if self.IS_REMOVE_COVID:
            df = self.df_monthly_att_all_covid_removed
        else:
            df = self.df_monthly_att_all


        years = [year for year in range(self.START_YEAR, self.END_YEAR + 1)]

        for year in years:
            try:
                df.loc[f'{year}-04', 'Attendance'] += df.loc[f'{year}-03', 'Attendance']
            except:
                pass
            try:
                df.loc[f'{year}-09', 'Attendance'] += df.loc[f'{year}-10', 'Attendance']
            except:
                pass
                # 3月の行は削除
            try:
                df.drop(f'{year}-03', inplace=True)
            except:
                pass
            # 10月の行は削除
            try:
                df.drop(f'{year}-10', inplace=True)
            except:
                pass

        if self.IS_REMOVE_COVID:
            self.df_monthly_att_all_addup_covid_removed = df
            return self.df_monthly_att_all_addup_covid_removed
        else:
            self.df_monthly_att_all_addup = df
            return self.df_monthly_att_all_addup

    def get_monthly_att(self, year):
        """指定した年の月ごとの合計観客数を返す。これがすべての起点

        Args:
            year (int): 年

        Returns:
            DataFrame: 指定した年の月ごとの合計観客数
        """
        # self.dfから指定した年のデータを抽出
        df_ = self.df[self.df['Year'] == year]

        df_monthly_att = df_.groupby('Month').agg({'Attendance': 'sum'}).astype(int)

        df_monthly_att.index = [str(year) + '-' + str(month) for month in df_monthly_att.index]
        df_monthly_att.index = pd.to_datetime(df_monthly_att.index, format='%Y-%m').strftime('%Y-%m')

        return df_monthly_att

    def plot_monthly_att(self, year, title=None):
        """指定した年の月ごとの合計観客数をプロットする

        Args:
            year (int): 年
        """
        if title is None:
            title = f'{str(year)} Monthly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'

        df = self.get_monthly_df(year=year)

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        # データ点をプロット
        plt.scatter(df.index, df['Attendance'], color='red', label='Data Points')
        plt.plot(df)
        plt.show()

    def plot_monthly_att_only_plot(self, year, title=None):
        """指定した年の月ごとの合計観客数をプロットする。データ点のみ

        Args:
            year (int): 年
        """
        if title is None:
            title = f'{str(year)} Monthly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'

        df= self.get_monthly_df(year=year)

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.scatter(df.index, df['Attendance'], color='red', label='Data Points')
        plt.show()

    def plot_monthly_att_bar(self, year, title=None):
        """指定した年の月ごとの合計観客数をプロットする。棒グラフ

        Args:
            year (int): 年
        """
        if title is None:
            title = f'{str(year)} Monthly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'
        df= self.get_monthly_df(year=year)

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.bar(df.index, df['Attendance'])
        plt.show()

    def plot_monthly_att_bar_all(self, title=None):
        """すべての年の月ごとの合計観客数をプロットする。棒グラフ

        Args:
            year (int): 年
        """
        if title is None:
            title = f'Monthly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'
        df_ = self.get_monthly_all_df()

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        plt.bar(df_.index, df_['Attendance'])
        plt.show()

    def get_monthly_att_all(self):
        """すべての年の月ごとの合計観客数を返す

        Edit:
            DataFrame: すべての年の月ごとの合計観客数
        """
        df_monthly_att_all = pd.DataFrame()
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            df_monthly_att = self.get_monthly_att(year)
            df_monthly_att_all = pd.concat([df_monthly_att_all, df_monthly_att], axis=0)

        self.df_monthly_att_all = df_monthly_att_all

    def get_yearly_att(self):
        """すべての年の年ごとの合計観客数を作成

        Edit:
            DataFrame: すべての年の年ごとの合計観客数
        """
        self.df_yearly_att = self.df.groupby('Year').agg({'Attendance': 'sum'}).astype(int)

        if self.IS_REMOVE_COVID:
            self.df_yearly_att.drop(2021, inplace=True)


    def plot_yearly_att_bar(self, title=None):
        """すべての年の年ごとの合計観客数をプロットする。棒グラフ

        Args:
            title (str, optional): グラフのタイトル. Defaults to None.
        """
        if title is None:
            title = f'Yearly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Attendance')
        # 軸ラベルは全ての年度を表示
        plt.xticks(self.df_yearly_att.index, self.df_yearly_att.index)
        # 棒の上に数値を表示
        for x, y in zip(self.df_yearly_att.index, self.df_yearly_att['Attendance']):
            plt.text(x, y, y, ha='center', va='bottom')
        plt.bar(self.df_yearly_att.index, self.df_yearly_att['Attendance'])
        plt.show()

    def plot_monthly_att_all(self, title=None):
        """すべての年の月ごとの平均観客数をプロットする。折れ線グラフ
        """
        if title is None:
            title = f'Monthly Attendance (Remove Covid:{self.IS_REMOVE_COVID}, Add Up:{self.IS_ADD_UP})'
        df_ = self.get_monthly_all_df()

        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        plt.scatter(df_.index, df_['Attendance'], color='red', label='Data Points')
        plt.plot(df_)
        plt.show()


    ## 試合時間に関する分析
    def get_game_time(self):
        """すべての年の平均試合時間を返す

        Returns:
            DataFrame: すべての年の平均試合時間
        """
        df_game_time = self.df.groupby('Year').agg({'Game Time (minutes)': 'mean'}).astype(int)

        if self.IS_REMOVE_COVID:
            df_game_time.drop(2021, inplace=True)

        return df_game_time

    def add_mar_oct_to_apr_sep_game_time(self):
        """各年の3月と10月を4月と9月に合算する。
        Returns:
            DataFrame: 3月と10月を4月と9月に合算したデータフレーム
        """
        # self.dfのMonthが3だったら4, 10だったら9に変更
        self.df_game_time_addup = self.df.copy()
        self.df_game_time_addup['Month'] = self.df_game_time_addup['Month'].replace({3: 4, 10: 9})


    def get_monthly_game_time(self, year):
        """指定した年の月ごとの平均試合時間を返す

        Args:
            year (int): 年

        Returns:
            DataFrame: 指定した年の月ごとの平均試合時間
        """
        if self.IS_ADD_UP:
            if self.df_game_time_addup is None:
                self.add_mar_oct_to_apr_sep_game_time()
            df_tmp = self.df_game_time_addup
        else:
            df_tmp = self.df
        df_ = df_tmp[df_tmp['Year'] == year]

        df_monthly_game_time = df_.groupby('Month').agg({'Game Time (minutes)': 'mean'}).astype(int)

        df_monthly_game_time.index = [str(year) + '-' + str(month) for month in df_monthly_game_time.index]
        df_monthly_game_time.index = pd.to_datetime(df_monthly_game_time.index, format='%Y-%m').strftime('%Y-%m')

        return df_monthly_game_time

    def get_monthly_game_time_all(self):
        """すべての年の月ごとの平均試合時間を返す"""
        df_monthly_game_time_all = pd.DataFrame()
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            if self.IS_REMOVE_COVID:
                if year == 2021:
                    continue
            df_monthly_game_time = self.get_monthly_game_time(year)
            df_monthly_game_time_all = pd.concat([df_monthly_game_time_all, df_monthly_game_time], axis=0)
        return df_monthly_game_time_all

    def plot_game_time(self):
        # すべての年の平均試合時間を棒グラフにする
        # バーの上にその試合時間を表示
        game_time = self.get_game_time()
        plt.figure(figsize=(12, 8))
        plt.title('Yearly Game Time')
        plt.xlabel('Year')
        plt.ylabel('Game Time (minutes)')
        plt.xticks(game_time.index, game_time.index)
        for x, y in zip(game_time.index, game_time['Game Time (minutes)']):
            plt.text(x, y, y, ha='center', va='bottom')
        plt.bar(game_time.index, game_time['Game Time (minutes)'])
        plt.show()
