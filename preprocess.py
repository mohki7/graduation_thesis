import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
import holidays
import datetime

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning


#! コロナのデータを追加したい。というより、新たなデータを追加する汎用的な関数を作りたい
    # - dfを渡すと、dfに追加する関数
    #     - とりあえずコロナ期間削除、3, 10月統合版に追加
    # - あとInterruptedTimeSeries.pyも、変数の指定を整理して汎用的にする

class Att_Analysis:
    def __init__(self, is_remove_covid=True, is_addup=True, how_completion_outlier=False):
        """
        Args:
            is_remove_covid (bool, optional): コロナの影響で中止になった2021年を除外するかどうか. Defaults to True
            is_addup (bool, optional): 3月と10月を4月と9月に合算するかどうか. Defaults to True
            completion_outlier(choice, optional): 2022年4月の外れ値を補完するかどうか. Defaults to False。with_mean, with_predict, with_predict_with_mean, with_medianから選択

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
        self.df_holidays = None
        self.df_non_holidays = None

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

        self.how_completion_outlier = how_completion_outlier

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
                # self.how_completion_outlierがFalse出ないときのみcompletion_outlier()を実行
                if self.how_completion_outlier is False:
                    return self.df_monthly_att_all_addup_covid_removed
                else:
                    self.completion_outlier()
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

    def get_holidays_df(self):
        """祝日のみのデータフレームと祝日以外のデータフレームを返す
        """
        self.is_holiday(self.df)
        self.df_holidays = self.df[self.df['is_holiday'] == 1]
        self.df_non_holidays = self.df[self.df['is_holiday'] == 0]

        self.get_monthly_att_all_for_holidays() # holidays関連のdfで月ごとの観客数を取得
        self.get_monthly_att_all_for_non_holidays() # non_holidays関連のdfで月ごとの観客数を取得
        self.remove_covid_for_holidays() # holidays関連のdfでコロナの影響を除去
        self.add_mar_oct_to_apr_sep_for_holidays() # hokidays関連のdfで3月と10月を4月と9月に合算

        return self.df_holidays, self.df_non_holidays

    def remove_covid_for_holidays(self):
        """holidays関連のdfでコロナの影響を除去
        """
        # self.df_holidays = self.df_holidays.drop([f'2021-0{month}' for month in range(4, 10)], axis=0).drop(['2021-10'])
        self.df_holidays = self.df_holidays.drop([f'2021-0{month}' for month in range(4, 10)], axis=0)
        # self.df_non_holidays = self.df_non_holidays.drop([f'2021-0{month}' for month in range(4, 10)], axis=0).drop(['2021-10'])
        self.df_non_holidays = self.df_non_holidays.drop([f'2021-0{month}' for month in range(4, 10)], axis=0)

    def add_mar_oct_to_apr_sep_for_holidays(self):
        """holiday関連のdfで3月と10月を4月と9月に合算
        """
        years = [year for year in range(self.START_YEAR, self.END_YEAR + 1)]

        for year in years:
            try:
                self.df_holidays.loc[f'{year}-04', 'Attendance'] += self.df_holidays.loc[f'{year}-03', 'Attendance']
            except:
                pass
            try:
                self.df_holidays.loc[f'{year}-09', 'Attendance'] += self.df_holidays.loc[f'{year}-10', 'Attendance']
            except:
                pass
            try:
                self.df_holidays.drop(f'{year}-03', inplace=True)
            except:
                pass
            try:
                self.df_holidays.drop(f'{year}-10', inplace=True)
            except:
                pass

            try:
                self.df_non_holidays.loc[f'{year}-04', 'Attendance'] += self.df_non_holidays.loc[f'{year}-03', 'Attendance']
            except:
                pass
            try:
                self.df_non_holidays.loc[f'{year}-09', 'Attendance'] += self.df_non_holidays.loc[f'{year}-10', 'Attendance']
            except:
                pass
            try:
                self.df_non_holidays.drop(f'{year}-03', inplace=True)
            except:
                pass
            try:
                self.df_non_holidays.drop(f'{year}-10', inplace=True)
            except:
                pass

    def get_monthly_att_all_for_holidays(self):
        """holidays関連のdfで月ごとの観客数を取得
        """
        df_monthly_att_all = pd.DataFrame()
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            df_monthly_att = self.get_monthly_att_for_holidays(year)
            df_monthly_att_all = pd.concat([df_monthly_att_all, df_monthly_att], axis=0)

        self.df_holidays = df_monthly_att_all

    def get_monthly_att_all_for_non_holidays(self):
        """holidays関連のdfで月ごとの観客数を取得
        """
        df_monthly_att_all = pd.DataFrame()
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            df_monthly_att = self.get_monthly_att_for_non_holidays(year)
            df_monthly_att_all = pd.concat([df_monthly_att_all, df_monthly_att], axis=0)

        self.df_non_holidays = df_monthly_att_all

    def get_monthly_att_for_holidays(self, year):
        """holidays関連のdfで月ごとの観客数を取得
        """
        # self.dfから指定した年のデータを抽出
        df_ = self.df_holidays[self.df_holidays['Year'] == year]

        df_monthly_att = df_.groupby('Month').agg({'Attendance': 'sum'}).astype(int)

        df_monthly_att.index = [str(year) + '-' + str(month) for month in df_monthly_att.index]
        df_monthly_att.index = pd.to_datetime(df_monthly_att.index, format='%Y-%m').strftime('%Y-%m')

        return df_monthly_att

    def get_monthly_att_for_non_holidays(self, year):
        """non_holidays関連のdfで月ごとの観客数を取得
        """
        # self.dfから指定した年のデータを抽出
        df_ = self.df_non_holidays[self.df_non_holidays['Year'] == year]

        df_monthly_att = df_.groupby('Month').agg({'Attendance': 'sum'}).astype(int)

        df_monthly_att.index = [str(year) + '-' + str(month) for month in df_monthly_att.index]
        df_monthly_att.index = pd.to_datetime(df_monthly_att.index, format='%Y-%m').strftime('%Y-%m')

        return df_monthly_att

    def is_holiday(self, df):
        # USの祝日を取得
        us_holidays = holidays.UnitedStates()
        # dateをdatetime型に変換
        dates = df['Date'].apply(lambda x: x if isinstance(x, datetime.datetime) else datetime.datetime.strptime(x, '%Y-%m-%d'))

        holidays_list = []
        # 土日祝日または金曜日の場合は1を返す
        for date in dates:
            if date.weekday() >= 4 or date in us_holidays: # 4は金曜日。金曜日も動員数が多いらしいので金曜日も含むように。
                holidays_list.append(1)
            else:
                holidays_list.append(0)
        df['is_holiday'] = holidays_list

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

    def merge_new_df(self, df):
        """dfを受け取り、self.dfに追加する

        Args:
            df (_type_): add_up, covid_removedに応じたデータフレーム
        """
        # self.df_monthly_att_all_addup_covid_removedに年月をキーとして、dfを追加
        self.df_monthly_att_all_addup_covid_removed.reset_index(inplace=True)
        self.df_monthly_att_all_addup_covid_removed.rename(columns={'index': 'Date'}, inplace=True)
        self.df_monthly_att_all_addup_covid_removed['Date'] = pd.to_datetime(self.df_monthly_att_all_addup_covid_removed['Date'], format='%Y-%m')
        self.df_monthly_att_all_addup_covid_removed = pd.merge(self.df_monthly_att_all_addup_covid_removed, df, on='Date', how='left')
        self.df_monthly_att_all_addup_covid_removed['Date'] = self.df_monthly_att_all_addup_covid_removed['Date'].dt.strftime('%Y-%m')
        self.df_monthly_att_all_addup_covid_removed.set_index('Date', inplace=True)
        self.df_monthly_att_all_addup_covid_removed.index.name = None
        return self.df_monthly_att_all_addup_covid_removed

    def completion_outlier(self):
        """2022年4月の外れ値を補完する関数
        - 補完方法を指定して補完するようにしたい。
            - with_mean : 過去の4月の平均値で補完
            - with_predict : 各年の5-9月の観客者数から、4月の観客者数を予測して補完
            - with_predict_with_mean : with_predictの結果をさらに過去の4月の平均値で補完
            - with_median : 過去の4月の中央値で補完
        """
        if self.how_completion_outlier == 'with_mean':
            df = self.df_monthly_att_all_addup_covid_removed.copy()
            # 過去の4月の観客者数のみを抽出
            # 2023年4月は介入後のため、性質が異なるので除いたほうが良いと判断
            april_df = df[df.index.str.contains('04')]
            # 2023年4月を除外
            april_df = april_df.drop('2023-04')
            mean_value = april_df.loc['2013-04':'2019-04', 'Attendance'].mean()
            df.loc['2022-04', 'Attendance'] = int(mean_value)
            self.df_monthly_att_all_addup_covid_removed = df
            print(f'2022年4月は平均値{int(mean_value)}で補完しました。')

        elif self.how_completion_outlier == 'with_predict':
            self.predict_april_attendance()

        elif self.how_completion_outlier == 'with_median':
            df = self.df_monthly_att_all_addup_covid_removed.copy()
            # 過去の4月の観客者数のみを抽出
            april_df = df[df.index.str.contains('04')]
            # 2023年4月を除外
            april_df = april_df.drop('2023-04')
            median_value = april_df.loc['2013-04':'2019-04', 'Attendance'].median()
            df.loc['2022-04', 'Attendance'] = median_value
            self.df_monthly_att_all_addup_covid_removed = df
            print(f'2022年4月は中央値{int(median_value)}で補完しました。')


    def predict_april_attendance(self):
        """各年の5-9月の観客者数から、4月の観客者数を予測して補完
        """
        df = self.df_monthly_att_all_addup_covid_removed.copy()
        df_tmp = df[df.index < '2022-04-01']

        # SARIMAを用いて、2022年4月のデータを予測
        dict_param = self.optim_param_sarimax(df_tmp)
        order=(dict_param['order_p'], dict_param['d_order'], dict_param['ma_order'])
        seasonal_order=(dict_param['seasonal_ar_order'], dict_param['seasonal_d_order'], dict_param['seasonal_ma_order'], 6)

        model = SARIMAX(df_tmp, order=order, seasonal_order=seasonal_order)
        results = model.fit()

        pred = results.predict(start='2022-04', end='2022-04')
        # predを図示
        # plt.figure(figsize=(12, 8))
        # plt.title('Predicted Attendance')
        # plt.xlabel('Month')
        # plt.xticks(rotation=90)
        # plt.ylabel('Attendance')
        # plt.plot(df, color='red')
        # plt.plot(pred, color='blue')
        # plt.legend(['Data Points', 'Predicted Attendance'])
        # plt.show()
        pred = int(pred.values[0])

        df.loc['2022-04'] = pred
        self.df_monthly_att_all_addup_covid_removed = df

        print(f"AIC:{results.aic}")
        print(f'2022年4月は予測モデルで補完しました。値:{pred}')

    def optim_param_sarimax(self, df, n_trials=300):
        """SARIMAXモデルのパラメータをOptunaで最適化する

        Args:
            n_trials (int, optional): 試行回数。増やすとより良いパラメータの組み合わせを見つけられるが、計算時間も増加する Defaults to 100.
        """
        seed = 0
        def objective(trial):
            # SARIMAXモデルのパラメータを設定
            order=(
                trial.suggest_int('order_p', 0, 3),
                trial.suggest_int('d_order', 0, 2),
                trial.suggest_int('ma_order', 0, 3)
            )
            seasonal_order=(
                trial.suggest_int('seasonal_ar_order', 0, 3),
                trial.suggest_int('seasonal_d_order', 0, 2),
                trial.suggest_int('seasonal_ma_order', 0, 3))
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    warnings.filterwarnings('ignore', category=ValueWarning)
                    warnings.filterwarnings('ignore', category=UserWarning)

                    model = SARIMAX(df,
                                    order=order, seasonal_order=seasonal_order)
                    model_fit = model.fit(disp=False)
                    return model_fit.aic
            except Exception as e:
                return float('inf')

        # Optunaによる最適化
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        # 最適なパラメータを返す
        return study.best_params