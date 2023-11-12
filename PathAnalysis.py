import pandas as pd
import numpy as np
from preprocess import Att_Analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf

class PathAnalysis:
    """経路分析を行うクラス"""

    def __init__(self, intervention='2023-04'):
        self.att_df = Att_Analysis(is_remove_covid=True, is_addup=True).get_monthly_all_df().reset_index()
        self.game_df = pd.read_csv("./data/monthly_average_game_time.csv")
        self.df = None
        self.intervention = intervention
        self.model1 = None

    def prepare_df(self):
        """monthly_average_game_time.csvと観客者数のデータを結合する
        """
        self.att_df.rename(columns={"index": "date"}, inplace=True)

        # game_dfのindexをatt_dfに合わせる
        self.game_df = self.game_df.set_index('Unnamed: 0').rename_axis('date').reset_index()
        self.game_df['date'] = pd.to_datetime(self.game_df['date'], format='%Y/%m/1')
        self.game_df['date'] = self.game_df['date'].dt.strftime('%Y-%m')

        # データの結合
        self.df = pd.merge(self.att_df, self.game_df, on='date')

        # RuleChange列を追加
        self.df['RuleChange'] = 0
        self.df.loc[self.df['date'] >= f'{self.intervention}', 'RuleChange'] = 1

    def fit(self):
        """経路分析を行う
        """
        if self.df is None:
            self.prepare_df()
        # model1: ルール改正が試合時間に与える影響
        self.model1 = smf.ols(formula='Q("Game Time (minutes)") ~ Q("RuleChange")', data=self.df).fit()

        # model2: 試合時間が観客者数に与える影響
        self.model2 = smf.ols(formula='Q("Attendance") ~ Q("Game Time (minutes)")', data=self.df).fit()

        # model3: ルール改正が観客者数に与える影響
        self.model3 = smf.ols(formula='Q("Attendance") ~ Q("RuleChange")', data=self.df).fit()

    def show_summary(self):
        """fitした結果を表示する
        """
        if self.model1 is None:
            self.fit()
        return self.model1.summary(), self.model2.summary(), self.model3.summary()

    def plot_model(self, model):
        """modelをプロットする
        """
        return sm.graphics.plot_partregress_grid(model)


    def plot_models(self):
        """model1, 2, 3の結果をプロットする
        """
        if self.model1 is None:
            self.fit()
        return self.plot_model(self.model1), self.plot_model(self.model2), self.plot_model(self.model3)