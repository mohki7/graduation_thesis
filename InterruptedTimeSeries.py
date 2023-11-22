import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler
import warnings

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
# from .autonotebook import tqdm as notebook_tqdm

class ITS:
    """中断時系列分析を行うクラス
    """
    def __init__(self, df, intervention, method, interaction=False, period=6, order=3, optim_params_periodic_ols=False):
        """
        Args:
            DataFrame関連
                df (DataFrame): 中断時系列分析を行うためのデータフレーム。インデックスは日付、列は観客数
                df_before (DataFrame): 介入前のデータフレーム。dfと同じ形式
                df_after (DataFrame): 介入後のデータフレーム。dfと同じ形式
                df_its (DataFrame): OLSで分析するためのデータフレーム。t, xt, t*xtの列を持つ（時間のインデックス、level change、slope change）
                df_period (DataFrame): Periodic OLSで分析するためのデータフレーム。t, xt, t*xt, sin(2πt/period), cos(2πt/period)の列を持つ（時間のインデックス、level change、slope change、sin、cos）

            モデルの設定
                全体
                    intervention (string): 介入が行われた日付。日付の形式は'YYYY-MM'
                    method (string): 中断時系列分析に用いる手法。'OLS', 'SARIMA', 'ARIMA', 'Periodic OLS'から選択
                    interaction (bool, optional): 交互作用項を考慮するかどうか。time since startとlevel changeの。Defaults to False.
                    model: モデルを格納する
                    model_name: モデルの名前を格納する

                Periodic OLS
                    period (int, optional): 周期性のパラメータ。何期で1周期とするか。Defaults to 6.（4月から9月までなので6でちょうどよい）
                    order (int, optional): フーリエで何次元まで使うかどうかのパラメータ。Defaults to 3.
                    optim_params_periodic_ols (bool, optional): パラメータを最適化するかどうか。Defaults to False.

                OLS
                    variables(list): OLSで用いる変数のリスト。['time since start', 'level change', 'slope change']がデフォルト。t, xt, t*xt

        Functions:
            前処理関連
                separate_data(): 介入前と介入後のデータに分ける
                prepare_data(): モデルで学習できるようにデータを準備する。OLS用、交互作用考慮
                prepare_data_for_period_ols(): Periodic OLSのためのデータを準備する。t, xt, t*xt, sin, cosの列を持つデータフレームを作成し、self.df_periodに格納する #! これ、prepare_data()に統合できたら綺麗だよね


            描画関連
                plot_data(): 介入前と介入後のデータをプロットする。ただただdfをプロットするだけ
                plot_predict(alpha: 何%の信頼区間を描画するか,
                            is_counterfactual: 反事実も描画するかどうか,
                            is_prediction_std: 学習・予測の信頼区間を表示するかどうか): モデルで学習・予測した結果をプロット。
                calc_counterfactual(): 反実仮想を計算する。level changeとslope changeを0にしたデータフレームを作成し、モデルで予測する。もし介入してなかったら、どうなっていたかを出力

            学習関連
                fit(): 中断時系列分析に用いる手法を選択し、学習する。とりあえずfit()しておけば、model.methodに基づいて設定したモデルで学習する
                fit_ols(): 線形回帰で中断時系列分析を行う
                fit_slope_after_intervention(): 介入後の傾き（β_1 + β_3）についてsummaryを出す。
                fit_sarima(): SARIMAで中断時系列分析を行う
                fit_arima(): ARIMAで中断時系列分析を行う
                fit_periodic_ols(): Periodic OLSで中断時系列分析を行う

            モデル関連
                Periodic OLS
                    optim_params_periodic_ols(): 周期回帰に使うperiodとorderをR^2スコアが最大になるようにパラメータを最適化する。#! これは後でちゃんと作ろう。今は手動でパラメータを決めている

            結果確認
                show_summary(): とりあえずこれを呼び出せば、model.methodに基づいて設定したモデルのsummaryを出す。
        """
        self.df = df # インスタンスに渡された元のデータフレーム
        self.df_before = None # 介入前のデータフレーム
        self.df_after = None # 介入後のデータフレーム
        self.df_its = None # 分析用のデータフレーム
        self.df_period = None # Periodic OLSの分析のためのデータフレーム

        self.intervention = intervention
        self.method = method
        self.interaction = interaction

        self.period = period
        self.order = order

        self.model = None # モデルを格納する
        self.model_name = None # モデルの名前を格納する

        self.separate_data()
        self.variables = ['time since start', 'level change', 'slope change'] # t, xt, t*xt

        self.optim_params_periodic_ols=optim_params_periodic_ols

    def separate_data(self):
        """介入前と介入後のデータに分ける
        """
        self.df_before = self.df[self.df.index < self.intervention]
        self.df_after = self.df[self.df.index >= self.intervention]
        return self.df_before, self.df_after

    def plot_data(self):
        """介入前と介入後のデータをプロットする
        """
        plt.figure(figsize=(12, 8))
        plt.title('Monthly Attendance')
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        plt.scatter(self.df_before.index, self.df_before['Attendance'], color='red', label='Monthly Attendance (before intervention)')
        plt.scatter(self.df_after.index, self.df_after['Attendance'], color='blue', label='Monthly Attendance (after intervention)')
        plt.plot(self.df_before['Attendance'], color='red')
        plt.plot(self.df_after['Attendance'], color='blue')
        plt.legend()
        plt.show()

    def prepare_data(self):
        """データフレームを準備する。介入前後のダミー変数、月のインデックス、そして介入後に1ずつ増えていく列を追加
        """

        if self.interaction:
            self.variables.append('level change * time since start')
        self.df_its = self.df
        self.df_its[self.variables[0]] = [i for i in range(0, self.df_its.shape[0])] # time since start
        # self.interventionが何番目の行かを取得
        self.intervention_index = self.df_its.index.get_loc(self.intervention)
        # 介入前後のダミー変数を追加
        # self.intervention_index以降は1、それ以前は0
        # 短期的な影響を見る level change
        self.df_its[self.variables[1]] = [0 if i < self.intervention_index else 1 for i in range(self.df_its.shape[0])] # level change
        # 介入後は1ずつ増えていく列を追加
        # 長期的な影響を見る slope change
        self.df_its[self.variables[2]] = [i - self.intervention_index if i >= self.intervention_index else 0 for i in range(self.df_its.shape[0])] # slope change

        # 交互作用項を追加
        if self.interaction:
            self.df_its[self.variables[3]] = self.df_its[self.variables[1]] * self.df_its[self.variables[0]] # level change * time since start
        return self.df_its

    def fit_slope_after_intervention(self):
        """介入後の傾きについての列を追加（β_1 + β_3）
        データフレームに追加し、介入後の傾きが有意かを確認する
        x*txが有意であれば、介入後の傾きは有意であると言える
        """
        self.prepare_data()
        # 介入後の傾きについての列を追加（β_1 + β_3）
        self.df_its_1_3 = self.df_its.copy(deep=True)
        self.df_its_1_3['t(1-xt)'] = self.df_its[self.variables[0]] * (1- self.df_its[self.variables[1]])
        self.df_its_1_3['t*xt'] = self.df_its[self.variables[0]] * self.df_its[self.variables[1]]

        X = self.df_its_1_3[['t(1-xt)', 't*xt', self.variables[1]]]
        y = self.df_its_1_3['Attendance']

        mod2 = sm.OLS(y, sm.add_constant(X))
        res2 = mod2.fit()
        print(res2.summary())

    def fit(self):
        """中断時系列分析に用いる手法を選択する
        Option:
            OLS: 線形回帰
            Periodic OLS: 周期回帰
            SARIMA: SARIMA
            ARIMA: ARIMA
        """
        if self.method == 'OLS':
            self.fit_ols()
        elif self.method == 'Periodic OLS':
            self.fit_periodic_ols()
        elif self.method == 'SARIMA':
            self.fit_sarima()
        elif self.method == 'ARIMA':
            self.fit_arima()
        else:
            print('Please select method from OLS, SARIMA, ARIMA')

    def fit_ols(self):
        """線形回帰で中断時系列分析を行う
        """
        if self.df_its is None:
            self.prepare_data()

        X = sm.add_constant(self.df_its.reset_index()[[self.variables[0], self.variables[1], self.variables[2]]])
        if self.interaction:
            X = sm.add_constant(self.df_its.reset_index()[[self.variables[0], self.variables[1], self.variables[2], self.variables[3]]])
        y = self.df_its.reset_index()['Attendance']
        self.model = sm.OLS(y, X).fit()
        self.model_name = 'OLS'

    def prepare_data_for_period_ols(self):
        """周期回帰のためのデータを用意する
        """
        if self.df_its is None:
            self.prepare_data()
        self.df_period = self.df_its.copy(deep=True)

        # 周期period, order次までのフーリエ級数を用意する
        fourier = Fourier(period=self.period, order=self.order)

        # データをDeteministicProcessにより生成
        dp = DeterministicProcess(
            index=self.df_period.index,
            order=0,
            period=self.period,
            # fourier=fourier,
            drop=True,
            constant=False,
            additional_terms=[fourier]
        )

        H = dp.in_sample()

        self.df_period = pd.concat([self.df_period, H], axis=1)

    def optim_params_period_ols(self):
        """周期回帰に使う最適パラメータを探索する
        self.period, self.orderを最適化する
        R^2値が最大となるようにしたい
        #! これは後でかな。
        """
        return 6, 3
        # return period, order

    def fit_periodic_ols(self):
        """周期回帰で中断時系列分析を行う
        """
        if self.df_period is None:
            self.prepare_data_for_period_ols()

        # 最適パラメータを探索
        if self.optim_params_periodic_ols:
            self.period, self.order = self.optim_params_periodic_ols()

        row_list = [self.variables[i] for i in range(len(self.variables))]
        # 'sin(1,6)', cos(1,6)', 'sin(2,6)', 'cos(2,6)', 'sin(3,6)', 'cos(3,6)'のように列名を作成
        sin_cos_list = [f'sin({i},{self.period})' for i in range(1, self.order+1)] + [f'cos({i},{self.period})' for i in range(1, self.order+1)]
        row_list.extend(sin_cos_list)
        X = sm.add_constant(self.df_period.reset_index()[row_list])
        y = self.df_period.reset_index()['Attendance']

        # モデルを作成
        self.model = sm.OLS(y, sm.add_constant(X)).fit()
        # 訓練
        self.model_name = 'Periodic OLS'

    def fit_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """SARIMAで中断時系列分析を行う

        Args:
            order (tuple, optional): _description_. Defaults to (1, 1, 1).
            seasonal_order (tuple, optional): _description_. Defaults to (1, 1, 1, 12).

        Returns:
            _type_: _description_
        """
        self.model_name = "SARIMA"

        self.model = SARIMAX(self.df_before['Attendance'], order=order, seasonal_order=seasonal_order) #? 他のパラメータ、orderとseasonal_orderとは？どうやって決める？
        results = self.model.fit(disp=False) #? dispって何？
        return results

    def fit_arima(self):
        self.model_name = "ARIMA"
        return print('ARIMA')

    def show_summary(self):
        """結果を表示する
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()

        return self.model.summary()

    def plot_predict(self, alpha=0.05, is_counterfactual=False, is_prediction_std=False):
        """予測結果を図示する
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()

        # 予測結果を取得
        pred = self.model.predict()

        # 予測結果をプロット
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 8))
        plt.title('Predicted Monthly Attendance')
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        plt.plot(self.df.index, self.df['Attendance'], color='red')
        plt.scatter(self.df.index, self.df['Attendance'], color='red', label='Monthly Attendance (before intervention)')
        plt.plot(pred, color='blue', label='Predicted Monthly Attendance')

        if is_counterfactual:
            # 反実仮想を取得
            counterfactual = self.calc_counterfactual()
            # 介入後のデータのみをプロット
            plt.plot(counterfactual[self.intervention_index:], color='green', label='Counterfactual Monthly Attendance')
            # plt.plot(counterfactual, color='green', label='Counterfactual Monthly Attendance')
        if is_prediction_std:
            # 信頼区間を取得
            st, data, ss2 = summary_table(self.model, alpha=alpha)
            y_predict_l, y_predict_u = data[:, 4:6].T
            plt.plot(y_predict_l, color='orange', linestyle='--', label=f'Lower {(1-alpha)*100}% Confidence Interval', alpha=0.5)
            plt.plot(y_predict_u, color='orange', linestyle='--', label=f'Upper {(1-alpha)*100}% Confidence Interval', alpha=0.5)


            if is_counterfactual:
                cf_predict_l = counterfactual - (pred - y_predict_l)
                cf_predict_u = counterfactual - (pred - y_predict_u)
                plt.plot(cf_predict_l[self.intervention_index:], color='purple', linestyle='--', label=f'Lower {(1-alpha)*100}% Confidence Interval for Counterfactual', alpha=0.5)
                plt.plot(cf_predict_u[self.intervention_index:], color='purple', linestyle='--', label=f'Upper {(1-alpha)*100}% Confidence Interval for Counterfactual', alpha=0.5)

        plt.axvline(self.intervention, color='black', linestyle='-.', label='Intervention Date')
        plt.legend()
        plt.show()

    def calc_counterfactual(self):
        """反実仮想を計算する
        """
        if self.df_its is None:
            self.prepare_data()

        if self.df_period is None:
            self.prepare_data_for_period_ols()

        if self.method == 'OLS':
            cf_data = self.df_its[[self.variables[0], self.variables[1], self.variables[2]]].copy(deep=True)
            if self.interaction:
                cf_data = self.df_its[[self.variables[0], self.variables[1], self.variables[2], self.variables[3]]].copy(deep=True)

        if self.method == 'Periodic OLS':
            cf_data = self.df_period.drop(columns=['Attendance'])

        cf_data['level change'] = 0
        cf_data['slope change'] = 0

        # 反事実を予測
        cf_data.insert(0, 'cep', [1]*len(cf_data)) # 定数の列を追加
        y_cf_predict = self.model.predict(cf_data)

        return y_cf_predict


class MITS:
    """中断時系列分析を行うクラス。複数の介入点を設定できる
    """
    def __init__(self, df, interventions, method, interaction=False, period=6, order=3, optim_params_periodic_ols=False, optim_params_sarimax=False, optim_params_arimax=False, seed=0):
        """
        Args:
            DataFrame関連
                df (DataFrame): 中断時系列分析を行うためのデータフレーム。インデックスは日付、列は観客数
                df_before (DataFrame): 介入前のデータフレーム。dfと同じ形式
                df_after (DataFrame): 介入後のデータフレーム。dfと同じ形式
                df_its (DataFrame): OLSで分析するためのデータフレーム。t, xt, t*xtの列を持つ（時間のインデックス、level change、slope change）
                df_period (DataFrame): Periodic OLSで分析するためのデータフレーム。t, xt, t*xt, sin(2πt/period), cos(2πt/period)の列を持つ（時間のインデックス、level change、slope change、sin、cos）
                X (DataFrame): モデルに入力する説明変数。分析に用いた変数間の相関関係を確認するために用いる

            モデルの設定
                全体
                    interventions (list of string): 介入が行われた日付。複数指定可能。日付の形式は'YYYY-MM'
                    method (string): 中断時系列分析に用いる手法。'OLS', 'SARIMA', 'ARIMA', 'Periodic OLS'から選択
                    interaction (bool, optional): 交互作用項を考慮するかどうか。time since startとlevel changeの。Defaults to False.
                    model: モデルを格納する
                    model_name: モデルの名前を格納する

                Periodic OLS
                    period (int, optional): 周期性のパラメータ。何期で1周期とするか。Defaults to 6.（4月から9月までなので6でちょうどよい）
                    order (int, optional): フーリエで何次元まで使うかどうかのパラメータ。Defaults to 3.
                    optim_params_periodic_ols (bool, optional): パラメータを最適化するかどうか。Defaults to False.

                OLS
                    variables(list): OLSで用いる変数のリスト。['time since start', 'level change', 'slope change']がデフォルト。t, xt, t*xt

        Functions:
            前処理関連
                separate_data(): 介入前と介入後のデータに分ける
                prepare_data(): モデルで学習できるようにデータを準備する。OLS用、交互作用考慮
                prepare_data_for_period_ols(): Periodic OLSのためのデータを準備する。t, xt, t*xt, sin, cosの列を持つデータフレームを作成し、self.df_periodに格納する #! これ、prepare_data()に統合できたら綺麗だよね


            描画関連
                plot_data(): 介入前と介入後のデータをプロットする。ただただdfをプロットするだけ
                plot_predict(alpha: 何%の信頼区間を描画するか,
                            is_counterfactual: 反事実も描画するかどうか,
                            is_prediction_std: 学習・予測の信頼区間を表示するかどうか): モデルで学習・予測した結果をプロット。
                calc_counterfactual(): 反実仮想を計算する。level changeとslope changeを0にしたデータフレームを作成し、モデルで予測する。もし介入してなかったら、どうなっていたかを出力

            学習関連
                fit(): 中断時系列分析に用いる手法を選択し、学習する。とりあえずfit()しておけば、model.methodに基づいて設定したモデルで学習する
                fit_ols(): 線形回帰で中断時系列分析を行う
                fit_slope_after_intervention(): 介入後の傾き（β_1 + β_3）についてsummaryを出す。
                fit_sarima(): SARIMAで中断時系列分析を行う
                fit_arima(): ARIMAで中断時系列分析を行う
                fit_periodic_ols(): Periodic OLSで中断時系列分析を行う

            モデル関連
                Periodic OLS
                    optim_params_periodic_ols(): 周期回帰に使うperiodとorderをR^2スコアが最大になるようにパラメータを最適化する。#! これは後でちゃんと作ろう。今は手動でパラメータを決めている
                    calculate_hyperparameters_periodic_regression(): 周期回帰のハイパーパラメータを計算するメソッド

            結果確認
                show_summary(): とりあえずこれを呼び出せば、model.methodに基づいて設定したモデルのsummaryを出す。
                show_correration(): 変数間の相関係数を出力する

            #!工事中
                ・交互作用項の実装
                ・パラメータ最適化の実装
        """
        self.df = df # インスタンスに渡された元のデータフレーム
        self.df_its = None # 分析用のデータフレーム
        self.df_period = None # Periodic OLSの分析のためのデータフレーム
        self.X = None # モデルに入力する説明変数
        self.df_sarimax = None # SARIMAXの分析のためのデータフレーム
        self.df_arimax = None # ARIMAXの分析のためのデータフレーム


        self.intervention = interventions
        self.num_interventions = len(interventions)
        self.method = method
        self.interaction = interaction

        self.period = period
        self.order = order

        self.model = None # モデルを格納する
        self.model_name = None # モデルの名前を格納する

        # self.separate_data()
        # self.variables = ['time since start', 'level change', 'slope change'] # t, xt, t*xt

        self.optim_params_periodic_ols=optim_params_periodic_ols
        self.optim_params_sarimax = optim_params_sarimax
        self.optim_params_arimax = optim_params_arimax #! ここあとで、最適化するかどうかのパラメータは一つにまとめよう。
        self.seed = seed

    def separate_data(self):
        """介入ごとのデータに分ける
        Returns:
            list of DataFrame: 介入ごとのデータフレームを返す
        """
        # self.intervention+1個のデータフレームを作成し、すべて返す
        dfs = []
        for i in range(self.num_interventions+1):
            if i == 0:
                dfs.append(self.df[self.df.index < self.intervention[i]])
            elif i == self.num_interventions:
                dfs.append(self.df[self.df.index >= self.intervention[i-1]])
            else:
                dfs.append(self.df[(self.df.index >= self.intervention[i-1]) & (self.df.index < self.intervention[i])])
        return dfs

    def plot_data(self):
        """介入ごとのデータをプロットする
        """
        dfs = self.separate_data()
        plt.figure(figsize=(12, 8))
        plt.title('Monthly Attendance')
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        for i in range(self.num_interventions+1):
            plt.scatter(dfs[i].index, dfs[i]['Attendance'], label=f'Monthly Attendance (intervention {i})')
            plt.plot(dfs[i]['Attendance'])
        plt.legend()
        plt.show()

    def prepare_data(self):
        """データフレームを準備する。介入前後のダミー変数、月のインデックス、そして介入後に1ずつ増えていく列を追加。介入点ごとに変数を追加
        #! 交互作用項の実装なし
        """
        self.df_its = self.df
        # time indexは介入の数によらないので、ここで追加
        self.df_its[f'time since start'] = [j for j in range(1, self.df_its.shape[0]+1)] # time since start
        # 介入ごとに変数を追加
        for i in range(self.num_interventions):
            # self.interventionが何番目の行かを取得
            intervention_index = self.df_its.index.get_loc(self.intervention[i])
            # 介入前後のダミー変数を追加
            # self.intervention_index以降は1、それ以前は0
            # 短期的な影響を見る level change
            self.df_its[f'level change {i}'] = [0 if j < intervention_index else 1 for j in range(self.df_its.shape[0])] 
            # 介入後は1ずつ増えていく列を追加
            # 長期的な影響を見る slope change
            self.df_its[f'slope change {i}'] = [j - intervention_index if j >= intervention_index else 0 for j in range(1, self.df_its.shape[0]+1)]
        return self.df_its

    def fit_slope_after_intervention(self):
        """介入後の傾きについての列を追加（β_1 + β_3 + β_5 + ...）
        データフレームに追加し、介入後の傾きが有意かを確認する
        t*X_itが有意であれば、i番目の介入後の傾きは有意であると言える
        t*(X_i-1t - X_it)が有意であれば、i-1番目からi番目の間の傾きが有意であると言える
        例: 2022-04, 2023-04の2回の介入があった場合、t*X_2023-04t, t*(X_2022-04t - X_2023-04t)を追加
        t*X_2023-04tが有意であれば、2023-04の介入後の傾きは有意であると言える。
        t*(X_2022-04t - X_2023-04t)が有意であれば、2022-04から2023-04の間の傾きが有意であると言える。
        #! 周期回帰の場合はどうする？
        #! 交互作用項の実装なし
        """
        self.prepare_data()
        # 介入後の傾きについての列を追加
        self.df_its_slope = self.df_its.copy(deep=True)
        for i in range(self.num_interventions):
            if i == 0:
                self.df_its_slope[f't*(1 - level change {i})'] = (1-self.df_its_slope[f'level change {i}']) * self.df_its_slope['time since start']
            else:
                self.df_its_slope[f't*(level change {i-1} - level change {i})'] = (self.df_its_slope[f'level change {i-1}']-self.df_its_slope[f'level change {i}']) * self.df_its_slope['time since start']
                self.df_its_slope[f'level change {i-1} - level change {i}'] = (self.df_its_slope[f'level change {i-1}']-self.df_its_slope[f'level change {i}'])
            # 最後に追加
            if i == self.num_interventions-1:
                self.df_its_slope[f't*level change {i}'] = self.df_its_slope[f'level change {i}'] * self.df_its_slope['time since start']

        rows = ['t*(1 - level change 0)', f't*level change {self.num_interventions-1}', f'level change {self.num_interventions-1}']
        rows += [f't*(level change {i-1} - level change {i})' for i in range(1, self.num_interventions)]
        rows += [f'level change {i-1} - level change {i}' for i in range(1, self.num_interventions)]
        X = self.df_its_slope[rows]
        # X = self.df_its_1_3[['t(1-xt)', 't*xt', self.variables[1]]]
        y = self.df_its_slope['Attendance']

        mod2 = sm.OLS(y, sm.add_constant(X))
        res2 = mod2.fit()
        print(res2.summary())
        # return X

    def fit(self):
        """中断時系列分析に用いる手法を選択する
        Option:
            OLS: 線形回帰
            Periodic OLS: 周期回帰
            SARIMA: SARIMAX
            ARIMA: ARIMAX
        """
        if self.method == 'OLS':
            self.fit_ols()
        elif self.method == 'Periodic OLS':
            self.fit_periodic_ols()
        elif self.method == 'SARIMAX':
            self.fit_sarimax()
        elif self.method == 'ARIMAX':
            self.fit_arimax()
        elif self.method == "State Space Model":
            self.fit_state_space_model()
        else:
            print('Please select method from OLS, Periodic OLS, SARIMAX, ARIMAX, State Space Model')

    def fit_ols(self):
        """線形回帰で中断時系列分析を行う
        """
        if self.df_its is None:
            self.prepare_data()

        X = sm.add_constant(self.df_its.reset_index().drop(columns=['index', 'Attendance']))
        # if self.interaction:
        #     X = sm.add_constant(self.df_its.reset_index()[[self.variables[0], self.variables[1], self.variables[2], self.variables[3]]])
        y = self.df_its.reset_index()['Attendance']
        self.X = X
        self.model = sm.OLS(y, X).fit()
        self.model_name = 'OLS'

    def prepare_data_for_period_ols(self):
        """周期回帰のためのデータを用意する
        """
        if self.df_its is None:
            self.prepare_data()
        self.df_period = self.df_its.copy(deep=True)

        # 周期period, order次までのフーリエ級数を用意する
        fourier = Fourier(period=self.period, order=self.order)

        # データをDeteministicProcessにより生成
        dp = DeterministicProcess(
            index=self.df_period.index,
            order=0,
            period=self.period,
            # fourier=fourier,
            drop=True,
            constant=False,
            additional_terms=[fourier]
        )

        H = dp.in_sample()

        self.df_period = pd.concat([self.df_period, H], axis=1)

    def prepare_data_for_arimax(self):
        """ARIMAX分析用のデータを用意する
        """
        if self.df_its is None:
            self.prepare_data()
        self.df_arimax = self.df_its.copy(deep=True)

    def optim_param_sarimax(self, n_trials=300):
        """SARIMAXモデルのパラメータをOptunaで最適化する

        Args:
            n_trials (int, optional): 試行回数。増やすとより良いパラメータの組み合わせを見つけられるが、計算時間も増加する Defaults to 100.
        """
        if self.df_sarimax is None:
            self.prepare_data_for_sarimax()
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
                trial.suggest_int('seasonal_ma_order', 0, 3),
                6)
            # order=(
            #     trial.suggest_int('order_p', 0, 6),
            #     trial.suggest_int('d_order', 0, 6),
            #     trial.suggest_int('ma_order', 0, 6)
            # )
            # seasonal_order=(
            #     trial.suggest_int('seasonal_ar_order', 0, 6),
            #     trial.suggest_int('seasonal_d_order', 0, 6),
            #     trial.suggest_int('seasonal_ma_order', 0, 6), 6)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    warnings.filterwarnings('ignore', category=ValueWarning)
                    warnings.filterwarnings('ignore', category=UserWarning)
                    # warnings.filterwarnings('ignore', category=RuntimeWarning)
                    model = SARIMAX(self.df_sarimax['Attendance'], exog=self.df_sarimax.drop(columns=['Attendance', 'time since start']), # columns=['time since start']を追加するかどうかは不明
                                    order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    return model_fit.aic
            except Exception as e:
                return float('inf')

        # Optunaによる最適化
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        print(f"seed値:{self.seed}")

        # 最適なパラメータを返す
        return study.best_params

    def optim_param_arimax(self, n_trials=300):
        """ARIMAXモデルのパラメータをOptunaで最適化する

        Args:
            n_trials (int, optional): 試行回数。増やすとより良いパラメータの組み合わせを見つけられるが、計算時間も増加する Defaults to 100.
        """
        if self.df_arimax is None:
            self.prepare_data_for_arimax()
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
            # order=(
            #     trial.suggest_int('order_p', 0, 6),
            #     trial.suggest_int('d_order', 0, 6),
            #     trial.suggest_int('ma_order', 0, 6)
            # )
            # seasonal_order=(
            #     trial.suggest_int('seasonal_ar_order', 0, 6),
            #     trial.suggest_int('seasonal_d_order', 0, 6),
            #     trial.suggest_int('seasonal_ma_order', 0, 6))
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    warnings.filterwarnings('ignore', category=ValueWarning)
                    warnings.filterwarnings('ignore', category=UserWarning)
                    # warnings.filterwarnings('ignore', category=RuntimeWarning)
                    model = ARIMA(self.df_sarimax['Attendance'], exog=self.df_sarimax.drop(columns=['Attendance', 'time since start']), # columns=['time since start']を追加するかどうかは不明
                                    order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    return model_fit.aic
            except Exception as e:
                return float('inf')

        # Optunaによる最適化
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        print(f"seed値:{self.seed}")

        # 最適なパラメータを返す
        return study.best_params

    def prepare_data_for_sarimax(self):
        """SARIMAX分析用のデータを用意する
        """
        if self.df_its is None:
            self.prepare_data()
        self.df_sarimax = self.df_its.copy(deep=True)

    def optim_params_period_ols(self):
        """周期回帰に使う最適パラメータを探索する
        self.period, self.orderを最適化する
        R^2値が最大となるようにしたい
        #! これは後でかな。
        """
        return 6, 3
        # return period, order

    def fit_periodic_ols(self):
        """周期回帰で中断時系列分析を行う
        """
        if self.df_period is None:
            self.prepare_data_for_period_ols()

        # 最適パラメータを探索
        if self.optim_params_periodic_ols is True:
            self.period, self.order = self.optim_params_period_ols()
            print(f'最適化されたパラメータ: period={self.period}, order={self.order}')

        # row_list = [self.variables[i] for i in range(len(self.variables))]
        # 'sin(1,6)', cos(1,6)', 'sin(2,6)', 'cos(2,6)', 'sin(3,6)', 'cos(3,6)'のように列名を作成
        # sin_cos_list = [f'sin({i},{self.period})' for i in range(1, self.order+1)] + [f'cos({i},{self.period})' for i in range(1, self.order+1)]
        # row_list.extend(sin_cos_list)
        X = sm.add_constant(self.df_period.reset_index().drop(columns=['index', 'Attendance']))
        y = self.df_period.reset_index()['Attendance']
        self.X = X
        # モデルを作成
        self.model = sm.OLS(y, sm.add_constant(X)).fit()
        # 訓練
        self.model_name = 'Periodic OLS'

    def fit_sarimax(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """SARIMAで中断時系列分析を行う

        Args:
            order (tuple, optional): _description_. Defaults to (1, 1, 1).
            seasonal_order (tuple, optional): _description_. Defaults to (1, 1, 1, 12).

        Returns:
            _type_: _description_
        #! ここは未改修。self.df_beforeとかないし。
        #! 介入は複数ある時に未対応
        """
        self.model_name = "SARIMAX"
        if self.df_sarimax is None:
            self.prepare_data_for_sarimax()
        # 最適化するオプションがあった場合、optunaで最適化したパラメータを用いる
        if self.optim_params_sarimax is True:
            dict_param = self.optim_param_sarimax()
            order=(dict_param['order_p'], dict_param['d_order'], dict_param['ma_order'])
            seasonal_order=(dict_param['seasonal_ar_order'], dict_param['seasonal_d_order'], dict_param['seasonal_ma_order'], 6)

        self.model = SARIMAX(self.df_sarimax['Attendance'],
                             exog=self.df_sarimax.drop(columns=['Attendance', 'time since start']), # columns=['time since start']を追加するかどうかは不明
                             order=order,
                             seasonal_order=seasonal_order).fit(disp=False) #? 他のパラメータ、orderとseasonal_orderとは？どうやって決める？

    def fit_arimax(self, order=(1, 1, 1)):
        """SARIMAで中断時系列分析を行う

        Args:
            order (tuple, optional): _description_. Defaults to (1, 1, 1).
            seasonal_order (tuple, optional): _description_. Defaults to (1, 1, 1, 12).

        Returns:
            _type_: _description_
        #! ここは未改修。self.df_beforeとかないし。
        #! 介入は複数ある時に未対応
        """
        self.model_name = "ARIMAX"
        if self.df_arimax is None:
            self.prepare_data_for_arimax()
        # 最適化するオプションがあった場合、optunaで最適化したパラメータを用いる
        if self.optim_params_arimax is True:
            dict_param = self.optim_param_arimax()
            order=(dict_param['order_p'], dict_param['d_order'], dict_param['ma_order'])
            seasonal_order=(dict_param['seasonal_ar_order'], dict_param['seasonal_d_order'], dict_param['seasonal_ma_order'])

        self.model = ARIMA(self.df_arimax['Attendance'],
                             exog=self.df_arimax.drop(columns=['Attendance', 'time since start']), # columns=['time since start']を追加するかどうかは不明
                             order=order).fit() #? 他のパラメータ、orderとseasonal_orderとは？どうやって決める？
        self.model_name = "ARIMAX"


    def show_summary(self):
        """結果を表示する
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()
        # モデルがOLSやPeriodic OLSの場合はVIFも表示:
        if self.method == 'OLS' or self.method == 'Periodic OLS':
            print("VIF:", self.calc_vif())
        return self.model.summary()

    def plot_sarimax_params(self):
        """SARIMAXの最適化されたパラメータの妥当性を確認する
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()

        # 残差のプロット
        residuals = self.model.resid
        plt.figure(figsize=(12, 8))
        plt.plot(residuals)
        plt.xticks(rotation=90)
        plt.title('Residuals')
        plt.show()

        # 残差のACFとPACF
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        sm.graphics.tsa.plot_acf(residuals, lags=min(26, len(residuals)//2), ax=ax[0])
        sm.graphics.tsa.plot_pacf(residuals, lags=min(26, len(residuals)//2), ax=ax[1])
        plt.show()

        # モデル診断
        self.model.plot_diagnostics(figsize=(12, 8))
        plt.show()

    def plot_arimax_params(self):
        self.plot_sarimax_params()

    def plot_predict(self, alpha=0.05, is_counterfactual=False, is_prediction_std=False):
        """予測結果を図示する
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()

        # 予測結果を取得
        pred = self.model.predict()

        # 予測結果をプロット
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 8))
        plt.title('Predicted Monthly Attendance')
        plt.xlabel('Month')
        plt.ylabel('Attendance')
        plt.xticks(rotation=90)
        plt.plot(self.df.index, self.df['Attendance'], color='red')
        plt.scatter(self.df.index, self.df['Attendance'], color='red', label='Monthly Attendance (before intervention)')
        plt.plot(pred, color='blue', label='Predicted Monthly Attendance')

        if is_counterfactual:
            # 反実仮想を取得
            counterfactual = self.calc_counterfactual()
            # 介入後のデータのみをプロット
            for i in range(self.num_interventions):
                intervention_idx_ = self.df_its.index.get_loc(self.intervention[i])
                plt.plot(counterfactual[intervention_idx_:], color='green', label=f'{i} Counterfactual Monthly Attendance')
        if is_prediction_std:
            # 信頼区間を取得
            st, data, ss2 = summary_table(self.model, alpha=alpha)
            y_predict_l, y_predict_u = data[:, 4:6].T
            plt.plot(y_predict_l, color='orange', linestyle='--', label=f'Lower {(1-alpha)*100}% Confidence Interval', alpha=0.5)
            plt.plot(y_predict_u, color='orange', linestyle='--', label=f'Upper {(1-alpha)*100}% Confidence Interval', alpha=0.5)


            if is_counterfactual:
                cf_predict_l = counterfactual - (pred - y_predict_l)
                cf_predict_u = counterfactual - (pred - y_predict_u)
                for i in range(self.num_interventions):
                    intervention_idx_ = self.df_its.index.get_loc(self.intervention[i])
                    plt.plot(cf_predict_l[intervention_idx_:], color='purple', linestyle='--', label=f'Lower {(1-alpha)*100}% Confidence Interval for Counterfactual', alpha=0.5)
                    plt.plot(cf_predict_u[intervention_idx_:], color='purple', linestyle='--', label=f'Upper {(1-alpha)*100}% Confidence Interval for Counterfactual', alpha=0.5)

        for i in range(self.num_interventions):
            plt.axvline(self.intervention[i], color='black', linestyle='-.', label=f'Intervention Date {i}')

        plt.legend()
        plt.show()

    def calc_counterfactual(self):
        """反実仮想を計算する
        """
        if self.df_its is None:
            self.prepare_data()

        if self.df_period is None:
            self.prepare_data_for_period_ols()

        cf_data = None
        if self.method == 'OLS':
            cf_data = self.df_its.reset_index().drop(columns=['index', 'Attendance']).copy(deep=True)
            # if self.interaction:
            #     cf_data = self.df_its[[self.variables[0], self.variables[1], self.variables[2], self.variables[3]]].copy(deep=True)
        elif self.method == 'SARIMAX':
            cf_data = self.df_sarimax.drop(columns=['Attendance'])
        elif self.method == 'Periodic OLS':
            cf_data = self.df_period.drop(columns=['Attendance'])

        if cf_data is None:
            raise ValueError("Method must be one of 'OLS', 'SARIMAX', or 'Periodic OLS'")

        for i in range(self.num_interventions):
            cf_data[f'level change {i}'] = 0
            cf_data[f'slope change {i}'] = 0

        # 反事実を予測
        cf_data.insert(0, 'cep', [1]*len(cf_data)) # 定数の列を追加
        y_cf_predict = self.model.predict(cf_data)

        return y_cf_predict

    def calc_vif(self):
        """多重共線性の確認のためのVIFを計算する

        Returns:
            _type_: _description_
        """
        vif = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]
        return vif

    def show_correlation(self):
        """変数間の相関係数を出力する
        """
        if self.X is None:
            self.fit()
        plt.figure(figsize=(20, 16))  # 画像サイズを大きく設定
        sns.set(font_scale=2)  # 文字サイズを設定
        heatmap = sns.heatmap(self.X.corr(), annot=True, cmap='bwr', fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')  # X軸のラベルを回転
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)  # Y軸のラベルは回転させない
        plt.tight_layout()  # レイアウトの調整
        plt.show()  # ヒートマップを表示
        print("VIF:", self.calc_vif())  # 相関係数のテーブルも出力する場合はコメントアウトを解除

    def calculate_hyperparameters_periodic_regression(self):
        """周期回帰のハイパーパラメータをR^2値が最小になるように計算するメソッド

        Returns:
            _type_: 最適化されたperiod, order
        """
        best_score = -np.inf
        best_params = (0, 0)

        for period in range(1, 13):
            for order in range(1, 5):
                try:
                    self.period = period
                    self.order = order
                    self.fit_periodic_ols()
                    score = self.model.rsquared

                    if score > best_score:
                        best_score = score
                        best_params = (period, order)
                except:
                    continue

        self.period, self.order = best_params
        return best_params

    def fit_state_space_model(self):
        """状態空間モデルで中断時系列分析を行う
        """
        self.model_name="State Space Model"
        if self.df_its is None:
            self.prepare_data()

        self.model = sm.tsa.UnobservedComponents(
            self.df_its['Attendance'], # 観客者数
            trend=True, # トレンド項
            seasonal=6, # 季節性
            level='local level', # モデルのタイプ
            exog=self.df_its.drop(columns=['Attendance'])).fit() # 介入変数

    def plot_state_space_model(self):
        """状態空間モデルの結果をプロットする
        """
        if self.model is None:
            self.fit()
        # モデルの名前が現在のモデルと異なる場合も実行
        if self.model_name != self.method:
            self.fit()
        if self.method != "State Space Model":
            raise ValueError("Method must be 'State Space Model'")

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(self.df_its["Attendance"], label="Observations")
        plt.axvline(self.intervention[0], color='r', label="Intervention", linestyle='--')
        plt.xticks(rotation=90)
        ax.set(title='Attendance with Intervention', xlabel='Date', ylabel='Attendance')
        plt.legend()
        plt.show()

    def fit_hierarchical_bayesian_model(self):
        """階層ベイズモデルで中断時系列分析を行う
        """
        # ここに階層ベイズモデルの実装を追加します
        
