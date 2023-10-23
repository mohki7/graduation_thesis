import requests
import pandas as pd
import statsapi
from datetime import datetime
from tqdm import tqdm
import time

class MLBDataFetcher:
    def __init__(self):
        """MLB Stats APIからデータを取得するクラス
        """
        self.BASE_URL = "http://statsapi.mlb.com/api/v1/schedule"

    def create_initial_dataframe(self, start_date, end_date):
        """初期のデータフレームを作成する関数

        Args:
            start_date (str): 取得したいデータの開始日。例: "2019-01-01"
            end_date (str): 取得したいデータの終了日。例: "2019-12-31"

        Add:
            self.df : 初期のデータフレーム。列は["Game ID", "Date", "Home Team", "Away Team", "Stadium Name"]
        """
        params = {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date
        }

        response = requests.get(self.BASE_URL, params=params)
        data = response.json()

        # データフレームの初期化
        games = []
        # 日ごとの試合ごとにループ
        for date in tqdm(data["dates"]):
            for game in date["games"]:
                game_id = game["gamePk"]
                # レギュラーシーズンの試合のみを集めたいので、game_typeを取得
                try:
                    game_type = self.get_game_type(game_id)
                # HTTPErrorが発生した場合は、5秒待ってから次のリクエストを送る
                except requests.exceptions.HTTPError as e:
                    print(f"Error fetching data for game ID {game_id}: {e}")
                    time.sleep(5)  # Wait for 5 seconds before next request
                    continue

                # レギュラーシーズンの試合のみを取得
                if game_type == 'R':
                    game_date = datetime.strptime(date["date"], "%Y-%m-%d")
                    home_team = game["teams"]["home"]["team"]["name"]
                    away_team = game["teams"]["away"]["team"]["name"]
                    venue = game["venue"]["name"]

                    games.append([game_id, game_date, home_team, away_team, venue])

        self.df = pd.DataFrame(games, columns=["Game ID", "Date", "Home Team", "Away Team", "Stadium Name"])

    def get_game_type(self, game_id: int) -> str:
        """MLB Stats APIからその試合がレギュラーシーズンがどうかを取得する

        Args:
            game_id (int): gameId

        Returns:
            str: _description_
        """
        game_data = statsapi.get('game', {'gamePk': game_id})
        return game_data['gameData']['game']['type']

    def add_game_info_to_dataframe(self):
        """get_game_info()を使って、データフレームに試合の情報（観客者数・試合時間）をデータフレームに追加する関数

        Edit:
            self.df (_type_): データフレーム。列は["Game ID", "Date", "Home Team", "Away Team", "Stadium Name", "Attendance", "Game Time (minutes)"]
        """
        self.df["Attendance"] = [self.get_game_info(game_id)[0] for game_id in self.df["Game ID"]]
        self.df["Game Time (minutes)"] = [self.get_game_info(game_id)[1] for game_id in self.df["Game ID"]]

    def get_game_info(self, gameId: int) -> tuple:
        """試合の情報を取得する関数

        Args:
            gameId (int): gameId

        Returns:
            tuple: (attendance, game_time_minutes) : 試合の観客数, 試合時間(分)
                もし、試合の情報が取得できなかった場合は、(None, None)を返す
        """
        boxscore = statsapi.boxscore_data(gameId)
        try:
            attendance = next(item['value'] for item in boxscore['gameBoxInfo'] if item.get('label') == 'Att')
            attendance = int(attendance.replace('.', '').replace(',', ''))
            game_time_minutes = self.convert_to_minutes(next(item['value'] for item in boxscore['gameBoxInfo'] if item.get('label') == 'T').replace('.', ''))
            return attendance, game_time_minutes
        except StopIteration:
            return None, None

    def convert_to_minutes(self, time_str: str) -> int:
        """時間を分に変換する関数
        """
        # '(より後ろ'が存在する場合、それを除外
        if '(' in time_str:
            time_str = time_str.split('(')[0].strip()
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    def fetch_and_save(self, year):
        """指定した年のデータを取得して、CSVファイルとして保存する関数

        Args:
            year (str): 取得したいデータの年。例: "2019"
        """
        self.create_initial_dataframe(f"{year}-01-01", f"{year}-12-31")
        self.add_game_info_to_dataframe()
        self.df.to_csv(f"./data/mlb_{year}.csv", index=False)

    def main(self):
        years_str = input("年を入力してください(例:2019,2020):")
        years = years_str.split(",")  # カンマ区切りで複数の年を入力できるようにする

        for year in years:
            self.fetch_and_save(year)

if __name__ == "__main__":
    fetcher = MLBDataFetcher()
    fetcher.main()