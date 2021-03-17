import pandas as pd

from util import is_empty


class DurationLabeler:
    def __init__(self):
        self.min_year_timestamp = 0
        self.arrival_time = 0

    def fit_transform(self, df: pd.DataFrame, arrival_time: int) -> pd.DataFrame:
        if is_empty(df):
            return df
        min_timestamp = pd.Timestamp(df["time"].min(), unit="s")
        self.min_year_timestamp = pd.Timestamp(year=min_timestamp.year, month=1, day=1, hour=0).timestamp()
        self.arrival_time = arrival_time
        df = df.assign(time_scaled=df["time"] - self.min_year_timestamp)
        # data can be labeled
        print("applying arrival time {} to df: e.g. {}".format(arrival_time, df.iloc[0]["time"]))
        if arrival_time > 0:
            df = df.assign(label=arrival_time - df["time"])
        else:
            df = df.assign(label=-1)
        df = df.drop(columns=["time"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if is_empty(df):
            return df
        df = df.assign(time=df["time_scaled"] + self.min_year_timestamp)
        df = df.drop(columns=["time_scaled"])
        return df
