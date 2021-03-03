import pandas as pd

from util import is_empty


class YearScaler:
    def __init__(self):
        self.min_year_timestamp = 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if is_empty(df):
            return df
        min_timestamp = pd.Timestamp(df["time"].min(), unit="s")
        self.min_year_timestamp = pd.Timestamp(year=min_timestamp.year, month=1, day=1, hour=0).timestamp()
        df = df.assign(time_scaled= df["time"] - self.min_year_timestamp)
        df = df.drop(columns=["time"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if is_empty(df):
            return df
        df = df.assign(time=df["time_scaled"] + self.min_year_timestamp)
        df = df.drop(columns=["time_scaled"])
        return df
