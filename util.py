import os
import pandas as pd
import re

script_dir = os.path.abspath(os.path.dirname(__file__))


def is_empty(df: pd.DataFrame) -> bool:
    return len(df.index) == 0


def make_valid_file_name(destination_name: any) -> str:
    dest = str(destination_name)
    return re.sub(r'\W', '', dest).upper()


def write_to_console(message):
    filler = "*" * len(message)
    print("\t********** {} ***********".format(filler))
    print("\t********** {} ***********".format(message))
    print("\t********** {} ***********".format(filler))
