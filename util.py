import re
import numpy as np


def get_destination_file_name(destination_name):
    dest = str(destination_name)
    return re.sub(r'\W', '', dest).upper()


def write_to_console(message):
    filler = "*" * len(message)
    print("\t********** {} ***********".format(filler))
    print("\t********** {} ***********".format(message))
    print("\t********** {} ***********".format(filler))


# Compute port area identified by center point and radius
# df: DataFrame of vessels with the same destination
def compute_port_area(df, r_extend=0.1):
    at_anchor_df = df["Navigational status"] == "At anchor"
    lat_df = at_anchor_df["Latitude"].unique()
    long_df = at_anchor_df["Longitude"].unique()

    # center point of port based on anchor-coordinates
    center = np.array([lat_df.mean(), long_df.mean()])

    # radius based on deviations of anchor-coordinates from center point
    r_lat = max(abs(lat_df.min() - center[0]), center[0] - abs(lat_df.max()))
    r_long = max(abs(long_df.min() - center[1]), center[1] - abs(long_df.max()))
    r = np.array([r_lat*r_extend, r_long*r_extend])

    return center, r
