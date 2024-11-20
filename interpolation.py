import os
from utils import interpolate_IDW, interpolate_kriging

root = "./collect_points/days/"
points = [f"{root}{point}" for point in os.listdir(root)]


for point_path in points:
    # Only intepolate files with more that 3 points
    if int(point_path.split("/")[-1].split("_")[0]) > 3:
        interpolate_kriging(point_path, "./kriging_results_per_days")

for point_path in points:
    if int(point_path.split("/")[-1].split("_")[0]) > 3:
        interpolate_IDW(point_path, "./idw_results_per_days")
