import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin


def kriging(file_path, save_path):
    # Create save folder
    os.makedirs(save_path, exist_ok=True)

    # Load data
    data = pd.read_csv(file_path)
    lats = data["Latitude"]
    lons = data["Longitude"]
    values = data["ChlA"]

    # interpolation grade
    gridx = np.linspace(lons.min(), lons.max(), 100)
    gridy = np.linspace(lats.min(), lats.max(), 100)

    # Kriging
    OK = OrdinaryKriging(lons, lats, values, variogram_model="power")
    z, ss = OK.execute("grid", gridx, gridy)

    # Resolution adjust
    resolution = (gridx[1] - gridx[0], gridy[1] - gridy[0])
    transform = from_origin(gridx.min(), gridy.max(), *resolution)

    # Criar o arquivo .tif
    with rasterio.open(
        f"{save_path}/{file_path.split('/')[-1].replace('.csv', '')}.tif",
        "w",
        driver="GTiff",
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype=z.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(z, 1)


def IDW(file_path, save_path):
    def distance_matrix(x0, y0, x1, y1):
        """Make a distance matrix between pairwise observations.
        Note: from <http://stackoverflow.com/questions/1871536>
        """

        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T

        d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
        d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

        # calculate hypotenuse
        return np.hypot(d0, d1)

    def simple_idw(x, y, z, xi, yi, power=1):
        """Simple inverse distance weighted (IDW) interpolation
        Weights are proportional to the inverse of the distance, so as the distance
        increases, the weights decrease rapidly.
        The rate at which the weights decrease is dependent on the value of power.
        As power increases, the weights for distant points decrease rapidly.
        """

        dist = distance_matrix(x, y, xi, yi)

        # In IDW, weights are 1 / distance
        weights = 1.0 / (dist + 1e-12) ** power

        # Make weights sum to one
        weights /= weights.sum(axis=0)

        # Multiply the weights for each interpolated point by all observed Z-values
        return np.dot(weights.T, z)

    def plot(x, y, z, grid):
        """Plot the input points and the result"""
        plt.figure(figsize=(15, 10))
        plt.imshow(
            grid,
            extent=(x.min(), x.max(), y.max(), y.min()),
            cmap="rainbow",
            interpolation="gaussian",
        )
        plt.scatter(x, y, c=z, cmap="rainbow", edgecolors="black")
        plt.colorbar()

    # Create save folder
    os.makedirs(save_path, exist_ok=True)

    # Carregar os dados CSV
    data = pd.read_csv(file_path)
    lats = data["Latitude"].values
    lons = data["Longitude"].values
    values = data["ChlA"].values

    # Criar grade de interpolação
    xi = np.linspace(lons.min(), lons.max(), 100)
    yi = np.linspace(lats.min(), lats.max(), 100)

    # Criar malha bidimensional
    xi, yi = np.meshgrid(xi, yi)

    # Achatar a malha em 1D
    xi_flat, yi_flat = xi.flatten(), yi.flatten()

    # Interpolar usando IDW
    grid1 = simple_idw(lats, lons, values, xi_flat, yi_flat, power=3)

    # Reformar para uma grade 2D
    grid1 = grid1.reshape(xi.shape)  # Aqui o xi.shape traz as dimensões corretas

    # resolution = (xi[1] - xi[0], yi[1] - yi[0])
    # transform = from_origin(xi.min(), yi.max(), *resolution)
    # Criar o arquivo .tif
    with rasterio.open(
        f"{save_path}/{file_path.split('/')[-1].replace('.csv', '')}.tif",
        "w",
        driver="GTiff",
        height=grid1.shape[0],
        width=grid1.shape[1],
        count=1,
        dtype=grid1.dtype,
        crs="EPSG:4326",
        # transform=transform,
    ) as dst:
        dst.write(grid1, 1)


root = "./collect_points/days/"
points = [f"{root}{point}" for point in os.listdir(root)]
for point_path in points:
    if int(point_path.split("/")[-1].split("_")[0]) > 10:
        kriging(point_path, "./kriging_result")


# root = "./collect_points/days/"
# points = [f"{root}{point}" for point in os.listdir(root)]
# for point_path in points:
#     if int(point_path.split("/")[-1].split("_")[0]) > 1:
#         IDW(point_path, "./idw_result")
