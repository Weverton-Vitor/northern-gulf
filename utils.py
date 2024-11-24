import glob
import os

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.windows import from_bounds


def img2df_separate_bands(path_bands: str, export_path: str) -> pd.DataFrame:
    tif_files = glob.glob(path_bands + "*.tif")
    tif_files.sort()

    # Lista para armazenar os dados das bandas
    series_bands = []

    # Ler as bandas
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            band = src.read(1)
            band_name = tif_file.split("_")[-2]

            print("Bandas: ", band_name)

            series_bands.append(
                pd.Series(band.flatten(), name=band_name, dtype=pd.Float32Dtype)
            )

    df_img = pd.DataFrame(series_bands).transpose()

    if export_path:
        df_img.to_csv(export_path, index=False)

    return df_img


def csv_to_array_points(
    csv_file_path: str, lat_field: str = "Latitude", lon_field: str = "Longitude"
) -> ee.FeatureCollection:
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file at {csv_file_path}")
        return None

    if not all(col in df.columns for col in [lon_field, lat_field]):
        print(f"Error: CSV file must have {lon_field} and {lat_field} columns.")
        return None

    features = []
    for index, row in df.iterrows():
        longitude = row[lon_field]
        latitude = row[lat_field]
        name = (
            row["name"] if "name" in df.columns else f"Point_{index + 1}"
        )  # Default name if not provided
        features.append(
            ee.Feature(ee.Geometry.Point(longitude, latitude), {"name": name})
        )
    return ee.FeatureCollection(features)


def extract_pixel_values(
    collection: ee.ImageCollection,
    site_coords_features: ee.FeatureCollection = None,
    csv_point_file_path: str = "",
    path_export: str = "",
    scale: int = 30,
):
    pt_collection = None

    if csv_point_file_path:
        pt_collection = csv_to_array_points(csv_file_path=csv_point_file_path)
    elif site_coords_features is not None:
        pt_collection = site_coords_features

    if pt_collection is None:
        raise Exception("No points to extract")

    def extract_pixel_values(feat):
        # Filtering images that contains points
        filter_collect = collection.filterBounds(feat.geometry())

        # Extract pixels of a specific image
        def reduce_image(img):
            reduced = img.reduceRegion(
                geometry=feat.geometry(), scale=scale, reducer=ee.Reducer.mean()
            )

            return ee.Feature(None, reduced).set(
                {
                    "name": feat.get("name"),
                    "lat": feat.geometry().coordinates().get(0),
                    "lon": feat.geometry().coordinates().get(1),
                }
            )

        return filter_collect.map(reduce_image)

    # Aplicar a função de extração a todos os pontos e achatar a coleção resultante
    output_data = pt_collection.map(extract_pixel_values).flatten()

    if path_export:
        task = ee.batch.Export.table.toDrive(
            collection=output_data,
            folder=path_export,
            fileNamePrefix=output_data.getInfo()["features"][0]["id"],
            fileFormat="CSV",
        )

        task.start()

    # Prepare data to return
    df_result = {
        key: [] for key in output_data.getInfo()["features"][0]["properties"].keys()
    }
    for feature in output_data.getInfo()["features"]:
        for key, value in feature["properties"].items():
            df_result[key].append(value)

    return pd.DataFrame(df_result)


def crop_tif_based_on_another(tif_to_crop, reference_tif, output_tif):
    # Abre o arquivo de referência (segundo .tif)
    with rasterio.open(reference_tif) as ref_tif:
        # Obtém os bounds, transformação e CRS do arquivo de referência
        ref_bounds = ref_tif.bounds
        ref_crs = ref_tif.crs

        # Abre o arquivo que será cropado (primeiro .tif)
        with rasterio.open(tif_to_crop) as src_tif:
            src_crs = src_tif.crs

            # Verifica se os CRS são diferentes e reprojeta o primeiro TIF se necessário
            if src_crs != ref_crs:
                transform, width, height = calculate_default_transform(
                    src_tif.crs, ref_crs, src_tif.width, src_tif.height, *src_tif.bounds
                )
                reprojected_tif = src_tif.read(
                    out_shape=(src_tif.count, height, width),
                    resampling=Resampling.bilinear,
                )
            else:
                reprojected_tif = src_tif.read()

            # Calcula a janela (bounding box) com base nos bounds do arquivo de referência
            window = from_bounds(
                ref_bounds.left,
                ref_bounds.bottom,
                ref_bounds.right,
                ref_bounds.top,
                src_tif.transform,
            )
            print(window)

            # Lê os dados do primeiro TIF usando a janela calculada
            cropped_data = src_tif.read(window=window)

            # Atualiza o transform para o novo tamanho
            transform = src_tif.window_transform(window)
            # Escreve o arquivo cropado em um novo .tif
            with rasterio.open(
                output_tif,
                "w",
                driver="GTiff",
                height=cropped_data.shape[1],
                width=cropped_data.shape[2],
                count=src_tif.count,
                dtype=cropped_data.dtype,
                crs=src_crs,
                transform=transform,
            ) as dst_tif:
                dst_tif.write(cropped_data)


def interpolate_kriging(
    file_path: str,
    save_path: str,
    lat_field: str = "Latitude",
    lon_field: str = "Longitude",
    target_field: str = "ChlA",
):
    # Create save folder
    os.makedirs(save_path, exist_ok=True)

    # Load data
    data = pd.read_csv(file_path)

    lats = data[lat_field]
    lons = data[lon_field]
    values = data[target_field]

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


def interpolate_IDW(
    file_path: str,
    save_path: str,
    lat_field: str = "Latitude",
    lon_field: str = "Longitude",
    target_field: str = "ChlA",
):
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
    lats = data[lat_field].values
    lons = data[lon_field].values
    values = data[target_field].values

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

    resolution = (xi[0, 1] - xi[0, 0], yi[1, 0] - yi[0, 0])

    transform = from_origin(xi.min(), yi.max(), *resolution)
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
        transform=transform,
    ) as dst:
        dst.write(grid1, 1)


if __name__ == "__main__":
    path_to_bands = "/media/weverton/D/Remote Sensing/Water Quality/mobile_bay/scenes/2006-06-06/crop/"
    df = img2df_separate_bands(path_bands=path_to_bands, export_path="2006-06-06.csv")
    print(df.head)
