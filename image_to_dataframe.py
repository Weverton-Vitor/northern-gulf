import rasterio
import pandas as pd
import numpy as np
import glob
import os

# Caminho para os arquivos de banda (ajuste conforme necessário)


path_to_bands = (
    "/media/weverton/D/Remote Sensing/Water Quality/mobile_bay/scenes/2007-05-08/crop/"
)

# Usar glob para pegar todos os arquivos .tif das bandas
tif_files = glob.glob(path_to_bands + "*.tif")
# Ordenar os arquivos de banda se necessário
tif_files.sort()

# Lista para armazenar os dados das bandas
series_bands = []

# Ler as bandas
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        band = src.read(1)  # Ler a banda (assumindo que cada .tif tem uma banda)
        band_name = tif_file.split("_")[-2]

        series_bands.append(
            pd.Series(band.flatten(), name=band_name, dtype=pd.Float32Dtype)
        )


# Criar o DataFrame
band_names = [
    f"Banda_{i+1}" for i in range(len(tif_files))
]  # Nomear as colunas com 'Banda_1', 'Banda_2', etc.


# print(series_bands)

df = pd.DataFrame(series_bands).transpose()

# Exibir o DataFrame
print(df.head())
print(df.tail())

df.to_csv("2007-05-08.csv", index=False)
