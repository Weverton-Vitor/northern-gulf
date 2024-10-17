import rasterio
import os

from rasterio.warp import calculate_default_transform, Resampling
from rasterio.windows import from_bounds


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


root = "/media/weverton/D/Remote Sensing/Water Quality/mobile_bay/scenes/2007-05-08/"

tifs_to_crop = os.listdir(root)
tifs_to_crop = [f"{root}{path}" for path in tifs_to_crop if "SR" in path]
reference_tif = "/media/weverton/D/Remote Sensing/Water Quality/mobile_bay/scenes/2007-05-08/2007-05-08-00:00_2007-05-08-23:59_Landsat_4-5_TM_L2_ChlA.TIF"

# Executa a função
for tif in tifs_to_crop:
    crop_tif_based_on_another(tif, reference_tif, tif)
