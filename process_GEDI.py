#!/usr/bin/env python3
# Script to process GEDI data and then export the processed GEDI data to a
# google drive folder
#
# Steven Lu
# March 8, 2024

from tqdm import tqdm
import argparse

# Google Earth Engine
import ee
ee.Authenticate()
ee.Initialize(project='biodiversity-jpl',
              opt_url="https://earthengine-highvolume.googleapis.com")


# Define constants for the GEE GEDI collection and US Pacific shapefile
GEDI_GEE_COLLECTION = 'LARSE/GEDI/GEDI02_A_002_MONTHLY'
US_PACIFIC_SHAPEFILE = 'projects/biodiversity-jpl/assets/USPacific_outlineBuferred'

# GEDI utils
list_valid_degradation = ee.List(
    [0, 3, 8, 10, 13, 18, 20, 23, 28, 30, 38, 40, 43, 48, 60, 63, 68]
)
list_valid_degradation = ee.Image.constant(list_valid_degradation)

list_tropic_region = ee.List([4, 5, 6])
list_tropic_region = ee.Image.constant(list_tropic_region)

list_nontropic_region = ee.List([1, 2, 3, 7])
list_nontropic_region = ee.Image.constant(list_nontropic_region)


def quality_function(gedi_data):
    return (
        gedi_data.updateMask(gedi_data.select("quality_flag").eq(1))
        .updateMask(
            gedi_data.select("degrade_flag")
            .eq(list_valid_degradation)
            .reduce(ee.Reducer.anyNonZero())
        )
        .updateMask(gedi_data.select("num_detectedmodes").gt(0))
        .updateMask(
            (
                gedi_data.select("region_class")
                .eq(list_tropic_region)
                .reduce(ee.Reducer.anyNonZero())
                .And(gedi_data.select("sensitivity").gt(0.90))
            ).Or(
                (
                    gedi_data.select("region_class")
                    .eq(list_nontropic_region)
                    .reduce(ee.Reducer.anyNonZero())
                    .And(gedi_data.select("sensitivity").gt(0.95))
                )
            )
        )
        .updateMask(
            (
                (
                    gedi_data.select("elev_lowestmode").subtract(
                        gedi_data.select("digital_elevation_model_srtm")
                    )
                ).abs()
            ).lt(80)
        )
        .updateMask(gedi_data.select("solar_elevation").lt(0))
        .updateMask(gedi_data.select("leaf_off_flag").eq(0))
        .updateMask(gedi_data.select("beam").gte(5))
    )


def export_to_google_drive(ee_image, filename, crs, resolution):
    task = ee.batch.Export.image.toDrive(
        image=ee_image,
        description=filename,
        crs=crs,
        scale=resolution,
        maxPixels=1e13,
        fileFormat='GeoTIFF',
        folder=f'GEDI_{resolution}m'
    )
    task.start()


def main(crs: str, resolution: float, year_range: list) -> None:
    # Loading in the USPacific ESRI shapefile
    # Note: The shapefile was previously uploaded to the GEE asset at
    # projects/biodiversity-jpl/assets/USPacific_outlineBuferred
    us_pacific_shape = ee.FeatureCollection(US_PACIFIC_SHAPEFILE).geometry()

    for year in tqdm(range(year_range[0], year_range[1]+1)):
        # Only using data from April to October to limit snow cover effects
        start_date = ee.Date.fromYMD(year, 4, 1)
        end_date = ee.Date.fromYMD(year, 10, 30)

        gedi = (ee.ImageCollection(GEDI_GEE_COLLECTION)
                .filterBounds(us_pacific_shape)
                .filter(ee.Filter.date(start_date, end_date))
                .filter(ee.Filter.calendarRange(4, 10, 'month'))
        )

        gedi_filt = gedi.map(quality_function).filterBounds(us_pacific_shape)
        gedi_filt = gedi_filt.select('rh98').mosaic().clip(us_pacific_shape)

        resampled_gedi = gedi_filt.reproject(crs=crs, scale=resolution)
        pixel_area = resampled_gedi.pixelArea()
        mask = pixel_area.gte(0.5 * resolution ** 2)
        gedi_filt = resampled_gedi.updateMask(mask).multiply(100).toInt16()

        export_to_google_drive(gedi_filt, f'gedi_rh98_{resolution}m_{year}',
                               crs, resolution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--crs', type=str, default='EPSG:4326',
                        help='Coordinate system for the output')
    parser.add_argument('-r', '--resolution', type=float, default=30.,
                        help='Output resolution in meters/pixel')
    parser.add_argument('-y', '--year_range', nargs=2, type=int, required=True)

    args = parser.parse_args()
    main(**vars(args))

