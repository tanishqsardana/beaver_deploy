import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from streamlit_folium import folium_static
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, accuracy_score
import ee 
import os
import numpy as np
import pandas as pd
# import gdal
import tempfile
import rasterio

import json
from google.oauth2 import service_account


credentials_info = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"],
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
}

credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)

def S2_Export_for_visual(Dam_Collection):
    def extract_pixels(box):
        imageDate = ee.Date(box.get("Survey_Date"))
        StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")

        EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")

        boxArea = box.geometry()
        DateString = box.get("stringID")
        damId = box.get("id_property")
        DamStatus = box.get('Dam')
        DamDate = box.get('Damdate')
        DamGeo = box.get('Point_geo')
        S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        ## Add band for cloud coverage
        def add_cloud_mask_band(image):
            qa = image.select('QA60')

            # Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11

            # Both flags should be set to zero, indicating clear conditions.
            cloud_mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                qa.bitwiseAnd(cirrusBitMask).eq(0)
            )
            # Create a band with values 1 (clear) and 0 (cloudy or cirrus) and convert from byte to Uint16
            cloud_mask_band = cloud_mask.rename('cloudMask').toUint16()

            return image.addBands(cloud_mask_band)

        # Define the dataset
        S2_cloud_band = S2.map(add_cloud_mask_band)


        # Change band names
        oldBandNames = ['B2', 'B3', 'B4', 'B8','cloudMask']
        newBandNames = ['S2_Blue', 'S2_Green', 'S2_Red', 'S2_NIR','S2_Binary_cloudMask']#'S2_NDVI']

        S2_named_bands = S2_cloud_band.map(lambda image: image.select(oldBandNames).rename(newBandNames))

        # Define a function to add the masked image as a band to the images
        def addAcquisitionDate(image):
            date = ee.Date(image.get('system:time_start'))
            return image.set('acquisition_date', date)

        S2_cloud_filter = S2_named_bands.map(addAcquisitionDate)

        filteredCollection = S2_cloud_filter.filterDate(StartDate, EndDate).filterBounds(boxArea)

            
        def add_band(image):
            index = image.get("system:index")
            image_date = ee.Date(image.get('system:time_start'))
            image_month = image_date.get('month')
            image_year = image_date.get('year')
            cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
            # intersect = image.get('intersection_ratio')
            
            dataset = ee.Image('USGS/3DEP/10m')
            elevation_select = dataset.select('elevation')
            elevation = ee.Image(elevation_select)

            # Extract sample area from elevation
            point_geom = DamGeo

            # Extract elevation of dam location
            point_elevation = ee.Number(elevation.sample(point_geom, 10).first().get('elevation'))
            buffered_area = boxArea
            elevation_clipped = elevation.clip(buffered_area)

            # Create elevation radius around point to sample from
            point_plus = point_elevation.add(3)
            point_minus = point_elevation.subtract(5)        
            elevation_masked = elevation_clipped.where(elevation_clipped.lt(point_minus), 0).where(elevation_clipped.gt(point_minus), 1).where(elevation_clipped.gt(point_plus), 0)
            elevation_masked2 = elevation_masked.updateMask(elevation_masked.eq(1));

            # Add bands, create new "id" property to name the file, and clip the images to the ROI
            # Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect))\
            Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_"))\
                .set("Dam_id",damId)\
                .set("Dam_status",DamStatus)\
                .set("Image_month",image_month)\
                .set("Image_year",image_year)\
                .set("Area", boxArea)\
                .clip(boxArea)
            Full_image2 = Full_image.addBands(elevation_masked2)
            ## maybe add point geo as a property
            
        
            return Full_image2
            
        filteredCollection2 = filteredCollection.map(add_band)

        def calculate_cloud_coverage(image):
            cloud = image.select('S2_Binary_cloudMask')

            # Compute cloud coverage percentage using a simpler approach
            cloud_stats = cloud.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=image.geometry(),
                scale=10,
                maxPixels=1e9
            )
            
            clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
            cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)  # Invert the percentage

            return image.set('Cloud_coverage', cloud_coverage_percentage)


        filteredCloudCollection = filteredCollection2.map(calculate_cloud_coverage)
        # filteredCollection_overlap = filteredCloudCollection.filterMetadata('intersection_ratio', 'greater_than', 0.95)

        
        # Group by month and get the least cloudy image for each month
        def get_monthly_least_cloudy_images(Collection):
            months = ee.List.sequence(1, 12)
            def get_month_image(month):
                monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            
            monthly_images_list = months.map(get_month_image)
            monthly_images_collection = ee.ImageCollection.fromImages(monthly_images_list)
            return monthly_images_collection
            
        # filteredCollectionBands = get_monthly_least_cloudy_images(filteredCollection_overlap) 
        filteredCollectionBands = get_monthly_least_cloudy_images(filteredCloudCollection) 

        def addCloud(image):
            id = image.get("First_id")
            cloud = image.get("Cloud_coverage")
            Complete_id = image.set("Full_id", ee.String(id).cat("_Cloud_").cat(cloud))
            return Complete_id

        Complete_collection = filteredCollectionBands.map(addCloud)
        
        return Complete_collection 

    ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
    return ee.ImageCollection(ImageryCollections)



def compute_ndvi_mean(image):
    # Select elevation for geometry
    elevation_mask = image.select('elevation')

    # Compute NDVI
    ndvi_mean = image.normalizedDifference(['S2_NIR', 'S2_Red']).rename('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=elevation_mask.geometry(),
        scale=30,
        maxPixels=1e13
    )
    
    # Compute NDWI_Green
    ndwi_green_mean = image.normalizedDifference(['S2_Green', 'S2_NIR']).rename('NDWI_Green').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=elevation_mask.geometry(),
        scale=30,
        maxPixels=1e13
    )
    
    # Extract month and dam status
    month = image.get('Image_month')
    status = image.get('Dam_status')

    # Combine all metrics and metadata
    combined_metrics = ee.Dictionary({
        'NDVI': ndvi_mean.get('NDVI'),
        'NDWI_Green': ndwi_green_mean.get('NDWI_Green'),
        'Image_month': month,
        'Dam_status': status
    })

    # Return a feature with combined metrics as properties
    return ee.Feature(None, combined_metrics)      
    
def compute_lst(s2_image, landsat_col, boxArea):
    """Computes LST from the median of the filtered Landsat collection."""
    median_img = landsat_col.median().clip(boxArea)

    # Compute NDVI again, just to get min/max
    ndvi = median_img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    # Compute NDVI min/max
    ndvi_dict = ndvi.reduceRegion(
        reducer=ee.Reducer.minMax(), 
        geometry=boxArea, 
        scale=30, 
        maxPixels=1e13
    )

    ndvi_min = ee.Number(ee.Algorithms.If(ndvi_dict.contains('NDVI_min'), ndvi_dict.get('NDVI_min'), 0))
    ndvi_max = ee.Number(ee.Algorithms.If(ndvi_dict.contains('NDVI_max'), ndvi_dict.get('NDVI_max'), 0))

    # Compute Fraction of Vegetation (FV)
    fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min).max(1e-6)).pow(2).rename('FV')

    # Compute Emissivity (EM)
    em = fv.multiply(0.004).add(0.986).rename('EM')

    # Select the thermal band
    thermal = median_img.select('ST_B10').rename('thermal')

    # Compute LST in °C
    lst = thermal.expression(
        '(TB / (1 + (0.00115 * (TB / 1.438)) * log(em))) - 273.15',
        {'TB': thermal, 'em': em}
    ).rename('LST')

    return lst

def add_landsat_lst(s2_image):
    """
    For each Sentinel-2 image:
    1) Extract its year and month.
    2) Filter Landsat images in that same date range.
    3) Compute median LST over the geometry.
    4) Return the original S2 image with an added LST band (or 0 if none found).
    """
    year = ee.Number(s2_image.get('Image_year'))
    month = ee.Number(s2_image.get('Image_month'))
    
    start_date = ee.Date.fromYMD(year, month, ee.Number(1))
    end_date = start_date.advance(1, 'month')
    boxArea = s2_image.get('Area')  # This is assumed to be some geometry on your image properties

    # Function to apply scaling factors
    def apply_scale_factors(image):
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return (image
                .addBands(opticalBands, overwrite=True)
                .addBands(thermalBands, overwrite=True))

    # Function to mask clouds
    def cloud_mask(image):
        cloudShadowBitmask = (1 << 3)
        cloudBitmask = (1 << 5)
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(cloudShadowBitmask).eq(0).And(
               qa.bitwiseAnd(cloudBitmask).eq(0))
        return image.updateMask(mask)

    # Build the Landsat collection
    landsat_col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(boxArea)
        .map(apply_scale_factors)
        .map(cloud_mask)
    )

    # Compute NDVI stats on each image to ensure we only keep valid images
    def add_ndvi_stats(img):
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndvi_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=boxArea,
            scale=30,
            maxPixels=1e13
        )
        return img.setMulti(ndvi_dict)

    landsat_col = landsat_col.map(add_ndvi_stats)
    filtered_col = landsat_col.filter(ee.Filter.neq('NDVI_min', None))
    collection_size = filtered_col.size()

    # If no valid images, we return an empty LST band with 0
    empty_image = ee.Image.constant(0).rename(['LST']).clip(boxArea)

    lst_image = ee.Algorithms.If(
        collection_size.eq(0),
        empty_image,  # 0 LST if no valid Landsat images
        compute_lst(s2_image, filtered_col, boxArea)
    )

    lst_image = ee.Image(lst_image)
    # Add the LST band to the S2 image
    return s2_image.addBands(lst_image).set('size', collection_size)

def compute_all_metrics(image):
    """
    Returns an ee.Feature containing mean NDVI, NDWI_Green, and LST 
    for the geometry of interest.
    """
    # 1) Use the 'elevation' band (or any other reference band) to get geometry
    elevation_mask = image.select('elevation')
    geometry = elevation_mask.geometry()

    # 2) Compute NDVI using Sentinel-2 Red & NIR
    ndvi = image.normalizedDifference(['S2_NIR', 'S2_Red']).rename('NDVI')
    ndvi_mean = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e13
    ).get('NDVI')

    # 3) Compute NDWI_Green using Sentinel-2 Green & NIR
    ndwi_green = image.normalizedDifference(['S2_Green', 'S2_NIR']).rename('NDWI_Green')
    ndwi_green_mean = ndwi_green.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e13
    ).get('NDWI_Green')

    # 4) Select LST band (added by add_landsat_lst)
    lst_band = image.select('LST')
    lst_mean = lst_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e13
    ).get('LST')

    # 5) Extract metadata (month, year, dam status, etc.)
    month = image.get('Image_month')
    status = image.get('Dam_status')
    year = image.get('Image_year')

    # Combine all metrics & metadata into a dictionary
    combined_metrics = ee.Dictionary({
        'NDVI': ndvi_mean,
        'NDWI_Green': ndwi_green_mean,
        'LST': lst_mean,
        'Image_month': month,
        'Image_year': year,
        'Dam_status': status
    })

    # Return as an ee.Feature
    return ee.Feature(None, combined_metrics)
