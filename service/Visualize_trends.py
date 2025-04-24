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

####### Functions to for filtering collection
### Filtering without flow direction
def S2_Export_for_visual(Dam_Collection):
    def extract_pixels(box):
        try:
            imageDate = ee.Date(box.get("Survey_Date"))
            StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")
            EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")

            boxArea = box.geometry()
            damId = box.get("id_property")
            DamStatus = box.get('Dam')
            DamDate = box.get('Damdate')
            DamGeo = box.get('Point_geo')

            # Ensure Point_geo is a valid point geometry
            point_geo = ee.Algorithms.If(
                ee.Algorithms.IsEqual(DamGeo, None),
                boxArea.centroid(),  # Use centroid if Point_geo is None
                DamGeo
            )

            S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

            def add_cloud_mask_band(image):
                qa = image.select('QA60')
                cloudBitMask = 1 << 10
                cirrusBitMask = 1 << 11
                cloud_mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                    qa.bitwiseAnd(cirrusBitMask).eq(0)
                )
                cloud_mask_band = cloud_mask.rename('cloudMask').toUint16()
                return image.addBands(cloud_mask_band)

            S2_cloud_band = S2.map(add_cloud_mask_band)

            oldBandNames = ['B2', 'B3', 'B4', 'B8','cloudMask']
            newBandNames = ['S2_Blue', 'S2_Green', 'S2_Red', 'S2_NIR','S2_Binary_cloudMask']
            S2_named_bands = S2_cloud_band.map(lambda image: image.select(oldBandNames).rename(newBandNames))

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
                
                dataset = ee.Image('USGS/3DEP/10m')
                elevation_select = dataset.select('elevation')
                elevation = ee.Image(elevation_select)
                
                point_geom = point_geo  # Use processed point_geo
                point_elevation = ee.Number(elevation.sample(point_geom, 10).first().get('elevation'))
                buffered_area = boxArea
                elevation_clipped = elevation.clip(buffered_area)

                point_plus = point_elevation.add(3)
                point_minus = point_elevation.subtract(5)        
                elevation_masked = elevation_clipped.where(elevation_clipped.lt(point_minus), 0).where(elevation_clipped.gt(point_minus), 1).where(elevation_clipped.gt(point_plus), 0)
                elevation_masked2 = elevation_masked.updateMask(elevation_masked.eq(1))

                Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate))\
                    .set("Dam_id", damId)\
                    .set("Dam_status", DamStatus)\
                    .set("Image_month", image_month)\
                    .set("Image_year", image_year)\
                    .set("Area", boxArea)\
                    .set("id_property", damId)\
                    .set("Point_geo", point_geo)\
                    .clip(boxArea)
                
                return Full_image.addBands(elevation_masked2)

            filteredCollection2 = filteredCollection.map(add_band)

            def calculate_cloud_coverage(image):
                cloud = image.select('S2_Binary_cloudMask')
                cloud_stats = cloud.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=image.geometry(),
                    scale=10,
                    maxPixels=1e9
                )
                clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
                cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)
                return image.set('Cloud_coverage', cloud_coverage_percentage)

            filteredCloudCollection = filteredCollection2.map(calculate_cloud_coverage)

            def get_monthly_least_cloudy_images(Collection):
                months = ee.List.sequence(1, 12)
                def get_month_image(month):
                    monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                    return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
                
                monthly_images_list = months.map(get_month_image)
                return ee.ImageCollection.fromImages(monthly_images_list)
            
            filteredCollectionBands = get_monthly_least_cloudy_images(filteredCloudCollection)

            def addCloud(image):
                id = image.get("First_id")
                cloud = image.get("Cloud_coverage")
                return image.set("Full_id", ee.String(id).cat("_Cloud_").cat(cloud))

            Complete_collection = filteredCollectionBands.map(addCloud)
            return Complete_collection

        except Exception as e:
            st.warning(f"Error processing image: {str(e)}")
            return None

    ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
    return ee.ImageCollection(ImageryCollections)

##### Filtering with flowline 
def S2_Export_for_visual_flowdir(Dam_Collection, filtered_waterway):
    def extract_pixels(box):
        imageDate = ee.Date(box.get("Survey_Date"))
        StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")

        EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")

        boxArea = box.geometry()
        # DateString = box.get("stringID")
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
            
            ### buffered_geometry
            # buffered_geometry = boxArea.geometry()
            buffered_geometry = box.geometry()
            point_geom  = buffered_geometry.centroid()
            buffered_geometry = point_geom.buffer(200)
            
            waterway_state = filtered_waterway.filterBounds(buffered_geometry)
            
            dataset = ee.Image('USGS/3DEP/10m')
            elevation_select = dataset.select('elevation')
            elevation = ee.Image(elevation_select)
            
            
            # point_geom = firstFeature.geometry()
            point_elevation = ee.Number(elevation.sample(point_geom, 10).first().get('elevation'))
            
            # Clip and mask based on some +/- thresholds
            point_plus = point_elevation.add(3)
            point_minus = point_elevation.subtract(10)        
            elevation_clipped = elevation.clip(buffered_geometry)
            
            # 1 = within range, 0 = outside range
            elevation_masked = (elevation_clipped
                .where(elevation_clipped.lt(point_minus), 0)
                .where(elevation_clipped.gt(point_minus), 1)
                .where(elevation_clipped.gt(point_plus), 0))

            elevation_masked2 = elevation_masked.updateMask(elevation_masked.eq(1))
            
            def find_closest_flowline(point_geom, waterway = filtered_waterway):
                # Filter to flowlines within some max distance bounding box
                # (This helps avoid dealing with massive data.)
                candidate_fc = waterway.filterBounds(point_geom.buffer(100))
                
                # Compute distance from each flowline to the point:
                candidate_fc_with_dist = candidate_fc.map(lambda f: f.set('dist', f.geometry().distance(point_geom)))
                
                # Sort by distance ascending and take the first feature
                closest = ee.Feature(candidate_fc_with_dist.sort('dist').first())
                return closest
            
            
            main_flowline = ee.Feature(find_closest_flowline(point_geom))
            main_geom = main_flowline.geometry()
            
            # Compute the distance from your point to the line (in meters, if your CRS is in meters)
            distance_to_line = main_geom.distance(point_geom)
            
            # Buffer the point by this distance.
            # (Note: if the point lies exactly on the line, distance_to_line will be 0. You might need a check for that.)
            buffer_radius = ee.Number(distance_to_line).add(1)  # or some small number in degrees
            buffered_point = point_geom.buffer(buffer_radius)
            # buffered_point_2 = point_geom.buffer(distance_to_line)
            
            # The intersection of the line and this buffer gives the nearest point.
            closest_point_geom = main_geom.intersection(buffered_point, 1)
            
            coords = ee.List(closest_point_geom.coordinates())

            List = coords.flatten()
            
            new_coord = ee.List([
                ee.Number(List.get(0)), 
                ee.Number(List.get(1))
            ])
            closest_point = ee.Geometry.Point(new_coord)
            
            # closest_point = ee.Geometry.Point(coords.get(0))
            
            p1 = ee.Geometry.Point(new_coord)
            
            second_coord = ee.List([
                ee.Number(List.get(2)), 
                ee.Number(List.get(3))
            ])
            
            p2 = ee.Geometry.Point(second_coord)


            
            # Create the first linestring between p1 and p2
            line1 = ee.Geometry.LineString([p1.coordinates(), p2.coordinates()])
            
            # Cast each coordinate to an ee.Number so we can do arithmetic
            x1 = ee.Number(p1.coordinates().get(0))
            y1 = ee.Number(p1.coordinates().get(1))
            x2 = ee.Number(p2.coordinates().get(0))
            y2 = ee.Number(p2.coordinates().get(1))
            
            # Midpoint in latitude-longitude
            xm = x1.add(x2).divide(2)
            ym = y1.add(y2).divide(2)
            
            # Vector along line1
            dx = x2.subtract(x1)
            dy = y2.subtract(y1)
            
            # To rotate (dx, dy) by 90°, pick (dy, -dx) or (-dy, dx).
            # We'll choose (dy, -dx) here.
            dx_perp = dy
            dy_perp = dx.multiply(-1)
            
            
            dx_half = dx_perp.divide(2)
            dy_half = dy_perp.divide(2)
            length_factor = 10
            
            # Scale the perpendicular vector
            dx_long = dx_perp.multiply(length_factor).divide(2)  
            dy_long = dy_perp.multiply(length_factor).divide(2)  
            
            # Perpendicular line endpoints
            p3 = ee.Geometry.Point([xm.subtract(dx_long), ym.subtract(dy_long)])
            p4 = ee.Geometry.Point([xm.add(dx_long), ym.add(dy_long)])
            
            # Create the perpendicular LineString
            extended_perpendicular = ee.Geometry.LineString([p3.coordinates(), p4.coordinates()])
            
            buffer_distance = 130  # meters
            buffered_poly = extended_perpendicular.buffer(buffer_distance)
            
            
            bbox = buffered_poly.bounds()  # This is an ee.Geometry with a single ring
            boundingCoords = bbox.coordinates()         # ee.List
            boundingRing = ee.List(boundingCoords.get(0))  # ee.List of [ [west, south], [west, north], ... ]
            
            westSouth = ee.List(boundingRing.get(0))  # [west, south]
            westNorth = ee.List(boundingRing.get(1))  # [west, north]
            eastNorth = ee.List(boundingRing.get(2))  # [east, north]
            eastSouth = ee.List(boundingRing.get(3))  # [east, south]
            
            west  = ee.Number(westSouth.get(0))
            south = ee.Number(westSouth.get(1))
            east  = ee.Number(eastNorth.get(0))
            north = ee.Number(eastNorth.get(1))
            
            # Mid-latitude
            mid_lat = south.add(north).divide(2)
            
            # Create top/bottom rectangles
            top_rect = ee.Geometry.Rectangle([west, mid_lat, east, north])
            bot_rect = ee.Geometry.Rectangle([west, south, east, mid_lat])
            
            # 6) Intersect rectangles with the buffer to get two halves
            top_poly = buffered_poly.intersection(top_rect, maxError=1)
            bot_poly = buffered_poly.intersection(bot_rect, maxError=1)
            
            
            top_feature = ee.Feature(top_poly, {'id': 'top'})
            bot_feature = ee.Feature(bot_poly, {'id': 'bot'})
            
            
            # Step 2: Buffer the extended line just enough to make a thin clipping strip
            split_strip = extended_perpendicular.buffer(1)
            
            def get_closest_vertex_index(coords, pt):
                distances = coords.map(lambda c: ee.Geometry.Point(c).distance(pt))
                min_dist = distances.reduce(ee.Reducer.min())
                return ee.List(distances).indexOf(min_dist)
            
            # Get the full list of coordinates from the flowline geometry.
            line_coords = main_geom.coordinates()
            
            
            # Find the index of the vertex nearest to our computed closest point.
            closest_index = get_closest_vertex_index(line_coords, closest_point)
            
            upstream_coords = line_coords.slice(0, ee.Number(closest_index).add(1))
            downstream_coords = line_coords.slice(ee.Number(closest_index), line_coords.size())
            
            #################

            def ensure_two_coords(coords, main_coords, closest_idx, direction):
                """
                coords: The initial list of coordinates for upstream or downstream.
                main_coords: The entire coordinate list of the flowline.
                closest_idx: Index of the vertex nearest the point of interest.
                direction: 'up' or 'down' – determines where we add a fallback coordinate.
                """
                coords_list = ee.List(coords)
                size = coords_list.size()
                
                # If already >= 2, do nothing; otherwise add a neighbor from main_coords.
                return ee.Algorithms.If(
                    size.gte(2),
                    coords_list,
                    ee.Algorithms.If(
                        direction == 'up',
                        # Upstream fallback: add the vertex after closest_idx
                        coords_list.cat([
                            main_coords.get(
                                ee.Number(closest_idx)
                                  .add(1)
                                  .min(main_coords.size().subtract(1))
                            )
                        ]),
                        # Downstream fallback: add the vertex before closest_idx
                        coords_list.cat([
                            main_coords.get(
                                ee.Number(closest_idx)
                                  .subtract(1)
                                  .max(0)
                            )
                        ])
                    )
                )
            
            # Ensure at least two coordinates for both slices.
            upstream_coords_fixed = ensure_two_coords(upstream_coords, line_coords, closest_index, 'up')
            downstream_coords_fixed = ensure_two_coords(downstream_coords, line_coords, closest_index, 'down')
            
            # Convert them to ee.List for further manipulation.
            upstream_list = ee.List(upstream_coords_fixed)
            downstream_list = ee.List(downstream_coords_fixed)
            
            # 3) Remove the shared coordinate from whichever slice is longer.
            def remove_shared_coordinate(up_coords, down_coords):
                up_size = up_coords.size()
                down_size = down_coords.size()
                
                # If upstream is bigger, remove its last coordinate.
                # Otherwise (or if equal), remove the first coordinate from downstream.
                trimmed_up = ee.Algorithms.If(
                    up_size.gt(down_size),
                    up_coords.slice(0, up_size.subtract(1)),  # remove last from upstream
                    up_coords
                )
                trimmed_down = ee.Algorithms.If(
                    up_size.gte(down_size),
                    down_coords,
                    down_coords.slice(1)                     # remove first from downstream
                )
                return {
                    'up': trimmed_up,
                    'down': trimmed_down
                }
            
            # Apply the removal.
            removed_dict = remove_shared_coordinate(upstream_list, downstream_list)
            final_up_coords = ee.List(removed_dict.get('up'))
            final_down_coords = ee.List(removed_dict.get('down'))
            
            # 4) Convert to final LineString geometries.
            upstream_line = ee.Geometry.LineString(final_up_coords)
            downstream_line = ee.Geometry.LineString(final_down_coords)


            ##################
            # Define upstream and downstream lines
            # upstream_line = ee.Geometry.LineString(upstream_coords)
            # downstream_line = ee.Geometry.LineString(downstream_coords)
            
            def label_flow_basic(feature):
                intersects_up = feature.geometry().intersects(upstream_line, ee.ErrorMargin(1))
                intersects_down = feature.geometry().intersects(downstream_line, ee.ErrorMargin(1))
                
                return ee.Algorithms.If(
                    intersects_up,
                    # If up == True
                    ee.Algorithms.If(
                        intersects_down,
                        feature.set('flow', 'both'),
                        feature.set('flow', 'upstream')
                    ),
                    # else (up == False)
                    ee.Algorithms.If(
                        intersects_down,
                        feature.set('flow', 'downstream'),
                        feature.set('flow', 'unknown')
                    )
                )

            
            
            halves = ee.FeatureCollection([top_feature, bot_feature])
            
            # Label each half with the basic rule above
            labeled_halves = halves.map(label_flow_basic)
            features = labeled_halves.toList(labeled_halves.size())
            f1 = ee.Feature(features.get(0))
            f2 = ee.Feature(features.get(1))
            f1_flow = f1.getString('flow') ## upstream
            f2_flow = f2.getString('flow') ## both
            def opposite(flow_str):
                return ee.String(ee.Algorithms.If(flow_str.equals('upstream'),'downstream', 'upstream'))
            f1_new = ee.Feature(ee.Algorithms.If(f1_flow.equals('both'),f1.set('flow', opposite(f2_flow)),f1))
            f2_new = ee.Feature(ee.Algorithms.If(f2_flow.equals('both'),f2.set('flow', opposite(f1_flow)),f2))
            new_fc = ee.FeatureCollection([f1_new, f2_new])
            
            
            # # Filter into two variables
            upstream_half = new_fc.filter(ee.Filter.eq('flow', 'upstream')).geometry()
            downstream_half = new_fc.filter(ee.Filter.eq('flow', 'downstream')).geometry()
            
            
            # Filter out the main_flowline from the rest
            others = waterway_state.filter(ee.Filter.neq('system:index', main_flowline.get('system:index')))
            # 5) CLASSIFY THE OTHER FLOWLINES INTO UPSTREAM / DOWNSTREAM / UNCLASSIFIED
            def classify_flowline(feature,upstream,downstream):
                geom = feature.geometry()
                intersects_up = geom.intersects(upstream)
                intersects_down = geom.intersects(downstream)
                
                # Nested ee.Algorithms.If() to avoid using .and()
                return ee.Algorithms.If(
                    intersects_up,
                    ee.Algorithms.If(
                        intersects_down,
                        # Touches both => unclassified
                        feature.set('flow_part', 'unclassified'),
                        # Touches only upstream
                        feature.set('flow_part', 'upstream_flow')
                    ),
                    ee.Algorithms.If(
                        intersects_down,
                        # Touches only downstream
                        feature.set('flow_part', 'downstream_flow'),
                        # Touches neither => unclassified
                        feature.set('flow_part', 'unclassified')
                    )
                )
            
            classified_rest1 = others.map(lambda f: classify_flowline(f, upstream_line, downstream_line))
            
            
            upstream_others1 = classified_rest1.filter(ee.Filter.eq('flow_part', 'upstream_flow'))
            downstream_others1 = classified_rest1.filter(ee.Filter.eq('flow_part', 'downstream_flow'))
            unclassified_others1 = classified_rest1.filter(ee.Filter.eq('flow_part', 'unclassified'))
            
            upstreamWaterway = ee.FeatureCollection([ee.Feature(upstream_line)]).merge(upstream_others1)
            downstreamWaterway = ee.FeatureCollection([ee.Feature(downstream_line)]).merge(downstream_others1)
            
            classified_rest2 = unclassified_others1.map(lambda f: classify_flowline(f, upstreamWaterway, downstreamWaterway))
            
            
            upstream_others2 = classified_rest2.filter(ee.Filter.eq('flow_part', 'upstream_flow'))
            downstream_others2 = classified_rest2.filter(ee.Filter.eq('flow_part', 'downstream_flow'))
            unclassified_others2 = classified_rest2.filter(ee.Filter.eq('flow_part', 'unclassified'))
            
            upstreamWaterway2 = upstream_others2.merge(upstreamWaterway)
            downstreamWaterway2 = downstream_others2.merge(downstreamWaterway)
            
            classified_rest3 = unclassified_others2.map(lambda f: classify_flowline(f, upstreamWaterway2, downstreamWaterway2))
            
            
            upstream_others3 = classified_rest3.filter(ee.Filter.eq('flow_part', 'upstream_flow'))
            downstream_others3 = classified_rest3.filter(ee.Filter.eq('flow_part', 'downstream_flow'))
            
            upstreamWaterway3 = upstream_others3.merge(upstreamWaterway2)
            downstreamWaterway3 = downstream_others3.merge(downstreamWaterway2)
            
            
            # 6) DISSOLVE EACH GROUP INTO SINGLE GEOMETRIES
            upstreamGeometry = upstreamWaterway3.geometry().dissolve()
            downstreamGeometry = downstreamWaterway3.geometry().dissolve()
            
            
            # 7) BUFFER & MASK ELEVATION
            bufferDist = 100  # meters
            upstreamBuffered = upstreamGeometry.buffer(bufferDist)
            downstreamBuffered = downstreamGeometry.buffer(bufferDist)
            
            
            upstream_mask_img = ee.Image.constant(1).clip(upstreamBuffered)
            downstream_mask_img = ee.Image.constant(1).clip(downstreamBuffered)
            
            
            # Create inverse masks using downstream_half and upstream_half geometries
            downstream_half_mask = ee.Image.constant(0).paint(downstream_half, 1)
            upstream_half_mask = ee.Image.constant(0).paint(upstream_half, 1)
            
            
            upstream_final_mask = upstream_mask_img.updateMask(downstream_half_mask.Not())
            downstream_final_mask = downstream_mask_img.updateMask(upstream_half_mask.Not())
            
            # Apply the refined masks to the elevation image
            upstream_elev_mask = elevation_masked2.updateMask(
            elevation_masked2.mask().And(upstream_final_mask)
            )
            
            downstream_elev_mask = elevation_masked2.updateMask(
            elevation_masked2.mask().And(downstream_final_mask)
            )


            downstream_rename = downstream_elev_mask.rename('downstream')
            upstream_rename = upstream_elev_mask.rename('upstream')
            # Add bands, create new "id" property to name the file, and clip the images to the ROI
            # Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect))\
            Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_"))\
                .set("Dam_id",damId)\
                .set("Dam_status",DamStatus)\
                .set("Image_month",image_month)\
                .set("Image_year",image_year)\
                .set("Area", boxArea)\
                .clip(boxArea)
            # .addBands(upstream_elev_mask).addBands(downstream_elev_mask)
            return Full_image.addBands(downstream_rename).addBands(upstream_rename).addBands(elevation_masked2)
            
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



############# Functions to compute metrics
##### Compute LST
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


############# Functions to add additional datasets- Landsat LST, OPEN-ET ET
##### JUST LST
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

###### LST AND ET
def add_landsat_lst_et(s2_image):
    """Adds Landsat LST and OpenET data to a Sentinel-2 image."""
    
    year = ee.Number(s2_image.get('Image_year'))
    month = ee.Number(s2_image.get('Image_month'))
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')

    # Ensure boxArea is valid; if None, use the image bounds
    boxArea = ee.Algorithms.If(
        s2_image.get('Area'),
        s2_image.get('Area'),
        s2_image.geometry()  # Use image geometry as fallback
    )

    ## **STEP 1: PROCESS LANDSAT DATA FOR LST**
    def apply_scale_factors(image):
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(opticalBands, overwrite=True).addBands(thermalBands, overwrite=True)

    def cloud_mask(image):
        cloudShadowBitmask = (1 << 3)
        cloudBitmask = (1 << 5)
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(cloudShadowBitmask).eq(0).And(
               qa.bitwiseAnd(cloudBitmask).eq(0))
        return image.updateMask(mask)

    # Build the Landsat collection
    landsat_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                   .filterDate(start_date, end_date)
                   .filterBounds(boxArea)
                   .map(apply_scale_factors)
                   .map(cloud_mask))

    def add_ndvi_stats(img):
        # Compute NDVI
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

        # NDVI min/max
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

    empty_image = ee.Image.constant(0).rename(['LST']).clip(boxArea)

    lst_image = ee.Algorithms.If(
        collection_size.eq(0),
        empty_image,  
        compute_lst(s2_image, filtered_col, boxArea)  
    )
    lst_image = ee.Image(lst_image)

    ## **STEP 2: PROCESS ET DATA FROM OPENET**
    et_collection = (
        ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
        .filterDate(start_date, end_date)
        .filterBounds(boxArea)
    )

    # Compute the monthly ET mean for the given area
    et_monthly = et_collection.mean().select("et_ensemble_mad").rename("ET")

    # If ET data is missing, return an empty ET band
    et_final = ee.Algorithms.If(
        et_collection.size().eq(0),
        ee.Image.constant(0).rename("ET").clip(boxArea),  # Empty ET image
        et_monthly.clip(boxArea)
    )
    et_final = ee.Image(et_final)

    ## **STEP 3: ADD LST & ET TO THE SENTINEL-2 IMAGE**
    return s2_image.addBands(lst_image).addBands(et_final).set("size", collection_size)

########### Functions to calculate means
##### NDVI, LST
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


####### Compute ET and include upstream and downstream
##### NDVI, LST, ET
def compute_all_metrics_LST_ET(image):
    """
    Returns an ee.Feature containing mean NDVI, NDWI_Green, LST, and ET 
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

    # 4) Select LST band (added by add_landsat_lst_et)
    lst_band = image.select('LST')
    lst_mean = lst_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e13
    ).get('LST')

    # 5) Select ET band (added by add_landsat_lst_et)
    et_band = image.select('ET')
    et_mean = et_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e13
    ).get('ET')

    # 6) Extract metadata (month, year, dam status, etc.)
    month = image.get('Image_month')
    status = image.get('Dam_status')
    year = image.get('Image_year')
    id_property = image.get('id_property')  # Add id_property

    # Combine all metrics & metadata into a dictionary
    combined_metrics = ee.Dictionary({
        'NDVI': ndvi_mean,
        'NDWI_Green': ndwi_green_mean,
        'LST': lst_mean,
        'ET': et_mean,  # 🚀 New: Adding ET to the feature
        'Image_month': month,
        'Image_year': year,
        'Dam_status': status,
        'id_property': id_property  # Add id_property to the dictionary
    })

    # Return as an ee.Feature
    return ee.Feature(None, combined_metrics)


###### Upstream and downstream
def extract_coordinates_df(dam_data):
    """
    Extract coordinates from dam data and return a DataFrame with id_property and coordinates.
    """
    try:
        # Extract features from dam_data
        features = dam_data.getInfo()['features']
        
        # Create a list to store coordinates
        coords_data = []
        
        for feature in features:
            properties = feature['properties']
            id_property = properties.get('id_property')
            point_geo = properties.get('Point_geo')
            
            if point_geo:
                # Extract coordinates from Point_geo
                coords = point_geo['coordinates']
                coords_data.append({
                    'id_property': id_property,
                    'longitude': coords[0],
                    'latitude': coords[1]
                })
        
        # Convert to DataFrame
        coords_df = pd.DataFrame(coords_data)
        return coords_df
        
    except Exception as e:
        st.warning(f"Error extracting coordinates: {str(e)}")
        return pd.DataFrame(columns=['id_property', 'longitude', 'latitude'])

def compute_all_metrics_up_downstream(image):
    """
    Returns an ee.Feature containing separate upstream/downstream mean NDVI, NDWI_Green, LST, and ET.
    """
    # 1) Grab upstream/downstream mask bands
    upstream_mask = image.select('upstream')
    downstream_mask = image.select('downstream')
    
    # Check if bands exist
    valid_up = upstream_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    ).getNumber('upstream')
    
    # 2) Compute NDVI using Sentinel-2 Red & NIR
    ndvi = image.normalizedDifference(['S2_NIR', 'S2_Red']).rename('NDVI')
    
    # Upstream NDVI
    ndvi_up_img = ndvi.updateMask(upstream_mask)
    ndvi_up = ndvi_up_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    ).get('NDVI')
    
    # Downstream NDVI
    ndvi_down_img = ndvi.updateMask(downstream_mask)
    ndvi_down = ndvi_down_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    ).get('NDVI')

    # 3) Compute NDWI_Green using Sentinel-2 Green & NIR
    ndwi_green = image.normalizedDifference(['S2_Green', 'S2_NIR']).rename('NDWI_Green')
    
    # Upstream NDWI
    ndwi_up_img = ndwi_green.updateMask(upstream_mask)
    ndwi_up = ndwi_up_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    ).get('NDWI_Green')
    
    # Downstream NDWI
    ndwi_down_img = ndwi_green.updateMask(downstream_mask)
    ndwi_down = ndwi_down_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    ).get('NDWI_Green')

    # 4) LST band
    lst = image.select('LST')
    lst_up = lst.updateMask(upstream_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).get('LST')
    
    lst_down = lst.updateMask(downstream_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).get('LST')

    # 5) ET band
    et = image.select('ET')
    et_up = et.updateMask(upstream_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).get('ET')

    et_down = et.updateMask(downstream_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e13
    ).get('ET')
    
    # 6) Extract metadata
    month = image.get('Image_month')
    status = image.get('Dam_status')
    year = image.get('Image_year')
    id_property = image.get('id_property')

    # Combine everything into a dictionary
    combined_metrics = ee.Dictionary({
        'Image_month': month,
        'Image_year': year,
        'Dam_status': status,
        'id_property': id_property,
        
        'NDVI_up': ndvi_up,
        'NDVI_down': ndvi_down,
        
        'NDWI_up': ndwi_up,
        'NDWI_down': ndwi_down,
        
        'LST_up': lst_up,
        'LST_down': lst_down,
        
        'ET_up': et_up,
        'ET_down': et_down
    })

    return ee.Feature(None, combined_metrics)
