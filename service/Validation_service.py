import ee
import streamlit as st
from typing import Dict, Tuple, List
import numpy as np
import geemap.foliumap as geemap

def validate_dam_waterway_distance(dam_collection: ee.FeatureCollection, 
                                 waterway_fc: ee.FeatureCollection, 
                                 max_distance: float = 100) -> Dict:
    """
    Validate if dam locations are near waterways
    
    Args:
        dam_collection: Collection of dam locations
        waterway_fc: Collection of waterway features
        max_distance: Maximum allowed distance in meters
        
    Returns:
        Dictionary containing validation results
    """
    try:
        # Function to calculate distance to nearest waterway for each dam
        def calculate_distance(dam_feature):
            dam_geom = dam_feature.geometry()
            # Create a small buffer around the dam point
            dam_buffer = dam_geom.buffer(max_distance)
            # Find waterways within the buffer
            nearby_waterways = waterway_fc.filterBounds(dam_buffer)
            # Calculate minimum distance to any waterway
            def get_distance(waterway):
                distance = dam_geom.distance(waterway.geometry())
                return waterway.set('distance', distance)
            # Get distances and find minimum
            distances = nearby_waterways.map(get_distance)
            min_distance = ee.Algorithms.If(
                distances.size().gt(0),
                distances.sort('distance').first().get('distance'),
                ee.Number(999999)  # Large number for dams with no nearby waterways
            )
            return dam_feature.set('distance_to_waterway', min_distance)
        
        # Add distance to all dams
        dams_with_distance = dam_collection.map(calculate_distance)
        
        # Split into valid and invalid dams
        valid_dams = dams_with_distance.filter(ee.Filter.lte('distance_to_waterway', max_distance))
        invalid_dams = dams_with_distance.filter(ee.Filter.gt('distance_to_waterway', max_distance))
        
        # Get invalid dams info with index
        invalid_dams_info = invalid_dams.map(
            lambda f: ee.Feature(
                f.geometry(),
                {
                    'distance': f.get('distance_to_waterway'),
                    'coordinates': f.geometry().coordinates()
                }
            )
        )
        
        return {
            'valid_dams': valid_dams,
            'invalid_dams': invalid_dams,
            'invalid_dams_info': invalid_dams_info,
            'total_dams': dam_collection.size(),
            'valid_count': valid_dams.size(),
            'invalid_count': invalid_dams.size()
        }
    except Exception as e:
        st.error(f"Error in distance validation: {str(e)}")
        raise

def check_waterway_intersection(dam_collection: ee.FeatureCollection, 
                              waterway_fc: ee.FeatureCollection) -> Dict:
    """
    Check if dams intersect with waterways
    
    Args:
        dam_collection: Collection of dam locations
        waterway_fc: Collection of waterway features
        
    Returns:
        Dictionary containing intersection results
    """
    try:
        # Function to check intersection for each dam
        def check_intersection(feature):
            point = feature.geometry()
            # Create a small buffer around the point (e.g., 10 meters)
            buffered_point = point.buffer(10)
            # Check if the buffered point intersects with any waterway
            intersects = waterway_fc.filterBounds(buffered_point).size().gt(0)
            return feature.set('intersects_waterway', intersects)
        
        # Add intersection status to all dams
        dams_with_intersection = dam_collection.map(check_intersection)
        
        # Split into intersecting and non-intersecting dams
        intersecting_dams = dams_with_intersection.filter(ee.Filter.eq('intersects_waterway', True))
        non_intersecting_dams = dams_with_intersection.filter(ee.Filter.eq('intersects_waterway', False))
        
        return {
            'intersecting_dams': intersecting_dams,
            'non_intersecting_dams': non_intersecting_dams,
            'total_dams': dam_collection.size(),
            'intersecting_count': intersecting_dams.size(),
            'non_intersecting_count': non_intersecting_dams.size()
        }
    except Exception as e:
        st.error(f"Error in intersection validation: {str(e)}")
        raise

def generate_validation_report(validation_results: Dict) -> str:
    """
    Generate a human-readable validation report
    
    Args:
        validation_results: Dictionary containing validation results
        
    Returns:
        Formatted report text
    """
    try:
        report = []
        
        if 'invalid_count' in validation_results:
            invalid_count = validation_results['invalid_count'].getInfo()
            if invalid_count > 0:
                report.append(f"Found {invalid_count} dams that are too far from waterways:")
                # Get invalid dams info
                invalid_dams_info = validation_results['invalid_dams_info'].getInfo()
                for i, feature in enumerate(invalid_dams_info['features'], 1):
                    props = feature['properties']
                    coords = props['coordinates']
                    report.append(f"- Point #{i}, Distance: {props['distance']:.2f}m, Location: [{coords[0]:.6f}, {coords[1]:.6f}]")
            else:
                report.append("All dams are within the specified distance from waterways.")
        
        return "\n".join(report)
    except Exception as e:
        st.error(f"Error generating validation report: {str(e)}")
        return "Error generating validation report."

def visualize_validation_results(dam_collection: ee.FeatureCollection, 
                               waterway_fc: ee.FeatureCollection, 
                               validation_results: Dict) -> geemap.Map:
    """
    Create a map visualization of validation results
    
    Args:
        dam_collection: Collection of dam locations
        waterway_fc: Collection of waterway features
        validation_results: Dictionary containing validation results
        
    Returns:
        Map object with validation visualization
    """
    try:
        # Create a new map
        map_obj = geemap.Map()
        map_obj.add_basemap("SATELLITE")
        
        # Add waterway layer
        map_obj.addLayer(waterway_fc, {"color": "blue", "width": 2}, "Waterways")
        
        # Add valid dams in green
        if 'valid_dams' in validation_results:
            map_obj.addLayer(
                validation_results['valid_dams'],
                {"color": "green", "pointSize": 5},
                "Valid Dams"
            )
        
        # Add invalid dams in red
        if 'invalid_dams' in validation_results:
            map_obj.addLayer(
                validation_results['invalid_dams'],
                {"color": "red", "pointSize": 5},
                "Invalid Dams"
            )
        
        # Center the map on the dam collection
        map_obj.centerObject(dam_collection)
        
        return map_obj
    except Exception as e:
        st.error(f"Error creating validation map: {str(e)}")
        raise 