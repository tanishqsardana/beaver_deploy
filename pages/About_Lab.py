import streamlit as st

# st.title("About the Beaver Lab")

st.markdown('''
# Welcome to the Beaver Impacts Tool

The Beaver Impacts Tool is a web app developed by the Beaver Lab at [Collaborative Earth](https://www.collaborative.earth/) to help land managers and researchers understand the impact of beavers and beaver-based restoration on their land.

If you're reading this, **you have agreed to beta-test this tool.** Thanks! This means that after you play around with the tool, *you commit to filling out this feedback form:* [Beaver Impacts Feedback](https://docs.google.com/forms/d/e/1FAIpQLSeE1GP7OptA4-z8Melz2AHxNsddtL9ZgJVXdVVtxLsrljJ10Q/viewform)

Briefly, this tool allows users to:
1. Upload locations of observed dams or installed beaver dam analogs.
2. Upload or generate nearby locations without dams for comparison.
3. Pull satellite imagery for the provided locations for a given year.
4. Make plots of vegetation and water indices, temperature, and evapotranspiration over the course of a year, comparing dam to no-dam locations.
5. Download plots and/or raw data.

**To get started, click 'Analyze Impacts!' on the left.** The rest of this page is documentation to help you understand what the tool does. If you have any questions, please contact [Grace Lindsay](mailto:gracewlindsay@gmail.com).

## Detailed Documentation
We use the Google Earth Engine Python API under the hood to acess various data sources and perform analyses. Users work through the app according to the following steps:
1. **Upload Dam Locations.** Coordinates of the dam locations (or BDA, etc.) should be provided in a CSV (containing columns for latitude and longitude) or as a GeoJSON (GeoJSONs can be generated through a map interface [here](https://geojson.io/) if desired). Users then select the year for which they would like to perform the analysis by using a drop down menu. Note that due to data restrictions we can only offer analysis of data from 2018 through the end of 2024 at this time, and some years and locations may not have evapotranspiration data.

![Example CSV upload](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step1.png)

2. **Select Waterway Map.** Our analysis requires a waterway map. By default, we use the [2022 NHD dataset](https://www.usgs.gov/national-hydrography/national-hydrography-dataset). Users can choose to use the WWF Free Flowing Rivers map or upload their own map as a GEE Asset table instead. We also plan to offer more built-in options in the future.

![Pre-loaded NHD map](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step2.png)

3. **Validate Dam Locations**. We perform a check to ensure that uploaded dam locations lie close to a waterway. Because the waterway maps are not perfect, we consider dams valid if they fall within 50m of a marked waterway (note that this threshold can be changed by the user). If invalid dams are found, the user can choose to continue with or without these invalid dams as part of the analysis.

![Example validation output](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step3.png)

4. **Negative Location Selection**. In our analysis, we compare locations with dams to those without. Users can upload their own dam-negative locations as a CSV file (note: if uploading negatives, it is the user's responsibility to make sure these points are on a waterway and not too close to a dam). Alternatively, negative points can be generated automatically. To do so, the user provides inner and outer radius values. Our algorithm then selects negative locations that are: 1.) at least `inner_radius` meters away from any dam, 2.) no more than `outer_radius` meters away from a dam, and 3.) on a waterway.

![Negative generation explanation](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step4.png)

5. **Generate Analysis Buffers**. We collect Sentinel-2 imagery (10m spatial resolution) for the positive and negative dam locations. To determine the amount of space around each of these locations to analyze, the user selects a buffer radius. Within this buffer radius, we apply an elevation-based mask. Specifically: the dam location is used to determine a base elevation and points are only included in the analysis if they are less than 3m different in elevation from this base. This is consistent with [previous work](http://geode.colorado.edu/~small/docs/Fairfax_et_al-2018-Ecohydrology.pdf) and is motivated by the fact that dam impacts will be limited by elevation differences. In effect, this means that not all locations within the buffer will be included in the analysis.

![Buffer regions plotted on map](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step5.png)

6. **Visualize Trends and Download Data**. In the final step, plots of core variables for dam and non-dam locations are presented as monthly means (with 95% confidence intervals) for the chosen year. For each month, we choose the least cloudy Sentinel-2 image for each location. From this multi-spectral Sentinel-2 imagery, we calculate a vegetation index (NDVI), and water index (green NDWI). We also report Land Surface Temperature (using Landsat 8), and evapotranspiration (as provided by [OpenET](https://etdata.org/)).Users can download these figures, as well as a CSV of the monthly values for each dam and no-dam site. Depending on the number of dams, this analysis step may take several minutes. Users also have the option of performing this analysis separately for portions of the waterway that are upstream and downsteam of each dam, though this will take longer to perform (note: this analysis is currently based on only a rough estimate of which portions of the buffer region are up- vs. downstream; we are working to refine this approach).

![Running plot generation](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/Step6.png)

**Users can expect to see plots similar to these:**
![Output plots](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/combined_trends_demo.png)

**Exported CSV contain monthly values for each location:**
![Example CSV output](https://raw.githubusercontent.com/Chris-ZiyuK/beaver-app-st/refs/heads/main/assets/outputcsv.png)



### If you want to re-run your analysis with changed dam locations or variables, please refresh the Analyze Impacts page.
''')

#st.balloons()
