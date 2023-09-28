# Piccard

## Introduction
**piccard** is a Python package which provides an alternative framework to traditional harmonization techniques for combining spatial data with inconsistent geographic units across multiple years. It uses a network representation containing nodes and edges to retain all information available in the data. Nodes are used to represent all the geographic areas (e.g., census tracts, dissemination areas) for each year. An edge connects two nodes when the geographic area corresponding to the tail node has at least a 5% area overlap with the geographic area corresponding to the head node in the previous available year.

### Research
The method behind this package can be found in the following research paper:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dias, F., & Silver, D. (2018). Visualizing demographic evolution using geographically inconsistent census data. California Digital Library (CDL). https://doi.org/10.31235/osf.io/a3gtd

## Installation
The latest released version is available at the [Python Package Index (PyPI)](https://pypi.org/project/piccard)

```
pip install piccard
```

## Importing the package

```
from piccard import piccard as pc 
```

## Useful Functions

**piccard.preprocessing(ct_data, year, id)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return a cleaned GeoDataFrame of the input data with a new column showing the area of each census tract.  

**piccard.create_network(census_dfs, years, id, threshold=0.05)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Creates a network representation of the temporal connections present in *census_dfs* over *years* when each yearly geographic area has at most *threshold* percentage of overlap with its corresponding area(s) in the next year.  

**piccard.create_network_table(census_dfs, years, id, threshold=0.05)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return the final network table with all the temporal connections present in *census_dfs* over *years* when each yearly geographic area has at most *threshold* percentage of overlap with its corresponding area(s) in the next year.  

**piccard.draw_subnetwork(network_table, G, sample_pct=0.005)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Draws a subgraph of the network representation, using a sample_pct% path sample from the network table.  

**Note**: Further explanation of the parameters and example code for all the above functions can be found in the documentation.  

## Dependencies
[GeoPandas - Allows spatial operations in Python, making it easier to work with geospatial data](https://geopandas.org/en/stable/)  
[Matplotlib - a comprehensive library for creating visualizations](https://matplotlib.org/)  
[NetworkX - Adds support for analyzing networks represented by nodes and edges](https://networkx.org/)  
[NumPy - Adds support for large, multi-dimensional arrays and matrices, with functions to operate on these arrays](https://numpy.org/)  
[pandas - Offers data structures and operations for manipulating numerical tables](https://pandas.pydata.org/)  

## Authors 
Maliha Lodi, Fernando Calderon Figueroa, Daniel Silver 
