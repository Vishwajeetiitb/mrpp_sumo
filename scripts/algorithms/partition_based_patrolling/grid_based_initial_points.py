import numpy as np
from shapely.geometry import Polygon
import rospkg
import pickle
import plotly.graph_objects as go
from shapely.geometry import Polygon
import numpy as np

def create_grid(poly, grid_size):
    xmin, ymin, xmax, ymax = poly.bounds
    xcoords = np.arange(xmin, xmax+grid_size, grid_size)
    ycoords = np.arange(ymin, ymax+grid_size, grid_size)
    cells = []
    for x in range(len(xcoords)-1):
        for y in range(len(ycoords)-1):
            cell_poly = Polygon([(xcoords[x], ycoords[y]), 
                                 (xcoords[x+1], ycoords[y]), 
                                 (xcoords[x+1], ycoords[y+1]), 
                                 (xcoords[x], ycoords[y+1])])
            if poly.intersects(cell_poly):
                cells.append(cell_poly)
    return cells


def get_initial_pose_list(hull,communication_range,n):
    grid_size = communication_range
    cells = create_grid(hull, grid_size)
    fully_contained_cells = []
    for cell in cells:
        if cell.within(hull):
            fully_contained_cells.append(cell)

    partially_contained_cells = []
    for cell in cells:
        intersection = cell.intersection(hull)
        if intersection.area > 0 and not cell.within(hull):
            partially_contained_cells.append(intersection)

    contained_cells = fully_contained_cells + partially_contained_cells
    centroids = []
    areas = np.array([])
    for cell in contained_cells:
        
        if cell.geom_type =="MultiPolygon":
            for geom in cell.geoms:
                if hull.contains(geom.centroid):
                    areas = np.append(areas,geom.area)
                    centroids.append(geom.centroid)               
        else:
            if hull.contains(cell.centroid):
                areas = np.append(areas,cell.area)
                centroids.append(cell.centroid)

    decreasing_indices = np.argsort(areas)[::-1]
    sorted_cords = [[centroids[i].x,centroids[i].y] for i in decreasing_indices][0:n]
    return sorted_cords
