import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import rospkg
import pickle
import plotly.graph_objects as go
from shapely.geometry import Polygon
import numpy as np
import random
import networkx as nx
import os 

def grid_based_initial_points(graph_name, target_n):
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    hull_path = '{}/graph_ml/{}_hull'.format(dirname,graph_name)

    if os.path.exists(hull_path):
            with open(hull_path, "rb") as poly_file:
                hull = pickle.load(poly_file)
                
    left = 1
    right = (hull.area*10)**0.5
    n = None
    sorted_cords = []
    while n != target_n:
        grid_size = (left+right)/2
        # Get the Grid for hull
        xmin, ymin, xmax, ymax = hull.bounds
        xcoords = np.arange(xmin, xmax+grid_size, grid_size)
        ycoords = np.arange(ymin, ymax+grid_size, grid_size)
        cells = []
        for x in range(len(xcoords)-1):
            for y in range(len(ycoords)-1):
                cell_poly = Polygon([(xcoords[x], ycoords[y]),
                                    (xcoords[x+1], ycoords[y]),
                                    (xcoords[x+1], ycoords[y+1]),
                                    (xcoords[x], ycoords[y+1])])
                if hull.intersects(cell_poly):
                    cells.append(cell_poly)

        # Get fully and partially contained cells in hull
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
            if cell.geom_type == "MultiPolygon":
                for geom in cell.geoms:
                    if hull.contains(geom.centroid):
                        areas = np.append(areas, geom.area)
                        centroids.append(geom.centroid)
            else:
                if hull.contains(cell.centroid):
                    areas = np.append(areas, cell.area)
                    centroids.append(cell.centroid)

        decreasing_indices = np.argsort(areas)[::-1]
        sorted_cords = [[centroids[i].x, centroids[i].y]
                        for i in decreasing_indices]
        n = len(sorted_cords)
        if  n >= target_n:
            left = grid_size
        else:
            right = grid_size
        if np.var([left,grid_size,right]) < 0.01:
            target = target+1
            left = 1
            right = (hull.area*10)**0.5
    return random.sample(sorted_cords,target_n)

def random_initial_points(graph_name, n):

    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    hull_path = '{}/graph_ml/{}_hull'.format(dirname,graph_name)

    if os.path.exists(hull_path):
            with open(hull_path, "rb") as poly_file:
                hull = pickle.load(poly_file)
    

    minx, miny, maxx, maxy = hull.bounds

    random_x = None
    random_y = None
    base_station_points = []
    for i in range(n):
        is_inside = False
        while not is_inside:
            random_x = np.random.uniform(minx, maxx, 1)[0]
            random_y = np.random.uniform(miny, maxy, 1)[0]
            is_inside = hull.contains(Point([random_x, random_y]))
        base_station_points.append([random_x, random_y])

    return base_station_points




def random_initial_points_on_edges(graph_name,n):
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'
    G = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    edges = []
    for e in G.edges(): 
        shape = G[e[0]][e[1]]['shape'].split()
        for idx ,point in enumerate(shape):
            if idx != len(shape)-1:
                p1 = shape[idx]
                p2 = shape[idx+1]
                x1 = float(p1.split(",")[0])
                y1 = float(p1.split(",")[1])
                x2 = float(p2.split(",")[0])
                y2 = float(p2.split(",")[1])
                edges.append([(x1,y1),(x2,y2)])
    total_length = sum(((x2-x1)**2 + (y2-y1)**2)**0.5 for ((x1, y1), (x2, y2)) in edges)
    selected_points = []
    for i in range(n):
        random_num = random.uniform(0, total_length)
        segment_sum = 0
        for ((x1, y1), (x2, y2)) in edges:
            segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            segment_sum += segment_length
            if segment_sum >= random_num:
                distance = random_num - (segment_sum - segment_length)
                ratio = distance / segment_length
                x = x1 + ratio * (x2-x1)
                y = y1 + ratio * (y2-y1)
                selected_points.append([x, y])
                break
    return selected_points