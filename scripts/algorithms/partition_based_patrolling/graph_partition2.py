#!/usr/bin/env python3

import enum
from xxlimited import new
from matplotlib import axis
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import networkx as nx
from scipy._lib.decorator import decorator as _decorator
from yaml import Mark
import smallestenclosingcircle
import statistics
import datetime
import xml.etree.ElementTree as ET
from math import *
import os
import shutil
from shapely.geometry import Polygon as Shapely_polygon
from shapely.geometry import LineString as Shapely_line
from shapely.geometry import Point as Shapely_point
from shapely.ops import split as Shapely_split
from shapely.ops import cascaded_union as Shapely_cascaded_union
import sys
import pandas as pd
import rospkg
import alphashape
import pickle
import base_station_initial_points


graph_name = sys.argv[1]
no_of_base_stations = int(sys.argv[2])

dirname = rospkg.RosPack().get_path('mrpp_sumo')
graph_path = dirname + '/graph_ml/' + graph_name + '.graphml'
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/' + \
    graph_name + '/' + str(no_of_base_stations) + '_base_stations/'
graph = nx.read_graphml(graph_path)
all_edge_segments = []
for e in graph.edges():
    edge = graph[e[0]][e[1]]['shape'].split()
    points = np.array([p.split(',') for p in edge], dtype=float).tolist()
    for i in range(len(points)-1):
        if points[i][0] != points[i+1][0]:
            all_edge_segments.append((points[i], points[i+1]))


if os.path.exists(graph_results_path):
    shutil.rmtree(graph_results_path)
    os.makedirs(graph_results_path)
else:
    os.makedirs(graph_results_path)


__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d_clip']


# @_held_figure
def voronoi_plot_2d_clip(vor):


    if vor.points.shape[1] != 2:
        raise ValueError("voronoi diagram is not 2-D")

    center = vor.points.mean(axis=0)
    ptp_bound = hull_points.ptp(axis=0)

    finite_segments = []
    Infinite_segments = []

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            Infinite_segments.append([vor.vertices[i], far_point])

    final_regions = []
    for idx, point in enumerate(vor.points):

        enclosed_point_index = idx
        region_index = vor.point_region[enclosed_point_index]
        region_points_indices = vor.regions[region_index]

        if -1 not in region_points_indices:
            finite_region_index = vor.point_region[enclosed_point_index]
            finite_region_points_indices = vor.regions[finite_region_index]
            finite_region = vor.vertices[finite_region_points_indices]
            b = Shapely_polygon(finite_region.tolist())
            i_poly = hull.intersection(b)
            if i_poly.geom_type == 'MultiPolygon': i_poly = i_poly.geoms[0]

            x, y = i_poly.exterior.coords.xy
            finite_region = np.column_stack((x, y))[:-1, :]
            final_regions.append(finite_region)

        else:            
            Infinite_region_points_indices = region_points_indices
            while Infinite_region_points_indices.index(-1) != 0:
                Infinite_region_points_indices.append(
                    Infinite_region_points_indices.pop(0))
            Infinite_region_points_indices.remove(-1)
            Infinite_region = vor.vertices[Infinite_region_points_indices]
            point_index = np.where(vor.points == point)[0][0]
            point_ridge_indices = np.where(vor.ridge_points == point_index)[0]
            ridge_segment_vertices = np.take(
                vor.ridge_vertices, point_ridge_indices, axis=0)

            for ridge_segment_vertex, m in zip(ridge_segment_vertices, range(ridge_segment_vertices.shape[0])):

                if -1 in ridge_segment_vertex:
                    pointidx = vor.ridge_points[point_ridge_indices[m]]
                    # finite end voronoi vertex
                    i = ridge_segment_vertex[ridge_segment_vertex >= 0][0]

                    t = vor.points[pointidx[1]] - \
                        vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    if (vor.furthest_site):
                        direction = -direction
                    far_point = vor.vertices[i] + direction * ptp_bound.max()
                    if np.where(Infinite_region == vor.vertices[i])[0][0] == 0:
                        Infinite_region = np.insert(
                            Infinite_region, 0, far_point, axis=0)
                    else:
                        Infinite_region = np.append(
                            Infinite_region, [far_point], axis=0)

            b = Shapely_line(Infinite_region.tolist())
            x, y = b.coords.xy
            c = Shapely_split(hull, b)
            issue = True
            for poly in c.geoms:
                if poly.contains(Shapely_point(point)) or poly.intersects(Shapely_point(point)) or Shapely_point(point).distance(poly) < 1e-12:
                    x, y = poly.exterior.coords.xy
                    Infinite_region = np.column_stack((x, y))[:-1, :]
                    issue = False
            if issue:
                print('Issue detected')
                for poly in c.geoms:
                    print(poly)
                    print(Shapely_point(point).distance(poly), Shapely_point(point).intersects(
                        poly), poly.contains(Shapely_point(point)), poly.intersects(Shapely_point(point)))

                x, y = b.coords.xy
                plt.show()
                sys.exit()
            # appending to final set of polygon and its seed
            final_regions.append(Infinite_region)

    return np.array(final_regions, dtype=object)


def projection_on_edges(x, y):
    global all_edge_segments
    x, y = np.array([x, y])
    line_segments = np.array(all_edge_segments)

    # Calculate the distances from the point to each line segment
    x1, y1 = line_segments[:, 0, 0], line_segments[:, 0, 1]
    x2, y2 = line_segments[:, 1, 0], line_segments[:, 1, 1]
    dx, dy = x2 - x1, y2 - y1
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0, 1)
    closest = np.column_stack([x1 + t * dx, y1 + t * dy])
    dists = np.sqrt(np.sum((closest - [x, y]) ** 2, axis=1))

    # Find the index of the closest line segment
    idx = np.argmin(dists)

    return closest[idx].tolist()


def read_hull():
    hull_path = dirname+'/graph_ml/'+graph_name+'_hull'
    if os.path.exists(hull_path):
        with open(hull_path, "rb") as poly_file:
            hull = pickle.load(poly_file)
<<<<<<< HEAD
=======
    else:    
>>>>>>> 52b1e677f8c6dfe289a2a2c26f0051bfeadfce7d

    else:
        sys.exit("Hull is not generated please generate the hull first")

    hull_points = np.column_stack((hull.exterior.coords.xy))
    return hull, hull_points


def save_data():
    global base_stations_df
    for idx, p in enumerate(base_stations_coords):
        covered_nodes = []
        for node, data in graph.nodes(data=True):
            dist = ((data['x']-p[0])**2+(data['y']-p[1])**2)**0.5
<<<<<<< HEAD
            if dist <= radii[idx]:
                covered_nodes.append(node)
=======
            if dist <= radii[idx] : covered_nodes.append(node)
        
        if covered_nodes !=[]:
            base_station_dic = {'location' : [p],'Radius': radii[idx], 'covered_nodes' : [covered_nodes], 'Total_nodes_covered' : len(covered_nodes)}
            base_stations_df = pd.concat([base_stations_df,pd.DataFrame(base_station_dic,index=[idx])])        

    base_stations_df.to_csv(graph_results_path + graph_name + "_with_"+str(no_of_base_stations) + '_base_stations_edge.csv')
    np.save(graph_results_path + graph_name + "_with_"+str(no_of_base_stations) + '_base_stations_edge.npy',new_base_points)


>>>>>>> 52b1e677f8c6dfe289a2a2c26f0051bfeadfce7d

        if covered_nodes != []:
            base_station_dic = {'location': [p], 'Radius': radii[idx], 'covered_nodes': [
                covered_nodes], 'Total_nodes_covered': len(covered_nodes)}
            base_stations_df = pd.concat(
                [base_stations_df, pd.DataFrame(base_station_dic, index=[idx])])

    base_stations_df.to_csv(graph_results_path + graph_name +
                            "_with_"+str(no_of_base_stations) + '_base_stations_edge.csv')
    np.save(graph_results_path + graph_name + "_with_"+str(no_of_base_stations) +
            '_base_stations_edge.npy', base_stations_coords)


hull, hull_points = read_hull()

base_stations_df = pd.DataFrame()
radii = []

if no_of_base_stations == 1:
    x, y = hull.exterior.coords.xy
    region = np.column_stack((x, y))[:-1, :]
    c_x, c_y, r = smallestenclosingcircle.make_circle(region)
    new_base_stations_coords = [[c_x, c_y]]
    radii.append(r)
    save_data()
    sys.exit()

base_stations_coords = base_station_initial_points.random_initial_points_on_edges(
    graph_name, no_of_base_stations)
voronoi = Voronoi(base_stations_coords)

rho_old = None
rho_new = None
while True:
    a = datetime.datetime.now()
    voronoi = Voronoi(base_stations_coords)
    cliped_regions = voronoi_plot_2d_clip(voronoi)

    base_stations_coords = []
    radii = []
    rho_old = rho_new
    for idx, region in enumerate(cliped_regions):
        c_x, c_y, r = smallestenclosingcircle.make_circle(region)
        region_poly = Shapely_polygon(region)
        if region_poly.contains(Shapely_point([c_x, c_y])):
            base_stations_coords.append(projection_on_edges(c_x, c_y))
        else:
            base_stations_coords.append(
                base_station_initial_points.random_initial_points_on_edges(graph_name, 1)[0])

        radii.append(r)
        enclosing_circle = plt.Circle(
            (c_x, c_y), r, fill=False, color='#34eb43')

    rho_new = int(max(radii))
    x_axis = np.array(radii)
    x_axis = np.sort(x_axis)
    mean = statistics.mean(x_axis)
    b = datetime.datetime.now()
    c = b-a
    # print(graph_name ,'Iteration',i,'rho_new:',rho_new,' rho_old:',rho_old, '#Base stations:',len(starting_base_stations_coords),'Iteration time(secs):',c.seconds)
    if rho_new is not None and rho_old is not None:
        if abs(rho_new-rho_old)/max(rho_new, rho_old) < 0.001:
            break

save_data()
