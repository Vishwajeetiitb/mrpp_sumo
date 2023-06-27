#!/usr/bin/env python3

import numpy as np
from scipy.spatial import Voronoi
import networkx as nx
from yaml import Mark
import smallestenclosingcircle
import statistics
import datetime
import xml.etree.ElementTree as et
from math import *
import os
import shutil
from shapely.geometry import Polygon as Shapely_polygon
from shapely.geometry import LineString as Shapely_line
from shapely.geometry import Point as Shapely_point
from shapely.ops import split as Shapely_split
import sys
import pandas as pd
import rospkg
import alphashape
import pickle
import random
from time import sleep
import sumolib

class Deployer:
    def __init__(self, graph_name) -> None:
        self.graph_name = graph_name
        self.dirname = rospkg.RosPack().get_path('mrpp_sumo')

        self.graph = nx.read_graphml(
            '{}/graph_ml/{}.graphml'.format(self.dirname, self.graph_name))
        
        self.net_file  = '{}/graph_sumo/{}.net.xml'.format(self.dirname,self.graph_name)

        self.Compute_hull()
        self.deploy_tag = 'edge' # This variable is for knowing where deployed on edge or graph
        self.results_path = '{}/scripts/algorithms/partition_based_patrolling/deployment_results/{}'.format(self.dirname,self.graph_name)
        # if os.path.exists(self.results_path):
        #     shutil.rmtree(self.results_path)
        #     os.makedirs(self.results_path)
        # else:
        #     os.makedirs(self.results_path)
        


    def Compute_hull(self):
        print('Computing hull')
        graph_points = []
        for node, data in self.graph.nodes(data=True):
            graph_points.append(np.array((data['x'], data['y'])))
        edge_parse = et.parse(self.net_file)
        edge_root = edge_parse.getroot()
        for e in self.graph.edges():
            graph_points.append(np.array([self.graph.nodes[e[1]]['x'], self.graph.nodes[e[1]]['y']]))
            if 'shape' in self.graph[e[0]][e[1]]:
                shape = self.graph[e[0]][e[1]]['shape'].split()
                for idx, point in enumerate(shape):
                    p1 = shape[idx]
                    x1 = float(p1.split(",")[0])
                    y1 = float(p1.split(",")[1])
                    graph_points.append(np.array([x1, y1]))

        self.hull_path = '{}/graph_ml/{}_hull'.format(
            self.dirname, self.graph_name)

        if os.path.exists(self.hull_path):
            with open(self.hull_path, "rb") as poly_file:
                self.hull = pickle.load(poly_file)
        else:
            self.hull = alphashape.alphashape(graph_points).buffer(5)
            with open(self.hull_path, "wb") as poly_file:
                pickle.dump(self.hull, poly_file, pickle.HIGHEST_PROTOCOL)

        self.hull_points = np.column_stack((self.hull.exterior.coords.xy))
        print('Hull computed!')

    def Get_bounded_voronoi_regions(self, vor):

        if vor.points.shape[1] != 2:
            raise ValueError("voronoi diagram is not 2-D")

        center = vor.points.mean(axis=0)
        ptp_bound = self.hull_points.ptp(axis=0)

        finite_segments = []
        Infinite_segments = []

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end voronoi vertex

                t = vor.points[pointidx[1]] - \
                    vor.points[pointidx[0]]  # tangent
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

            if -1 not in region_points_indices and len(region_points_indices)>2:
                finite_region_points_indices = region_points_indices
                finite_region = vor.vertices[finite_region_points_indices]
                b = Shapely_polygon(finite_region.tolist())
                i_poly = self.hull.intersection(b)
                if i_poly.geom_type == 'MultiPolygon':
                    i_poly = i_poly.geoms[0]

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
                point_ridge_indices = np.where(
                    vor.ridge_points == point_index)[0]
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
                        far_point = vor.vertices[i] + \
                            direction * ptp_bound.max()
                        if np.where(Infinite_region == vor.vertices[i])[0][0] == 0:
                            Infinite_region = np.insert(
                                Infinite_region, 0, far_point, axis=0)
                        else:
                            Infinite_region = np.append(
                                Infinite_region, [far_point], axis=0)

                b = Shapely_line(Infinite_region.tolist())
                x, y = b.coords.xy
                c = Shapely_split(self.hull, b)
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
                    sys.exit()
                # appending to final set of polygon and its seed
                final_regions.append(Infinite_region)

        return np.array(final_regions, dtype=object)

    def Projection_on_edges(self, x, y):
        x, y = np.array([x, y])
        line_segments = np.array(self.all_edge_segments)

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

    def Deploy_on_edges(self,no_of_base_stations):
        print('Deploying on')
        self.deploy_tag = 'edge'
        self.all_edge_segments = []
        edge_parse = et.parse(self.net_file)
        edge_root = edge_parse.getroot()
        for e in edge_root.iter('edge'):
            edge = e[0].attrib['shape'].split()
            points = np.array([p.split(',')
                              for p in edge], dtype=float).tolist()
            for i in range(len(points)-1):
                if points[i][0] != points[i+1][0]:
                    self.all_edge_segments.append((points[i], points[i+1]))

        self.base_stations_coords = self.Random_initial_points_on_edges(
            no_of_base_stations)
        voronoi = Voronoi(self.base_stations_coords)

        rho_old = None
        rho_new = None
        while True:
            a = datetime.datetime.now()
            voronoi = Voronoi(self.base_stations_coords)
            cliped_regions = self.Get_bounded_voronoi_regions(voronoi)

            self.base_stations_coords = []
            self.radii = []
            rho_old = rho_new
            for idx, region in enumerate(cliped_regions):
                c_x, c_y, r = smallestenclosingcircle.make_circle(region)
                region_poly = Shapely_polygon(region)
                if region_poly.contains(Shapely_point([c_x, c_y])):
                    self.base_stations_coords.append(
                        self.Projection_on_edges(c_x, c_y))
                else:
                    self.base_stations_coords.append(
                        self.Random_initial_points_on_edges(1)[0])

                self.radii.append(r)

            rho_new = int(max(self.radii))
            x_axis = np.array(self.radii)
            x_axis = np.sort(x_axis)
            mean = statistics.mean(x_axis)
            b = datetime.datetime.now()
            c = b-a
            # print(self.graph_name ,'Iteration',i,'rho_new:',rho_new,' rho_old:',rho_old, '#Base stations:',no_of_base_stations,'Iteration time(secs):',c.seconds)
            if rho_new is not None and rho_old is not None:
                if abs(rho_new-rho_old)/max(rho_new, rho_old) < 0.001:
                    return self.base_stations_coords,int(max(self.radii))

    def Deploy_on_graph(self, no_of_base_stations):
        print('Deploying')
        self.deploy_tag = 'graph'
        self.all_edge_segments = []
        edge_parse = et.parse(self.net_file)
        edge_root = edge_parse.getroot()
        for e in edge_root.iter('edge'):
            edge = e[0].attrib['shape'].split()
            points = np.array([p.split(',')
                              for p in edge], dtype=float).tolist()
            for i in range(len(points)-1):
                if points[i][0] != points[i+1][0]:
                    self.all_edge_segments.append((points[i], points[i+1]))

        self.base_stations_coords = self.Random_initial_points(no_of_base_stations)
        voronoi = Voronoi(self.base_stations_coords)

        rho_old = None
        rho_new = None
        while True:
            a = datetime.datetime.now()
            voronoi = Voronoi(self.base_stations_coords)
            cliped_regions = self.Get_bounded_voronoi_regions(voronoi)

            self.base_stations_coords = []
            self.radii = []
            rho_old = rho_new
            for idx, region in enumerate(cliped_regions):
                c_x, c_y, r = smallestenclosingcircle.make_circle(region)
                region_poly = Shapely_polygon(region)
                if region_poly.contains(Shapely_point([c_x, c_y])):
                    self.base_stations_coords.append([c_x, c_y])
                else:
                    self.base_stations_coords.append(
                        self.Random_initial_points(1)[0])

                self.radii.append(r)

            rho_new = int(max(self.radii))
            x_axis = np.array(self.radii)
            x_axis = np.sort(x_axis)
            mean = statistics.mean(x_axis)
            b = datetime.datetime.now()
            c = b-a
            # print(self.graph_name ,'Iteration',i,'rho_new:',rho_new,' rho_old:',rho_old, '#Base stations:',no_of_base_stations,'Iteration time(secs):',c.seconds)
            if rho_new is not None and rho_old is not None:
                if abs(rho_new-rho_old)/max(rho_new, rho_old) < 0.001:
                    return self.base_stations_coords,max(self.radii)

    def Grid_based_initial_points(self, target_n):

        left = 1
        right = (self.hull.area*10)**0.5
        n = None
        sorted_cords = []
        while n != target_n:
            grid_size = (left+right)/2
            # Get the Grid for hull
            xmin, ymin, xmax, ymax = self.hull.bounds
            xcoords = np.arange(xmin, xmax+grid_size, grid_size)
            ycoords = np.arange(ymin, ymax+grid_size, grid_size)
            cells = []
            for x in range(len(xcoords)-1):
                for y in range(len(ycoords)-1):
                    cell_poly = Shapely_polygon([(xcoords[x], ycoords[y]),
                                                 (xcoords[x+1], ycoords[y]),
                                                 (xcoords[x+1], ycoords[y+1]),
                                                 (xcoords[x], ycoords[y+1])])
                    if self.hull.intersects(cell_poly):
                        cells.append(cell_poly)

            # Get fully and partially contained cells in hull
            fully_contained_cells = []
            for cell in cells:
                if cell.within(self.hull):
                    fully_contained_cells.append(cell)

            partially_contained_cells = []
            for cell in cells:
                intersection = cell.intersection(self.hull)
                if intersection.area > 0 and not cell.within(self.hull):
                    partially_contained_cells.append(intersection)

            contained_cells = fully_contained_cells + partially_contained_cells

            centroids = []
            areas = np.array([])
            for cell in contained_cells:
                if cell.geom_type == "MultiPolygon":
                    for geom in cell.geoms:
                        if self.hull.contains(geom.centroid):
                            areas = np.append(areas, geom.area)
                            centroids.append(geom.centroid)
                else:
                    if self.hull.contains(cell.centroid):
                        areas = np.append(areas, cell.area)
                        centroids.append(cell.centroid)

            decreasing_indices = np.argsort(areas)[::-1]
            sorted_cords = [[centroids[i].x, centroids[i].y]
                            for i in decreasing_indices]
            n = len(sorted_cords)
            if n >= target_n:
                left = grid_size
            else:
                right = grid_size
            if np.var([left, grid_size, right]) < 0.01:
                target = target+1
                left = 1
                right = (self.hull.area*10)**0.5
        return random.sample(sorted_cords, target_n)

    def Random_initial_points(self, n):

        minx, miny, maxx, maxy = self.hull.bounds

        random_x = None
        random_y = None
        base_station_points = []
        for i in range(n):
            is_inside = False
            while not is_inside:
                random_x = np.random.uniform(minx, maxx, 1)[0]
                random_y = np.random.uniform(miny, maxy, 1)[0]
                is_inside = self.hull.contains(
                    Shapely_point([random_x, random_y]))
            base_station_points.append([random_x, random_y])

        return base_station_points

    def Random_initial_points_on_edges(self, n):
        edges = []
        edge_parse = et.parse(self.net_file)
        edge_root = edge_parse.getroot()
        for e in edge_root.iter('edge'):
            shape = e[0].attrib['shape'].split()
            for idx, point in enumerate(shape):
                if idx != len(shape)-1:
                    p1 = shape[idx]
                    p2 = shape[idx+1]
                    x1 = float(p1.split(",")[0])
                    y1 = float(p1.split(",")[1])
                    x2 = float(p2.split(",")[0])
                    y2 = float(p2.split(",")[1])
                    edges.append([(x1, y1), (x2, y2)])
        total_length = sum(((x2-x1)**2 + (y2-y1)**2) **
                           0.5 for ((x1, y1), (x2, y2)) in edges)
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

    def save_data(self,device_range,n):
        base_stations_df = pd.DataFrame()

        for idx, p in enumerate(self.base_stations_coords):
            covered_nodes = []
            for node, data in self.graph.nodes(data=True):
                dist = ((data['x']-p[0])**2+(data['y']-p[1])**2)**0.5
                if dist <= self.radii[idx]:
                    covered_nodes.append(node)

            if covered_nodes != []:
                base_station_dic = {'location': [p], 'Radius': self.radii[idx], 'covered_nodes': [
                    covered_nodes], 'Total_nodes_covered': len(covered_nodes)}
                base_stations_df = pd.concat(
                    [base_stations_df, pd.DataFrame(base_station_dic, index=[idx])])
        path = '{}/on_{}/{}m_range'.format(self.results_path,self.deploy_tag,device_range)
        if not os.path.exists(path):
            os.makedirs(path)
        base_stations_df.to_csv('{}/{}_base_stations.csv'.format(path,n))

if __name__ == '__main__':
    deployer = Deployer('pipeline3')
    device_ranges = sorted([240],reverse=True)
    start_n = 3
    deploy_tag = 'graph'
    for device_range in device_ranges:
        vals = []
        for i in range (100):
            n = start_n
            while True:
                if deploy_tag == 'edge': points,r_max = deployer.Deploy_on_edges(no_of_base_stations=n)
                elif deploy_tag == 'graph' : points,r_max = deployer.Deploy_on_graph(no_of_base_stations=n)
                if r_max<=device_range :
                    if n not in vals:
                        print('{} Base stations for {}m communication range'.format(n,device_range))
                        vals.append(n)
                        deployer.save_data(device_range,n)
                    break
                else : n +=1
        start_n = min(vals)
        print('{} these are the set of base stations for {}m communication range'.format(vals,device_range))