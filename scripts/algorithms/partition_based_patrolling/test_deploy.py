import random
import plotly.graph_objects as go
from turtle import width
import plotly.graph_objects as go
import rospy
import rospkg
import networkx as nx
import xml.etree.ElementTree as ET
from ast import literal_eval
import pandas as pd
import numpy as np
from turtle import width
import plotly.graph_objects as go
import rospy
import rospkg
import networkx as nx
import xml.etree.ElementTree as ET
from ast import literal_eval
import pandas as pd
import numpy as np
from PIL import Image
import os
import urllib.parse


graph_name = 'parliament'
Range = 500
dirname = rospkg.RosPack().get_path('mrpp_sumo')
# no_of_base_stations = np.load(dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/required_no_of_base_stations.npy')[0]
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'

G = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')



def select_random_points_on_edges(graph_name,n):
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

# Define line segments 

## Edges of the graph
edge_x = []
edge_y = []
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
        point = pd.eval(point)
        edge_x.append(point[0])
        edge_y.append(point[1])
    edge_x.append(None)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='black'),
    hoverinfo='none',
    mode='lines')



# Generate random points
random_points = select_random_points_on_edges(graph_name,10)
# Plot 
print(random_points)

x_points, y_points = zip(*random_points)



# # # Create scatter plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', name='Line Segments',line=dict(color='black',width=0.5)))
# fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='markers', name='Random Points'))

# # # Set axis labels and title
# fig.update_layout(title='Random Points on Line Segments',
#                   xaxis_title='X Coordinate', yaxis_title='Y Coordinate')
# fig.update_yaxes(scaleanchor="x",scaleratio=1)

# # # Show plot
# fig.show()




