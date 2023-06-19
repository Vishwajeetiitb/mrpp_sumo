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
import plotly.graph_objs as go
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
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

graph_names = ['iit_delhi','stanford','iit_bombay']
dirname = rospkg.RosPack().get_path('mrpp_sumo')
# no_of_base_stations = np.load(dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/required_no_of_base_stations.npy')[0]
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'
color_list = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

# for i in graph_names:
#     for j in []: print(j+i)

subplot_names = [j+i for i in graph_names for j in ['Node Degree Distribution for ','Edge length distribution for ']]
fig = make_subplots(rows=3, cols=2,subplot_titles=subplot_names,horizontal_spacing = 0.05,vertical_spacing=0.05)
fig.update_layout(title='Map Geometry Distribution',title_x=0.5)

edge_dist_data = []
node_dist_data = []
for idx,graph_name in enumerate(graph_names):
    G = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    ## Edges of the graph
    edge_x = []
    edge_y = []
    edge_lengths = []
    for e in G.edges():
        # shape = G[e[0]][e[1]]['shape'].split()
        edge_lengths.append(G[e[0]][e[1]]['length'])
        # for point in shape:
        #     point = pd.eval(point)
        #     edge_x.append(point[0])
        #     edge_y.append(point[1])
        # edge_x.append(None)
        # edge_y.append(None)

    ## Nodes of the graph
    node_x = []
    node_y = []
    for node in G.nodes():
        node_x.append(G.nodes[node]['x'])
        node_y.append(G.nodes[node]['y'])

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        
    edge_dist_data.append(edge_lengths)
    node_dist_data.append(node_adjacencies)
    fig2 = ff.create_distplot([edge_lengths],[graph_name],show_hist=False,show_rug=False)

    fig.add_trace(go.Scatter(fig2['data'][0],legendgroup=idx,marker_color=color_list[idx],showlegend=False), row=idx+1, col=2)
    fig2 = ff.create_distplot([node_adjacencies],[graph_name],show_hist=False,show_rug=False)
    fig.add_trace(go.Scatter(fig2['data'][0],legendgroup=idx,showlegend=False,marker_color=color_list[idx]), row=idx+1, col=1)

fig.update_layout(xaxis2 = dict(range=[0, 600]),xaxis4 = dict(range=[0, 600]),xaxis6 = dict(range=[0, 600])) 
fig.update_layout(xaxis1 = dict(range=[0.9, 5.1]),xaxis3 = dict(range=[0.9, 5.1]),xaxis5 = dict(range=[0.9, 5.1])) 
fig.show()
