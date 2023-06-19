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

graph_name = 'iitb_full'
algo_name = 'mrpp_iot_250'
no_agents = 1
dirname = rospkg.RosPack().get_path('mrpp_sumo')
# no_of_base_stations = np.load(dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/required_no_of_base_stations.npy')[0]
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'

G = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
tree = ET.parse(dirname + '/graph_sumo/' + graph_name +".net.xml")
root = tree.getroot()


## Edges of the graph
edge_x = []
edge_y = []
for child in root:
    if child.tag == 'edge':
        shape = child[0].attrib['shape'].split()
        for point in shape:
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


## Nodes of the graph
node_x = []
node_y = []
for node in G.nodes():
    node_x.append(G.nodes[node]['x'])
    node_y.append(G.nodes[node]['y'])

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Jet',
        reversescale=True,
        color=[],
        size=6,
        colorbar=dict(
            thickness=15,
            title='Avg node Idleness (post steady state) ',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

## Idlness colour scheme 
idle = np.load(dirname+ "/post_process/"  + graph_name+ "/" + algo_name + "/"+ str(no_agents)+ "_agents/data_final.npy")
stamps = np.load(dirname+ "/post_process/" + graph_name+ "/" + algo_name + "/"+ str(no_agents)+ "_agents/stamps_final.npy")

steady_time_stamp = 3000
idle = idle[np.argwhere(stamps>steady_time_stamp)[0][0]:]  # Taking idlness values after steady state
avg_idle = np.average(idle,axis=0)
node_text = []
for idx, node in enumerate(G.nodes()):
    node_text.append('Avg Idleness: '+str(avg_idle[idx]))

node_trace.marker.color = avg_idle
node_trace.text = node_text





## Plot all data

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=graph_name +' Avg node Idleness (post steady state)  with '+ str(no_agents) +' Agents',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                yaxis=dict(scaleanchor="x", scaleratio=1))
                )

## Base stations 

if 'mrpp_iot' in algo_name:
    range = int(algo_name.split('_')[-1])
    base_stations_df = pd.read_csv(graph_results_path + graph_name + '/' + str(range) + '_range_base_stations.csv',converters={'location': pd.eval,'Radius': pd.eval})
    base_station_logo = Image.open(dirname + '/scripts/algorithms/partition_based_patrolling/plot/base.png')

    base_stations = []
    icons = []
    for idx, base_station in base_stations_df.iterrows():
        radius = base_station['Radius']
        location = base_station['location']
        base_stations.append(dict(type="circle",
        xref="x", yref="y",
        fillcolor="rgba(1,1,1,0.1)",
        x0=location[0]-radius, y0=location[1]-radius, x1=location[0]+radius, y1=location[1]+radius,
        line_color="LightSeaGreen",line_width = 0
                        ))

        icons.append(dict(
                source=base_station_logo,
                xref="x",
                yref="y",
                x=location[0]-radius/10,
                y=location[1]+radius/10,
                sizex = radius/5,
                sizey = radius/5
            ))

    fig.update_layout(shapes=base_stations, images=icons)

fig.show()
fig.write_html(dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name +"_"+  ".html")