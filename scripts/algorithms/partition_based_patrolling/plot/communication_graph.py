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
import pickle

graph_name = 'pipeline2'
range = 240
deploy_tag = 'graph'
dirname = rospkg.RosPack().get_path('mrpp_sumo')
# no_of_base_stations = np.load(dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/required_no_of_base_stations.npy')[0]
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'

G = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
tree = ET.parse(dirname + '/graph_sumo/' + graph_name +".net.xml")
root = tree.getroot()

# Edges of the graph
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
    line=dict(width=3, color='black'),
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
        showscale=False,
    
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Jet',
        reversescale=True,
        coloraxis = "coloraxis",
        color=[],
        size=20,
        colorbar=dict(
            thickness=15,
            title='Avg node Idleness (post steady state) ',
            xanchor='left',
            titleside='right'
        ),
        line_width=1))
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text


hull_path = dirname+'/graph_ml/'+graph_name+'_hull'
if os.path.exists(hull_path):
    with open(hull_path, "rb") as poly_file:
        hull = pickle.load(poly_file)
hull =hull.buffer(100)
hull_x,hull_y = hull.exterior.coords.xy
hull_x = hull_x.tolist()
hull_y = hull_y.tolist()
hull_trace = go.Scatter(
    x=hull_x, y=hull_y,
    line=dict(width=2, color='grey'),
    hoverinfo='none',
    mode='lines')
## Plot all data

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=graph_name,
                title_x = 0.5,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                # annotations=[ dict(
                #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                #     showarrow=False,
                #     xref="paper", yref="paper",
                #     x=0.005, y=-0.002 ) ],
                yaxis=dict(scaleanchor="x", scaleratio=1))
                )

# Base stations 
path = '{}/scripts/algorithms/partition_based_patrolling/deployment_results/{}/on_{}/{}m_range'.format(dirname,graph_name,deploy_tag,range)
n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
base_stations_df = pd.read_csv('{}/{}_base_stations.csv'.format(path,min(n)),converters={'location': pd.eval,'Radius': pd.eval})
base_station_logo = Image.open(dirname + '/scripts/algorithms/partition_based_patrolling/plot/base.png')

base_stations = []
icons = []
for idx, base_station in base_stations_df.iterrows():
    radius = 10
    location = base_station['location']
    base_stations.append(dict(type="circle",
    xref="x", yref="y",
    fillcolor="rgba(1,1,1,0.1)",
    x0=location[0]-base_station['Radius'], y0=location[1]-base_station['Radius'], x1=location[0]+base_station['Radius'], y1=location[1]+base_station['Radius'],
    line_color="LightSeaGreen",line_width = 0
                    ))

    icons.append(dict(
            source=base_station_logo,
            xref="x",
            yref="y",
            x=location[0]-radius,
            y=location[1]+radius,
            sizex = 2*radius,
            sizey = 2*radius
        ))

fig.update_layout(shapes=base_stations, images=icons)


file_name = graph_name + "_" + str(range) + "_range"
file_name = file_name + ".html"
plot_dir = dirname + 'scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/communication_graph/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig.write_html(plot_dir+file_name)

print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/communication_graph/' + urllib.parse.quote(file_name))


fig.show()


