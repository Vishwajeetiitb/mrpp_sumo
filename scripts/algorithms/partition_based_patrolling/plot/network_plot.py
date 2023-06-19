from turtle import title, width
from typing import Dict
import plotly.graph_objects as go
import rospkg
import networkx as nx
import xml.etree.ElementTree as ET
from ast import literal_eval
import pandas as pd
import numpy as np
from PIL import Image
from plotly.subplots import make_subplots
import os
import urllib.parse
import plotly.io as pio


graph_name = 'iitb_full'
algo_list = ['iot_communication_network_150','iot_communication_network_250','iot_communication_network_350','iot_communication_network_500','iot_communication_network_10000']
row_size = 2
col_size = 3
no_agents = 9
steady_time_stamp = 3000

# available_comparisons = ['Average Node Idleness', 'Worst Node Idleness']
# comparison_parameter_index = 0

dirname = rospkg.RosPack().get_path('mrpp_sumo')
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'


################################################################# Graph Configuration ###############################################################################
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
        showscale=False,
    
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='Jet',
        reversescale=True,
        coloraxis = "coloraxis",
        color=[],
        size=3,
        colorbar=dict(
            thickness=15,
            title='Avg node Idleness (post steady state) ',
            xanchor='left',
            titleside='right'
        ),
        line_width=1))

def get_base_station_shape(algo):
    range = int(algo.split('_')[-1])
    base_stations_df = pd.read_csv(graph_results_path + graph_name + '/' + str(range) + '_range_base_stations.csv',converters={'location': pd.eval,'Radius': pd.eval})
    base_station_logo = Image.open(dirname + '/scripts/algorithms/partition_based_patrolling/plot/base.png')
    base_stations = []
    icons = []
    for idx, base_station in base_stations_df.iterrows():
        radius = base_station['Radius']
        location = base_station['location']
        base_stations.append(dict(type="circle",
                                    xref="x",
                                    yref="y",
                                    fillcolor="rgba(1,1,1,0.06)",
                                    x0=location[0]-radius,
                                    y0=location[1]-radius,
                                    x1=location[0]+radius,
                                    y1=location[1]+radius,
                                    line_color="LightSeaGreen",line_width = 0))

        icons.append(dict(
                source=base_station_logo,
                xref="x",
                yref="y",
                x=location[0]-radius/10,
                y=location[1]+radius/10,
                sizex = radius/5,
                sizey = radius/5
            ))
    return base_stations,icons

###################################################################### Subplots ####################################################################################
# title='Average Node Idleness Network Plot for ' + str(no_agents) + ' Agents'
fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=[i for i in algo_list],vertical_spacing=0.1)
fig.update_layout(title='Graph simulation scenearios' ,
                titlefont_size=16,
                title_x=0.5,
                coloraxis = {'colorscale':'Jet'},
                
                )

for m,algo_name in enumerate(algo_list):
    idle = np.load(dirname+ "/post_process/"  + graph_name+ "/"+ algo_name + "/" + str(no_agents)+ "_agents/data_final.npy")
    stamps = np.load(dirname+ "/post_process/" + graph_name+ "/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npy")
    idle = idle[np.argwhere(stamps>steady_time_stamp)[0][0]:]  # Taking idlness values after steady state
    avg_idle = np.average(idle,axis=0)
    node_text = []
    for idx, node in enumerate(G.nodes()):
        node_text.append('Avg Idleness: '+str(avg_idle[idx]))

    node_trace.marker.color = avg_idle
    node_trace.text = node_text
    

    node_trace.showlegend = False
    edge_trace.showlegend = False
    fig.add_trace(edge_trace,row=int(m/col_size)+1,col=m%col_size+1)
    fig.add_trace(node_trace,row=int(m/col_size)+1,col=m%col_size+1,)

    if 'iot' in algo_name:
        base_station_shape,icons = get_base_station_shape(algo_name)
        for shape,img in zip(base_station_shape,icons):
            fig.add_shape(shape,row=int(m/col_size)+1,col=m%col_size+1)
            fig.add_layout_image(img,row=int(m/col_size)+1,col=m%col_size+1)


    
    









## Plot all data

# fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 title=graph_name +' Avg node Idleness (post steady state)  with '+ str(no_agents) +' Agents',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 # annotations=[ dict(
#                 #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                 #     showarrow=False,
#                 #     xref="paper", yref="paper",
#                 #     x=0.005, y=-0.002 ) ],
#                 yaxis=dict(scaleanchor="x", scaleratio=1))
#                 )

## Base stations 


    

#     fig.update_layout(shapes=base_stations, images=icons)


file_name = ""
for idx,algo in enumerate(algo_list):
    if not idx:
        file_name = algo
    else:
        file_name = file_name + " | " + algo
        
file_name = file_name + ".html"
plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/network_plot/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig.write_html(plot_dir+file_name)

print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/network_plot/' + urllib.parse.quote(file_name))
# fig.update_yaxes(range = [1,1])
# fig.update_layout(axis)

# fig.update_layout(yaxis1 = dict(range=[0, 2000]))
# fig.update_layout(yaxis2 = dict(range=[0, 2000]))
# fig.update_layout(yaxis3 = dict(range=[0, 2000]))
# fig.update_layout(yaxis4 = dict(range=[0, 2000]))
# fig.update_layout(yaxis5 = dict(range=[0, 2000]))
# fig['layout'].update(height=600, width=600, title='Stacked Subplots with Shared X-Axes')
# fig['layout'].update(height=2000, width=2000, title='Subplots with Shared X-Axes')

fig.update_xaxes(scaleanchor = "y",scaleratio = 1)
fig.update_yaxes(scaleanchor = "x",scaleratio = 1)
fig.write_image("./fig1.jpg",width = 3840, height = 2160)
fig.show()
