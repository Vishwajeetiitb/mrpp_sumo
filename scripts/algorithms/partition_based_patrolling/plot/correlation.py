from fileinput import filename
from platform import node
from turtle import color, stamp, title
from unicodedata import name
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import rospkg
import pandas as pd
import os
from slugify import slugify
import urllib.parse
import chart_studio.plotly as py
from plotly.offline import iplot
import itertools
from plotly.graph_objs import *

graph_name = 'pipeline2'
dirname = rospkg.RosPack().get_path('mrpp_sumo')
no_agents = 6
deploy_tag = 'graph'
device_ranges = [100,240,1000]
device_names = ['Zigbee','BLE','LoRa']
# available_comparisons = ['Idleness', 'Worst Idleness']
# comparison_parameter_index = 0
# scater_nodes_algo_index =  2# putting scatter for only one algo is better otherwise mess put -1 if don't require node scatter
row_size = 1
col_size = 3
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
# names = algo_list
# names = names.extend(['Range vs Deviation'])

fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=device_names)
fig.update_layout(title='Correlation of Instantaneous Graph Idle between agents ({})'.format(graph_name),title_x=0.5)

for idx,device_range in enumerate(device_ranges):
    path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,deploy_tag,device_range)
    n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
    df = pd.DataFrame()
    results_path = '{}/{}_base_stations/{}_agents/run_0'.format(path,min(n),no_agents)
    agent_masterdata = np.load('{}/agent_masterdata_final.npz'.format(results_path))['arr_0']
    print(agent_masterdata.shape)
    stamps = np.load('{}/stamps_final.npz'.format(results_path))['arr_0']
    agent_masterdata_graph_idle = np.transpose(np.mean(agent_masterdata,axis=2))
    # print(agent_masterdata_graph_idle.shape)
    corr= np.corrcoef(agent_masterdata_graph_idle)
    fig.add_trace(go.Heatmap(z=corr,text=np.around(corr,2),texttemplate="%{text}",showlegend=(False if idx==0 else False),showscale=(True if idx==0 else False),zmax=1,zmin=0),row=int(idx/col_size)+1,col=idx%col_size+1)

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
# fig.update_xaxes(scaleanchor = "y",scaleratio = 1,showgrid=False)
# fig.update_yaxes(scaleanchor = "x",scaleratio = 1,showgrid=False)
# fig.update_layout(yaxis1 = dict(range=[0, 6]),yaxis2 = dict(range=[0, 6]),yaxis3 = dict(range=[0, 6])) 
fig.update_layout(height=700,width=2100)
# fig.update_layout(xaxis_title = dict(text=))
fig.update_layout(
    xaxis1=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    ),

    xaxis2=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    ),

    xaxis3=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    ),

    yaxis1=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    ),

    yaxis2=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    ),

    yaxis3=dict(
        tickmode='array',
        tickvals = [i for i in range(no_agents)],
        ticktext=['Agent_{}'.format(i+1) for i in range(no_agents)]
    )
)
iplot(fig)