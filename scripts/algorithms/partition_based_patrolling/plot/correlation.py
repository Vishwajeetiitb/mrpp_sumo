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

dirname = rospkg.RosPack().get_path('mrpp_sumo')
no_agents = 7
algo_list = ['iot_communication_network_150','iot_communication_network_250','iot_communication_network_350','iot_communication_network_500','iot_communication_network_10000']
# available_comparisons = ['Idleness', 'Worst Idleness']
# comparison_parameter_index = 0
# scater_nodes_algo_index =  2# putting scatter for only one algo is better otherwise mess put -1 if don't require node scatter
row_size = 2
col_size = 3
graph_name = 'iit_bombay'
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

fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=algo_list)
fig.update_layout(title='Correlation of Instantaneous Graph Idle between agents ({})'.format(graph_name),title_x=0.5)

for idx,algo_name in enumerate(algo_list):
    agent_masterdata = np.load(dirname+ "/post_process/"  + graph_name+ "/run0/"+ algo_name + "/" + str(no_agents)+ "_agents/agent_masterdata_final.npz")['arr_0']
    stamps = np.load(dirname+ "/post_process/" + graph_name+ "/run0/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npz")['arr_0']
    agent_masterdata_graph_idle = np.transpose(np.mean(agent_masterdata,axis=2))
    # print(agent_masterdata_graph_idle.shape)
    corr= np.corrcoef(agent_masterdata_graph_idle)
    fig.add_trace(go.Heatmap(z=corr,text=np.around(corr,2),texttemplate="%{text}",showlegend=(False if idx==0 else False),showscale=(True if idx==0 else False),zmax=1,zmin=0),row=int(idx/col_size)+1,col=idx%col_size+1)
    
iplot(fig)