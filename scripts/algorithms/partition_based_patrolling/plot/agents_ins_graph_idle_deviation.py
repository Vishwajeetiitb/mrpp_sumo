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
graph_name = 'iit_delhi'
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

fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=[i for i in itertools.chain(algo_list,['Range vs Deviation'])])
fig.update_layout(title='Relative standard deviation of Graph Idle data among agents',title_x=0.5)

graph_std = []
for idx,algo_name in enumerate(algo_list):
    agent_masterdata = np.load(dirname+ "/post_process/"  + graph_name+ "/run0/"+ algo_name + "/" + str(no_agents)+ "_agents/agent_masterdata_final.npz")['arr_0']
    stamps = np.load(dirname+ "/post_process/" + graph_name+ "/run0/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npz")['arr_0']
    agent_masterdata_graph_idle = np.transpose(np.mean(agent_masterdata,axis=2))
    # print(agent_masterdata_graph_idle)
    # std = np.std(agent_masterdata_graph_idle,axis=0)[1:]/np.average(agent_masterdata_graph_idle,axis=0)[1:]
    std = np.std(agent_masterdata_graph_idle,axis=0)[1:]
    graph_std.append(np.average(std))
    # std = std.cumsum()
    # std = std/np.arange(1,std.shape[0]+1)
    # fig.add_trace(go.Scatter(x=stamps, y=std,mode='lines',marker=dict(color=color_list[-1],size=10),name='standard deviation percentage',showlegend=(False if idx==0 else False)),row=int(idx/col_size)+1,col=idx%col_size+1)
    for m,agent_data in enumerate(agent_masterdata_graph_idle):
        agent_data = agent_data.cumsum()/np.arange(1,agent_data.shape[0]+1)
        fig.add_trace(go.Scatter(x=stamps, y=agent_data,mode='lines',marker=dict(color=color_list[m],size=10),legendgroup=m+1,name='Agent '+str(m+1),showlegend=(True if idx==0 else False),opacity=1),row=int(idx/col_size)+1,col=idx%col_size+1)
graph_std = graph_std[0:len(graph_std)-1]
fig.add_trace(go.Scatter(x=[150,250,350,500],y=graph_std,mode='markers+lines+text',marker_color='red',text=np.around(graph_std,3),textposition='top center',showlegend=False),row=row_size,col=col_size)
iplot(fig)

# fig2 = go.Figure()
# agents_list = [1,3,5,7,9,11,13,15]
# for no_agents in agents_list:
#     graph_std = []
#     for idx,algo_name in enumerate(algo_list):
#         agent_masterdata = np.load(dirname+ "/post_process/"  + graph_name+ "/run0/"+ algo_name + "/" + str(no_agents)+ "_agents/agent_masterdata_final.npz")['arr_0']
#         stamps = np.load(dirname+ "/post_process/" + graph_name+ "/run0/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npz")['arr_0']
#         agent_masterdata_graph_idle = np.transpose(np.mean(agent_masterdata,axis=2))
#         # print(agent_masterdata_graph_idle)
#         std = np.std(agent_masterdata_graph_idle,axis=0)[1:]/np.average(agent_masterdata_graph_idle,axis=0)[1:]
#         std = np.std(agent_masterdata_graph_idle,axis=0)[1:]
#         std = std.cumsum()
#         std = std/np.arange(1,std.shape[0]+1)
#     graph_std = graph_std[0:len(graph_std)-1]
#     fig2.add_trace(go.Scatter(x=[150,250,350,500],y=graph_std,mode='markers+lines+text',text=np.around(graph_std,3),textposition='top center',name=str(no_agents)+' Agents'))


# iplot(fig2)