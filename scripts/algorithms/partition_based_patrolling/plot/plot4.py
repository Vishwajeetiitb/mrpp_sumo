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
import urllib.parse
import chart_studio.plotly as py
from plotly.offline import iplot
import sys
import subprocess


dirname = rospkg.RosPack().get_path('mrpp_sumo')

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
graph_name = 'pipeline3'

deploy_tag = ['graph']
bot = 15
device_ranges = [100,240,1000]
device_names = ['Zigbee','BLE V5', 'LoRa']
row_size = 1
col_size = 1
fig = make_subplots(rows=row_size, cols=col_size)

for idx,tag in enumerate(deploy_tag):
    for m, device_range in enumerate(device_ranges):
        path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,tag,device_range)
        n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
        results_path = '{}/{}_base_stations/{}_agents/run_0'.format(path,min(n),bot)
        row_id = int(idx/col_size) + 1
        col_id = idx%col_size + 1
        idle = np.load('{}/data_final.npz'.format(results_path))['arr_0']
        stamps = np.load('{}/stamps_final.npz'.format(results_path))['arr_0']
        val = np.average(idle,axis=1).cumsum()
        val = val/np.arange(1,val.shape[0]+1)
        fig.add_trace(go.Scatter(x=stamps, y=val,mode='lines',marker=dict(color=color_list[m]),showlegend=(True if idx==0 else False),name='{}'.format(device_names[m]),legendgroup=m),row=row_id,col=col_id)

# fig.update_yaxes(range = [0,2200])
fig.update_layout(title_text = "Instantaneous Graph Idleness for {}_Agents".format(bot),title_x=0.5)
fig.show()