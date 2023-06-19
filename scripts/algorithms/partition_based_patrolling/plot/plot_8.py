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
graph_name = 'iit_bombay'

tag = 'edge'
no_of_bots = [1,3,6,9,12,15]
device_ranges = [250,350,500]

fig = make_subplots(rows=1, cols=3,subplot_titles=['{}m communication range'.format(i) for i in device_ranges])

for idx,device_range in enumerate(device_ranges):
    path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,tag,device_range)
    n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
    for m, bot in enumerate(no_of_bots):
        results_path = '{}/{}_base_stations/{}_agents/run_0'.format(path,min(n),bot)
        row_id =1
        col_id = idx+1
        idle = np.load('{}/data_final.npz'.format(results_path))['arr_0']
        stamps = np.load('{}/stamps_final.npz'.format(results_path))['arr_0']
        val = np.average(idle,axis=1).cumsum()
        val = val/np.arange(1,val.shape[0]+1)
        fig.add_trace(go.Scatter(x=stamps, y=val,mode='lines',marker=dict(color=color_list[m]),showlegend=(True if idx==0 else False),name='{}_agents'.format(bot),legendgroup=m),row=row_id,col=col_id)

fig.update_layout(title_text = "Instantaneous Graph Idleness for Base station deployement on {}".format(tag),title_x=0.5)
fig.show()