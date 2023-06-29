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
graph_name = 'pipeline1'

tag = 'graph'
bot = 6
device_ranges = [100,240,1000]

fig = make_subplots(rows=1, cols=3,subplot_titles=['{}m Communication Range'.format(i) for i in device_ranges])

for idx,device_range in enumerate(device_ranges):
    path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,tag,device_range)
    for m,filename in enumerate(os.listdir(path)):
        results_path = '{}/{}/{}_agents/run_0'.format(path,filename,bot)
        row_id =1
        col_id = idx+1
        idle = np.load('{}/data_final.npz'.format(results_path))['arr_0']
        stamps = np.load('{}/stamps_final.npz'.format(results_path))['arr_0']
        val = np.average(idle,axis=1).cumsum()
        val = val/np.arange(1,val.shape[0]+1)
        fig.add_trace(go.Scatter(x=stamps, y=val,mode='lines',name='{}'.format(filename)),row=row_id,col=col_id)

fig.update_layout(title_text = "Instantaneous Graph Idleness for Base stations on {}".format(tag),title_x=0.5)
fig.show()