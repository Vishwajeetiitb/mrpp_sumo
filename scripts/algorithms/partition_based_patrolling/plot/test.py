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


deploy_tag = ['edge','graph']
no_of_bots = [1,3,6,9,12,15]
device_ranges = sorted([500,350,250],reverse=True)
configuration = {'deploy_tag':'S','device_ranges':'F','no_of_base_stations':'M','no_of_bots':'F'}

if configuration['device_ranges'] + configuration['no_of_base_stations'] =='MS': sys.exit('Invalid Configuration')

# Getting screen size
output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
resolution = output.split()[0].split(b'x')
width = int(resolution[0])
height = int(resolution[1])
aspect_ratio = width/height

plot_no_of_bots = []
plot_device_ranges = []
plot_deploy_tag = []
plot_no_of_base_stations = []

title_a = ""
title_b = ""
title_c = ""
title_d = ""
if configuration['deploy_tag'] in ['M','S']:
    plot_deploy_tag = deploy_tag
    if configuration['deploy_tag'] == 'S':
        row_size = 1
        col_size = 2
        subplot_names = ['base_stations_on_{}'.format(i) for i in plot_deploy_tag]
elif configuration['deploy_tag'] == 'F':
    plot_deploy_tag = [deploy_tag[0]]
    title_a = 'Base stations on {}'.format(plot_deploy_tag[0])
else: sys.exit('configuration is wrong')


if configuration['device_ranges'] in ['M','S']:
    plot_device_ranges = device_ranges
    if configuration['device_ranges'] == 'S':
        row_size = 1
        col_size  = 3
        subplot_names = ['{}m_range'.format(i) for i in plot_device_ranges]
elif configuration['device_ranges'] =='F':
    plot_device_ranges = [device_ranges[2]]
    title_b = '{}m Communication range'.format(plot_device_ranges[0])

else: sys.exit('configuration is wrong')


if configuration['no_of_base_stations'] in ['M','S']:  
    for deploy_tag in plot_deploy_tag:
        for device_range in plot_device_ranges:
            path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,deploy_tag,device_range)
            for filename in os.listdir(path):
                plot_no_of_base_stations.append(int(filename.split('_')[0]))
    if configuration['no_of_base_stations'] == 'M':
        legend_names = ['{}_base_stations'.format(i) for i in plot_no_of_base_stations]

    if configuration['no_of_base_stations'] == 'S':
        row_size = 1
        col_size = len(plot_no_of_base_stations)
        subplot_names = ['{}_base_stations'.format(i) for i in plot_no_of_base_stations] 
    

if configuration['no_of_base_stations'] == 'F':  
    for deploy_tag in plot_deploy_tag:
        for device_range in plot_device_ranges:
            path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,deploy_tag,device_range)
            n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
            plot_no_of_base_stations.append(min(n))
            title_c = 'Minium base stations of a range'
        
             


if configuration['no_of_bots'] in ['M','S']:
    plot_no_of_bots = no_of_bots
    if configuration['no_of_bots'] == 'S':
        row_size = 2
        col_size  = 3
        subplot_names = ['{}_agents'.format(i) for i in plot_no_of_bots]
elif configuration['no_of_bots'] =='F':
    plot_no_of_bots = [no_of_bots[2]]
    title_d = '{} Agents'.format(plot_no_of_bots[0])

else: sys.exit('configuration is wrong') 





# fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=subplot_names)
# legend_id = 0
# for i,tag in enumerate(plot_deploy_tag):
#     for j,device_range in enumerate(plot_device_ranges):
#         for k,base_station in enumerate(plot_no_of_base_stations):
#             for l,bot in enumerate(plot_no_of_bots):
#                     path = '{}/post_process/{}/on_{}/{}m_range/{}_base_stations/{}_agents/run_0'.format(dirname,graph_name,tag,device_range,base_station,bot)
#                     if os.path.exists(path):
#                         print(base_station)
#                         plot_number= i*(configuration['deploy_tag']=='S')+j*(configuration['device_ranges']=='S')+k*(configuration['no_of_base_stations']=='S')+l*(configuration['no_of_bots']=='S') 
#                         row_id = int(plot_number/col_size) + 1
#                         col_id = plot_number%col_size + 1
#                         idle = np.load('{}/data_final.npz'.format(path))['arr_0']
#                         stamps = np.load('{}/stamps_final.npz'.format(path))['arr_0']
#                         val = np.average(idle,axis=1).cumsum()
#                         val = val/np.arange(1,val.shape[0]+1)
#                         # print(legend_id,len(plot_no_of_base_stations),k,l)
#                         fig.add_trace(go.Scatter(x=stamps, y=val,mode='lines',showlegend=True),row=row_id,col=col_id)
#                         legend_id +=1

# print(plot_no_of_base_stations)
# # fig.update_yaxes(range =[0,8000])
# # fig.update_layout(title_text="{} {} {} {}".format(title_a,title_b,title_c,title_d))
# # fig.show()

