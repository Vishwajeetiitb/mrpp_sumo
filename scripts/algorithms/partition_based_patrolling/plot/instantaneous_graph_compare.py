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

dirname = rospkg.RosPack().get_path('mrpp_sumo')
no_agents_list = [3,6,8,10,12,15]
algo_list = ['iot_communication_network_150','iot_communication_network_250','iot_communication_network_350','iot_communication_network_500','iot_communication_network_10000']
available_comparisons = ['Idleness', 'Worst Idleness']
comparison_parameter_index = 0
scater_nodes_algo_index =  2# putting scatter for only one algo is better otherwise mess put -1 if don't require node scatter
row_size = 2
col_size = 3
graph_name = 'iit_madras'
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

fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=[str(i)+ " Agents" for i in no_agents_list])


for idx,no_agents in enumerate(no_agents_list):
    
    for m,algo_name in enumerate(algo_list):
        df = pd.DataFrame()
        idle = np.load(dirname+ "/post_process/"  + graph_name+ "/run0/"+ algo_name + "/" + str(no_agents)+ "_agents/data_final.npz")['arr_0']
        stamps = np.load(dirname+ "/post_process/" + graph_name+ "/run0/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npz")['arr_0']
        
        if comparison_parameter_index == 0 : 
            # val = np.average(idle,axis=1)
            val = np.average(idle,axis=1).cumsum()
            val = val/np.arange(1,val.shape[0]+1)
        elif comparison_parameter_index == 1 : val = np.max(idle,axis=1)
        fig.add_trace(go.Scatter(x=stamps, y=val,mode='lines',marker=dict(color=color_list[m]),legendgroup=m+1,name=algo_name,showlegend=(True if idx==0 else False)),row=int(idx/col_size)+1,col=idx%col_size+1)
        if scater_nodes_algo_index !=-1 and scater_nodes_algo_index ==m:
            # print(np.repeat(stamps,idle.shape[1]).shape,idle.flatten().shape)
            # p = pd.DataFrame()
            # p['node_idles'] = np.repeat(stamps,idle.shape[1])
            # p['stamp'] = 
            discrete = np.arange(0,stamps.shape[0],int(stamps.shape[0]/100))
            idle = np.take(idle,discrete,axis=0)
            stamps = np.take(stamps,discrete,axis=0)
            fig.add_trace(go.Scattergl(x=np.repeat(stamps,idle.shape[1]) , y=idle.flatten(),
                            mode='markers',
                                                marker=dict(
                            showscale=False,
                            # colorscale options
                            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                            colorscale='Jet',
                            reversescale=False,
                            size=5,
                            opacity = 0.2,
                            color=idle.flatten(),
                            colorbar=dict(
                                thickness=15,
                                title="Instantaneous node idleness values",
                                xanchor='left',
                                titleside='right'
                            ),
                            line_width=0.5),legendgroup=100,showlegend=(True if idx==0 else False),name=algo_name),row=int(idx/col_size)+1,col=idx%col_size+1)  
                            
        

    fig['layout']['xaxis'+str(idx+1)]['title']='Stamps'
    fig['layout']['yaxis'+str(idx+1)]['title']='Instantaneous Graph ' + available_comparisons[comparison_parameter_index]

fig.update_layout(title='Instantaneous Graph '+ available_comparisons[comparison_parameter_index]+ ' Plot for ' + graph_name,title_x=0.5)


file_name = ""
for idx,algo in enumerate(algo_list):
    if not idx:
        file_name = algo
    else:
        file_name = file_name + " | " + algo
        
file_name = file_name + ".html"

if comparison_parameter_index == 0 :
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/instantaneous_graph_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig.write_html(plot_dir+file_name)

    print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/instantaneous_graph_idle/' + urllib.parse.quote(file_name))
    iplot(fig)

if comparison_parameter_index == 1 :
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/instantaneous_graph_worst_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig.write_html(plot_dir+file_name)

    print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/instantaneous_graph_worst_idle/' + urllib.parse.quote(file_name))
    iplot(fig)
