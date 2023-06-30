from turtle import color
import plotly.express as px
import plotly.graph_objects as go
import rospkg
import numpy as np
import pandas as pd
import os 
import urllib.parse


graph_name = 'pipeline1'
no_agents_list = [1,3,6,9,12,15]
algo_list = ['iot_communication_network_150','iot_communication_network_250','iot_communication_network_350','iot_communication_network_500','iot_communication_network_10000']
deploy_tag = 'graph'
device_ranges = [100,240,1000]
device_names = ['Zigbee','BLE','LoRa']
# algo_list = ['run1/'+ i for i in algo_list]
steady_time_stamp = 3000
runs = 1
dirname = rospkg.RosPack().get_path('mrpp_sumo')
available_comparisons = ['avg_idleness', 'worst_idleness']


stamp_as_points = True
comparison_parameter_index = 0

# fig = go.Figure(layout=go.Layout(
#                 title=graph_name +' Average node Idleness distribution',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 # annotations=[ dict(
#                 #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                 #     showarrow=False,
#                 #     xref="paper", yref="paper",
#                 #     x=0.005, y=-0.002 ) ],
#                 yaxis=dict(scaleanchor="x", scaleratio=1)))

df = pd.DataFrame()
for no_agents in no_agents_list:
    for device_range in device_ranges:
        worst_idles = np.zeros
        avg_idles = np.zeros(0)
        for run_id in range(runs):
            path = '{}/post_process/{}/on_{}/{}m_range'.format(dirname,graph_name,deploy_tag,device_range)
            n = [int(filename.split('_')[0]) for filename in os.listdir(path)]
            df = pd.DataFrame()
            results_path = '{}/{}_base_stations/{}_agents/run_{}'.format(path,min(n),no_agents,run_id)
            idle = np.load('{}/data_final.npz'.format(results_path))['arr_0']
            stamps = np.load('{}/stamps_final.npz'.format(results_path))['arr_0']
            idle = idle[np.argwhere(stamps>steady_time_stamp)[0][0]:]  # Taking idlness values after steady state
            if run_id !=0:
                size = min(worst_idles.shape[0],idle.max(axis=int(stamp_as_points)).shape[0])
                worst_idles = worst_idles[0:size]+idle.max(axis=int(stamp_as_points))[0:size]
                avg_idles = avg_idles[0:size] + np.average(idle,axis=int(stamp_as_points))[0:size]
            else:
                worst_idles = idle.max(axis=int(stamp_as_points))
                avg_idles = np.average(idle,axis=int(stamp_as_points))

        worst_idles = worst_idles/runs
        avg_idles = avg_idles/runs
        df_temp = pd.DataFrame()

        df_temp['Worst Idleness'] = worst_idles
        df_temp['Average Idleness'] = avg_idles
        df_temp['Algorithm']= [device_range]*worst_idles.shape[0]
        df_temp['Agents'] = [no_agents]*worst_idles.shape[0]
        df = pd.concat([df,df_temp])


# fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default



file_name = ""
for idx,algo in enumerate(algo_list):
    if not idx:
        file_name = algo
    else:
        file_name = file_name + " | " + algo
        
file_name = file_name + ".html"


if comparison_parameter_index ==0:
    fig = px.box(df, x="Agents", y="Average Idleness", color="Algorithm")
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/avg_node_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(plot_dir+file_name)
    print("http://vishwajeetiitb.github.io/mrpp_iot/scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/avg_node_idle/' + urllib.parse.quote(file_name))

elif comparison_parameter_index ==1:
    fig = px.box(df, x="Agents", y="Worst Idleness", color="Algorithm")
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/wrost_node_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(plot_dir+file_name)
    print("http://vishwajeetiitb.github.io/mrpp_iot/scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/wrost_node_idle/' + urllib.parse.quote(file_name))


fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.12,
    xanchor="auto",
    x=0.5
))
# fig.update(layout_showlegend=False)
fig.show()