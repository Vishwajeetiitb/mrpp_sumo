import rospkg
from ast import literal_eval
import pandas as pd
import plotly.express as px

graph_name = 'iit_bombay'
ranges = [150,250,350,500]
dirname = rospkg.RosPack().get_path('mrpp_sumo')
# no_of_base_stations = np.load(dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/required_no_of_base_stations.npy')[0]
graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'

devices = []
for range in ranges:
    base_stations_df = pd.read_csv(graph_results_path + graph_name + '/' + str(range) + '_range_base_stations.csv',converters={'location': pd.eval,'Radius': pd.eval})
    devices.append(base_stations_df.shape[0])

print(ranges,devices)

fig = px.line(x=ranges, y=devices, title='Range vs Number of devices deployed for '+graph_name,text=devices)
fig.update_traces(textposition='top center')
fig.update_layout(title_x=0.5)
fig.show()