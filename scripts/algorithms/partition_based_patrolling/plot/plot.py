
import plotly.express as px
import numpy as np
import sys
import rospkg

dirname = rospkg.RosPack().get_path('mrpp_sumo')
no_agents = 3
# which_node = sys.argv[2]
which_node = 'a'
graph_name = 'iitb_full'
algo_name = 'iot_communication_network_500'
df = px.data.gapminder().query("country=='Canada'")


idle = np.load(dirname+ "/post_process/"+ graph_name+ "/"+ algo_name + "/"+ str(no_agents)+ "_agents/data_final.npy")
graph_idle = np.average(idle,axis=1)
stamps = np.load(dirname+ "/post_process/"+ graph_name+ "/"+ algo_name + "/" + str(no_agents)+ "_agents/stamps_final.npy")
print()

if which_node == 'a':
	fig = px.line(x=stamps, y=graph_idle)

else: 
	fig = px.line(x=stamps, y=idle[:,int(which_node)])

fig.write_html("./file.html")
fig.show()