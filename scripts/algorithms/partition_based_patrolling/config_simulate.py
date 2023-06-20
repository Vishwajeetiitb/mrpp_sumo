#! /usr/bin/env python3

import configparser as CP
import os
import rospkg
import glob
import rospy
import os
import sys

import networkx as nx
from datetime import datetime
import subprocess
from art import *

start_time  = datetime.now()
no_of_bots = [1,3,6,9,12,15]
# no_of_bots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
algos = ['iot_communication_network']
graphs = ['pipeline1']
iot_device_ranges = [100,300,500]
dir_name = rospkg.RosPack().get_path('mrpp_sumo')
no_of_runs = 1
deploy_tag = 'edge'
# rospy.init_node('config_simulate')
for graph_name in graphs:
    graph_path = dir_name +'/graph_ml/'+ graph_name + '.graphml'
    graph_net = nx.read_graphml(graph_path)
    for run_id in range(no_of_runs):
        for algo_name in algos:
            how_many_iterations = 1
            if 'iot' in algo_name:
                how_many_iterations = len(iot_device_ranges)
            for idx in range(how_many_iterations):
                device_range = iot_device_ranges[idx]
                path = '{}/scripts/algorithms/partition_based_patrolling/deployment_results/{}/on_{}/{}m_range/'.format(dir_name,graph_name,deploy_tag,device_range)
                for filename in os.listdir(path):
                    no_of_base_stations = int(filename.split('_')[0])
                    for no_agents in no_of_bots:
                        sim_dir = '{}/post_process/{}/on_{}/{}m_range/{}_base_stations/{}_agents/run_{}/'.format(dir_name,graph_name,deploy_tag,device_range,no_of_base_stations,no_agents,run_id)
                        if not os.path.exists(sim_dir):
                            run_start_time = datetime.now()
                            init_locations = " ".join(list(graph_net.nodes())[0:no_agents])
                            os.system("xterm -e roscore & sleep 3")
                            rospy.set_param('/init_locations',init_locations)
                            rospy.set_param('/use_sim_time',True)
                            rospy.set_param('/gui',False)
                            rospy.set_param('/graph',graph_name)
                            rospy.set_param('/init_bots',no_agents)
                            rospy.set_param('done',False)
                            rospy.set_param('/sim_length',30000)
                            rospy.set_param('/algo_name',algo_name)
                            rospy.set_param('/no_of_deads',0)
                            rospy.set_param('/run_id',run_id)
                            rospy.set_param('/random_string','test')
                            rospy.set_param('/deploy_tag',deploy_tag)
                            rospy.set_param('/no_of_base_stations',no_of_base_stations)

                            if 'iot' in algo_name: rospy.set_param('/iot_device_range',iot_device_ranges[idx])
                            for name in rospy.get_param_names()[4:]:
                                print(name,':',rospy.get_param(name))

                            # subprocess.call(['terminator', '--layout', 'grid', '-x', cmd1, '-x', cmd2, '-x', cmd3])
                            os.system("xterm -e rosrun mrpp_sumo sumo_wrapper.py & sleep 3")
                            os.system("xterm -e rosrun mrpp_sumo "+ algo_name +".py & sleep 3")
                            os.system("xterm -e rosrun mrpp_sumo command_center.py")
                            os.system("sleep 10")
                            # os.system("killall xterm & sleep 3")

                            run_end_time = datetime.now()
                            print('Algorithm took', run_end_time-run_start_time)
                            print('─' * 100,'\n')


end_time  = datetime.now()
print('\n','StartTime:',start_time, '|EndTime:',end_time,'|Total Time taken:',end_time-start_time)
tprint("Please run  post_data_process.py script!!")