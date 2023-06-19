#!/usr/bin/env python3

'''
RANDOM

ROS Params
'''

from dis import dis
from turtle import distance
import rospy
import rospkg
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse
from mrpp_sumo.srv import AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode 

import random as rn
import pandas as pd
from ast import literal_eval
import numpy as np
import os
import shutil
from time import sleep


rospack = rospkg.RosPack()


class MRPP_IOT:
    def __init__(self, graph):
        self.ready = False
        self.graph = graph
        self.stamp = 0.
        self.nodes = list(self.graph.nodes())
        self.Iot_device_range = 250
        self.graph_results_path = dirname + '/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'
        self.base_stations_df = pd.read_csv(self.graph_results_path + graph_name + '/' + str(self.Iot_device_range) + '_range_base_stations.csv',converters={'location': pd.eval,'Radius': pd.eval})
        
        self.num_bots = int(rospy.get_param('/init_bots'))
        self.dirname = rospkg.RosPack().get_path('mrpp_sumo')
        self.name = rospy.get_param("/random_string")
        self.algo_name = 'mrpp_iot2_packet_loss'
        self.sim_dir = dirname + '/post_process/'+ graph_name + '/'+ self.algo_name + '_'+str(self.Iot_device_range)+'/' + str(self.num_bots) + '_agents'
        if os.path.exists(self.sim_dir):
            shutil.rmtree(self.sim_dir)
            os.makedirs(self.sim_dir)
        else:
            os.makedirs(self.sim_dir)


        self.base_stations_arr = [] # A virtual array for storing data of all IoT devices (Base stations)


        # Variables for storing data in sheets
        self.data_arr = np.zeros([1,len(self.nodes)])
        self.global_idle = np.zeros(len(self.nodes))
        self.stamps = np.zeros(1) 


        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True
    
    def callback_idle(self, data):
        if self.stamp < data.stamp and not done:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp

            self.base_stations_arr = np.add(self.base_stations_arr,dev,dtype=object)

            self.global_idle +=dev

            for n in data.node_id:
                node_index = self.nodes.index(n)
                self.global_idle[node_index] = 0
                which_base_stations = self.base_stations_df.index[self.base_stations_df['covered_nodes'].str.contains('\''+ n + "\'") ].tolist()

                for base_station_id in which_base_stations:
                    nodes_set = literal_eval(self.base_stations_df.iloc[base_station_id]['covered_nodes'])
                    base_station_node_index = nodes_set.index(n)
                    self.base_stations_arr[base_station_id][base_station_node_index] = 0

            
            self.stamps = np.append(self.stamps,self.stamp)
            self.data_arr = np.append(self.data_arr,[self.global_idle],axis=0)
            
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        neigh = list(self.graph.successors(node))

        node_index = self.nodes.index(node)
        self.global_idle[node_index] = 0
        
        which_base_stations = self.base_stations_df.index[self.base_stations_df['covered_nodes'].str.contains('\''+ node + "\'") ].tolist()   
        for base_station_id in which_base_stations:
            nodes_set = literal_eval(self.base_stations_df.iloc[base_station_id]['covered_nodes'])
            base_station_node_index = nodes_set.index(node)
            self.base_stations_arr[base_station_id][base_station_node_index] = 0 
            
        option_nodes = []
        for n in neigh:
            for base_station_id in which_base_stations:
                if not self.is_packet_loss(base_station_id,node):
                    nodes_set = literal_eval(self.base_stations_df.iloc[base_station_id]['covered_nodes'])
                    vals = np.zeros(0)
                    nodes = np.zeros(0)
                    if n in nodes_set: 
                        base_station_node_index = nodes_set.index(n)
                        vals = np.append(vals,self.base_stations_arr[base_station_id][base_station_node_index])
                        nodes = np.append(nodes,n)
                        
                    if vals.size:
                        option_nodes.extend(nodes[np.where(vals==np.max(vals))[0]])

        if len(option_nodes):
            next_walk = [node, rn.sample(option_nodes,1)[0]]
        
        else:
            next_walk = [node, rn.sample(list(self.graph.successors(node)),1)[0]]

        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'mrpp_iot2_packet_loss' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)
    
    def save_data(self):
        print("Saving data")
        np.save(self.sim_dir+"/data.npy",self.data_arr)
        np.save(self.sim_dir+"/stamps.npy",self.stamps)
        np.save(self.sim_dir+"/nodes.npy",np.array(self.nodes))
        print("Data saved!")


    def initialize_base_stations(self):
        self.base_staion_good_range = self.Iot_device_range/2  # Range in meter where there is no packet loss
        for number_of_nodes in self.base_stations_df['Total_nodes_covered']:
            self.base_stations_arr.append(np.zeros(number_of_nodes))
        print("Base stations Initialized!")

        
    def is_packet_loss(self,base_station_id,node_id):
        data = self.graph.nodes[node_id]
        station_location = self.base_stations_df.iloc[base_station_id]['location']
        dist = ((data['x']-station_location[0])**2+(data['y']-station_location[1])**2)**0.5
        signal_prob = (self.base_staion_good_range**2)/(dist**2)
        if signal_prob > 1:  signal_prob = 1
        is_loss = np.random.choice([True, False], size=(1), p=[1-signal_prob, signal_prob])[0]
        return is_loss
        


    
        

if __name__ == '__main__':
    rospy.init_node('random', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    done = False
    s = MRPP_IOT(g)
    s.initialize_base_stations()
    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)
    while not done:
        done = rospy.get_param('/done')

    sleep(1.0)
    s.save_data()