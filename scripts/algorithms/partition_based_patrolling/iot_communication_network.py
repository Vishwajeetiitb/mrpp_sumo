#!/usr/bin/env python3

'''
RANDOM

ROS Params
'''

from dis import dis
from tkinter.tix import Tree
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
import sys

rospack = rospkg.RosPack()


class MRPP_IOT:
    def __init__(self, graph):
        self.ready = False
        self.graph = graph
        self.stamp = 0.
        self.nodes = list(self.graph.nodes())
        
        self.num_bots = int(rospy.get_param('/init_bots'))
        self.dirname = rospkg.RosPack().get_path('mrpp_sumo')
        self.name = rospy.get_param("/random_string")
        self.deploy_tag = rospy.get_param("/deploy_tag")
        self.no_of_base_stations = rospy.get_param("/no_of_base_stations")
        self.run_id = rospy.get_param('/run_id')
        self.algo_name = 'iot_communication_network'
        self.device_range =rospy.get_param('/iot_device_range')
        self.graph_results_path = '{}/scripts/algorithms/partition_based_patrolling/deployment_results'.format(dirname)
        self.base_stations_df = pd.read_csv('{}/{}/on_{}/{}m_range/{}_base_stations.csv'.format(self.graph_results_path,graph_name,self.deploy_tag,self.device_range,self.no_of_base_stations), converters={'location': pd.eval, 'Radius': pd.eval})

        self.total_nodes = len(self.nodes)
        # A virtual array for storing data of all IoT devices (Base stations)
        self.base_stations_arr = np.zeros(
            [self.no_of_base_stations, self.total_nodes])
        # A virtual array of agent for storing data of all IoT devices (Base stations)
        self.agents_arr = np.zeros([self.num_bots, self.total_nodes])

        # Variables for storing data in sheets
        self.data_arr = np.zeros([1, self.total_nodes])
        self.global_idle = np.zeros(self.total_nodes)
        self.stamps = np.zeros(1)
        self.agents_masterdata = np.zeros([1, self.num_bots, self.total_nodes])


        self.sim_dir = '{}/post_process/{}/on_{}/{}m_range/{}_base_stations/{}_agents/run_{}'.format(dirname,graph_name,self.deploy_tag,self.device_range,self.no_of_base_stations,self.num_bots,self.run_id)


        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_idle(self, data):
        if self.stamp < data.stamp and not done:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp

            self.base_stations_arr = np.add(self.base_stations_arr, dev)
            self.agents_arr = np.add(self.agents_arr, dev)
            self.global_idle += dev

            # Create set containing Agent id and Base stations containing that Agent(Agent Location)
            bot_base_stations = []
            for n, bot in zip(data.node_id, data.robot_id):
                node_index = self.nodes.index(n)
                self.global_idle[node_index] = 0
                which_base_stations = self.base_stations_df.index[self.base_stations_df['covered_nodes'].str.contains(
                    '\'' + n + "\'")].tolist()
                which_base_stations.append(bot)
                bot_base_stations.append(set(which_base_stations))

            # Connected component algorithm : To group network of connected base stations and Agents
            pool = set(map(frozenset, bot_base_stations))
            groups = []
            while pool:
                groups.append(set(pool.pop()))
                while True:
                    for candidate in pool:
                        if groups[-1] & candidate:
                            groups[-1] |= candidate
                            pool.remove(candidate)
                            break
                    else:
                        break

            # Comparison between elements of group starts here
            for group in groups:
                comparison_array = []
                for element in group:
                    if 'bot' in str(element):
                        node = data.node_id[data.robot_id.index(element)]
                        node_index = self.nodes.index(node)
                        bot_id = int(element.split('_')[-1])
                        self.agents_arr[bot_id][node_index] = 0
                        comparison_array.append(self.agents_arr[bot_id, :])
                    else:
                        comparison_array.append(
                            self.base_stations_arr[element, :])

                # Comparision of Data of all agents and base stations in same group (network)
                common_data = np.min(comparison_array, axis=0)

                for element in group:
                    if 'bot' in str(element):
                        bot_id = int(element.split('_')[-1])
                        self.agents_arr[bot_id, :] = common_data
                    else:
                        self.base_stations_arr[element, :] = common_data

            # Monitoring and saving data for graph
            self.stamps = np.append(self.stamps, self.stamp)
            self.data_arr = np.append(
                self.data_arr, [self.global_idle], axis=0)
            self.agents_masterdata = np.append(
                self.agents_masterdata, [self.agents_arr], axis=0)

    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot_id = int(req.name.split('_')[-1])
        
        neigh = list(self.graph.successors(node))
        idles = []
        for n in neigh:
            idx = self.nodes.index(n)
            idles.append(self.agents_arr[bot_id, idx])

        max_id = 0
        if len(neigh) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        next_walk = [node, neigh[max_id]]
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'iot_communication_network' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

    def save_data(self):
        
        if os.path.exists(self.sim_dir):
            shutil.rmtree(self.sim_dir)
            os.makedirs(self.sim_dir)
        else:
            os.makedirs(self.sim_dir)
        print("Saving data")
        np.save(self.sim_dir+"/data.npy", self.data_arr)
        np.save(self.sim_dir+"/stamps.npy", self.stamps)
        np.save(self.sim_dir+"/agents_masterdata.npy", self.agents_masterdata)
        np.save(self.sim_dir+"/nodes.npy", np.array(self.nodes))


    # def is_packet_loss(self,base_station_id,node_id):
    #     data = self.graph.nodes[node_id]
    #     station_location = self.base_stations_df.iloc[base_station_id]['location']
    #     dist = ((data['x']-station_location[0])**2+(data['y']-station_location[1])**2)**0.5
    #     signal_prob = (self.base_staion_good_range**2)/(dist**2)
    #     if signal_prob > 1:  signal_prob = 1
    #     is_loss = np.random.choice([True, False], size=(1), p=[1-signal_prob, signal_prob])[0]
    #     return is_loss

    # def data_transfer(reciever,sender):


if __name__ == '__main__':
    rospy.init_node('random', anonymous=True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    done = False
    s = MRPP_IOT(g)
    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    while not done:
        done = rospy.get_param('/done')

    sleep(1.0)
    s.save_data()
