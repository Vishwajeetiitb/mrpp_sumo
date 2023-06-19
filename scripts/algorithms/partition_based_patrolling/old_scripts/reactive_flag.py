#!/usr/bin/env python3

'''
Reactive with Flags
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode
import random as rn
import os
import shutil

class RR:

    def __init__(self, g):
        self.ready = False
        self.graph = g
        self.stamp = 0.
        self.num_bots = int(rospy.get_param('/init_bots'))
        self.nodes = list(self.graph.nodes())

        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

        self.algo_name = 'reactive_flag'
        self.sim_dir = dirname + '/post_process/'+ graph_name + '/'+ self.algo_name +'/' + str(self.num_bots) + '_agents'
        
        if os.path.exists(self.sim_dir):
            shutil.rmtree(self.sim_dir)
            os.makedirs(self.sim_dir)
        else:
            os.makedirs(self.sim_dir)

        
        # Variables for storing data in sheets
        self.data_arr = np.zeros([1,len(self.nodes)])
        self.global_idle = np.zeros(len(self.nodes))
        self.stamps = np.zeros(1) 

    def callback_idle(self, data):
        if self.stamp < data.stamp and not done:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                
            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.

            self.global_idle +=dev
            for n in data.node_id:
                node_index = self.nodes.index(n)
                self.global_idle[node_index] = 0

            self.stamps = np.append(self.stamps,self.stamp)
            self.data_arr = np.append(self.data_arr,[self.global_idle],axis=0)
                
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp

        self.graph.nodes[node]['idleness'] = 0.
        node_index = self.nodes.index(node)
        self.global_idle[node_index] = 0

        neigh = list(self.graph.successors(node))
        idles = []
        for n in neigh:
            idles.append(self.graph.nodes[n]['idleness'])

        max_id = 0
        if len(neigh) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        next_walk = [node, neigh[max_id]]
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'reactive_flag' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

    def save_data(self):
        print("Saving data")
        np.save(self.sim_dir+"/data.npy",self.data_arr)
        np.save(self.sim_dir+"/stamps.npy",self.stamps)
        np.save(self.sim_dir+"/nodes.npy",np.array(self.nodes))
        print("Data saved!")

if __name__ == '__main__':
    rospy.init_node('rr', anonymous= True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    s = RR(g)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')

    s.save_data()