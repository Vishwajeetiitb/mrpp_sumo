# This script fills the idleness data


#! /usr/bin/env python3
import numpy as np
import rospkg
import os
dir_name = rospkg.RosPack().get_path('mrpp_sumo')


all_dir = [x[0] for x in os.walk(dir_name + '/post_process/')]
for dir in all_dir:
    if os.path.exists(dir + '/agents_masterdata.npy'):
        data_arr = np.load(dir + '/data.npy').astype('int32')
        stamps = np.load(dir + '/stamps.npy').astype('int32')
        agent_masterdata = np.load(dir + '/agents_masterdata.npy').astype('int32')

        total_nodes = data_arr.shape[1]
        tmp_data = []
        tmp_stamps = []
        tmp_agent_masterdata = []
        for idx, data, stamp, agents_data in zip(range(stamps.shape[0]), data_arr, stamps, agent_masterdata):
            if idx < stamps.shape[0]-1:
                for i in range(stamps[idx+1]-stamps[idx]-1):

                    tmp_stamps.append(stamps[idx]+i+1)
                    tmp_data.append(data_arr[idx]+i+1)
                    tmp_agent_masterdata.append(agent_masterdata[idx]+i+1)

        stamps = np.append(stamps, tmp_stamps)
        data_arr = np.concatenate((data_arr, np.array(tmp_data)), axis=0)
        agent_masterdata = np.concatenate(
            (agent_masterdata, np.array(tmp_agent_masterdata)), axis=0)
        sort_indices = stamps.argsort()
        stamps_final = np.take(stamps, sort_indices)
        data_arr_final = np.take(data_arr, sort_indices, axis=0)
        agent_masterdata_final = np.take(
            agent_masterdata, sort_indices, axis=0)

        print(dir, data_arr_final.shape, stamps_final.shape,agent_masterdata_final.shape)
              
        np.savez_compressed(dir+"/data_final", data_arr_final)
        np.savez_compressed(dir+"/stamps_final", stamps_final)
        np.savez_compressed(dir+"/agent_masterdata_final", agent_masterdata_final)
        
        if os.path.exists(dir + '/agent_masterdata_final.npz'):
            os.system('rm ' + dir + '/*.npy')
