import argparse  

import sys 
import os  

# from .. import  
import glob
import argparse  

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)  

from matplotlib.pyplot import title      
from numpy.lib import NumpyVersion     
import numpy as np  
import argparse    
import copy as cp     
from sklearn.gaussian_process import GaussianProcessRegressor   
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct  
from sklearn.metrics import mean_squared_error  

from utils_functions import *    
from path_planning.cps.reps import GPREPS     
from utils import *   
from cps_policy import *  

np.set_printoptions(precision=4)     



if __name__ == '__main__':     
    parser = argparse.ArgumentParser()             

    # //////// mode /////////////  
    parser.add_argument('--ctl_mode', type=str, default="motion", help='choose mode first !!!!')      
    parser.add_argument('--bilateral', type=int, default=0, help='choose mode first !!!!')   
    
    parser.add_argument('--training_num', type=int, default=80, help='choose index first !!!!')     
    parser.add_argument('--iter_index', type=int, default=0, help='give numn epi first !!!!')            
    
    args = parser.parse_args()   


    cps_para = {
        'context_dim': 2, 
        'para_dim': 2, 
        'policy_type': 'reps', 
        'epi' : 0.1, 
        'beta' : 1.0, 
        'num_sim' : 2000,   
        'num_real' : 10,   
        'num_space' : 200,  
        'batch_num' : 7,  
        'policy_path': '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/scripts/cps_learning/policy_para',  
        'impedance_path': '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/scripts/cps_learning/impedance_para',   
        'data_pair_path' : '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/scripts/cps_learning/data_pair',    
        'subject_num': 0,  
        'exp_num': 0,  
        'iter_index': args.iter_index,  
        'context_range': {'speed': [0.8, 1.0, 1.1, 1.2, 1.3, 1.4], 'slope' : [-8, -4, -2, 0, 2, 4, 8]}   
    }   

    cps_policy = CPS(cps_para=cps_para)   

    # data_pair_path = cps_para['data_pair_path'] + '/subject_0/exp_num_0/eva_0/state_reward_pair.npz' 
    # np_file = np.load(data_pair_path)    
    
    # context_buf = np_file['context']
    # para_buf = np_file['para']   
    # reward_buf = np_file['reward']    
    
    # print(context_buf)  
    
    # policy_path = '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/data/data420'   
    # for index in range(1, 10):   
    #     path = policy_path + '/e' + str(index) + '/pair'  
    #     state_list = glob.glob(path + '/state_*.npy') 
    #     reward_list = glob.glob(path + '/reward_*.npy')    
    #     data_state = []  
    #     data_reward = []    
    #     for state_path, reward_path in zip(state_list, reward_list):  
    #         state_list = np.load(state_path)  
    #         reward_list = np.load(reward_path)  
    #         data_state.append(state_list) 
    #         data_reward.append(reward_list)   

    #         cps_policy.push_pair(context_list=state_list[:2], para_list=state_list[2:], reward_list=reward_list)    
    
    # cps_policy.update_policy(training_num=args.training_num)       

    # data_state = np.array(data_state)  
    # data_reward = np.array(data_reward)  
    # # print(data_state.shape, data_reward.shape) 
    # cps_policy.push_data(context_list=data_state[:, :2], para_list=data_state[:, 2:4], reward_list=data_reward)   

    # state_action_pair = np.load(policy_path + '/e' + str(index) + '/pair/' + 'state_' + str(state_index) + '.npy')     
    # reward_pair = np.load(policy_path + '/e' + str(index) + '/pair/' + 'reward_' + str(state_index) + '.npy')   

    # print("size of dataset :", cps_policy.cps_dataset._size)       
    # data_pair_path = cps_para['data_pair_path'] + '/eva_' + str(0)
    # cps_policy.cps_dataset.save(data_pair_path)    

    # score = cps_policy.update_gp(training_num=80)  
    # print("score :", score)     

    # cps_policy.update_policy(training_num=80)  

    # print(range(10))   

    # for index in range(5): 
    #     print("Please Enter to continue ...")
    #     input()  
    #     print('Output impedance parameters !!!')  