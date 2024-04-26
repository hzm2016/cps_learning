##############################################
####### 08/04/2024 revised by Jimmy Hou ######
##############################################
# context: the task infor   
# para: the latent parameter  
# state: [context, para]  
# reward: scalar  
# original para: 

# offline training 
# online training 

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


# sns.set(font_scale=1.5)     
np.set_printoptions(precision=4)      



##### generate original para ####
def generate_para_from_latent():  


    return para 


##### generate context para #####
def generate_context_para(args=None, context_range=None, para_range=None):             
    # range of angle values  
    # angle_range = np.linspace(-1.0 * np.pi, 1.0 * np.pi, args.nb_data)       
    
    angle_range = np.linspace(-0.7 * np.pi, 0.7 * np.pi, args.nb_data)       
    obs_range = np.random.random((args.nb_data, args.context_dim))     
    para_space = np.random.random((args.nb_data, args.para_dim))    
      
    center_range = np.arange(args.nb_data).reshape(-1, 1)      
    state_range = np.hstack((angle_range.reshape(-1, 1), obs_range[:, 2].reshape(-1, 1)))    
    state_range = np.hstack((state_range, para_space))    
    # angle_range = np.linspace(0.0, 2*np.pi, args.nb_data)      
    # print("center_range :", center_range.shape)    
    # print("state_range :", state_range.shape)    
    
    # index list     
    # index_list = np.random.choice(np.arange(50, args.nb_data-50), size=args.num_initial)    
    index_list = np.random.choice(np.arange(args.nb_data), size=args.num_initial)    
    
    # random x values for sampling    
    # num_samples = 10    
    # angle_list = np.random.choice(angle_range, size=args.num_initial)    
    angle_list = angle_range[index_list]     
    # angle = np.random.random((args.num_epi, 1)) * 2 * np.pi     
    # print(angle_list.reshape(-1, 1).shape)    
    # center_list = center_range[index_list]    
    # center_list = (angle_list/(2*np.pi) * args.nb_data).astype(int)   
    center_list = (((angle_list/(2*np.pi) * args.nb_data).reshape(-1, 1)).astype(int) + args.nb_data//2)%args.nb_data          
    # print("cos angle :", np.cos(angle), "sin angle :", np.sin(angle))         
    # context_info = np.random.random((args.num_initial, args.context_dim))    
    context_info = obs_range[index_list, :]   
    # print("context info :", context_info.shape)    
    
    ############## input random state ############  
    state_info = np.hstack((angle_list.reshape(-1, 1), context_info[:, 2].reshape(-1, 1)))        
    
    context_info[:, 0] = (np.cos(angle_list) * args.obs_radius).flatten()          
    context_info[:, 1] = (np.sin(angle_list) * args.obs_radius).flatten()            
    context_info[:, 2] = context_info[:, 2] * (context_range[0, 2] - context_range[1, 2]) + context_range[1, 2]     
      
    para_info = para_space[index_list, :]    
    
    print("index_list : \n", index_list)  
    print("context_info: \n", context_info)            
    print("para_info :\n", para_info)    
    print("center_list : \n", center_list)     
    print("angle_list : \n", angle_list)         
    print("state_info : \n", state_info)    

    state_info = np.hstack((state_info, para_info))    

    return context_info, para_info, context_range, para_range, angle_list, center_list, state_info, state_range, center_range             


def generate_initial_context_para(args=None, given_seed=None, context_range=None, para_range=None, 
                                  angle_range=None, size_range=None
                                ):     
    if given_seed is not None:  
        np.random.seed(given_seed)  
    
    all_angle_index_info = np.array(range(0, args.nb_data, 10))       
    np.random.shuffle(all_angle_index_info)       
    angle_index_info = all_angle_index_info[:args.num_initial]       
    
    all_obs_index_info = np.array(range(0, args.nb_data, 10))       
    np.random.shuffle(all_obs_index_info)       
    obs_index_info = all_obs_index_info[:args.num_initial]        
       
    context_info = np.random.random((args.num_initial, args.context_dim))     
    context_info[:, 0] = angle_index_info/args.nb_data          
    context_info[:, 1] = obs_index_info/args.nb_data    
    
    para_info = np.random.random((args.num_initial, args.para_dim))     
    obs_info = np.random.random((args.num_initial, 3))     
    
    ##### get obs info #####
    # for i in range(args.num_initial):  
    # print("context_info :", context_info)   
    # index_info = np.rint(context_info[:, 0] * args.nb_data).astype(int)     
    # print("index_info :", index_info)     
    angle_info = angle_range[angle_index_info]        
    print("angle_info :", angle_info)        
    center_info = (((angle_info/(2*np.pi) * args.nb_data).reshape(-1, 1)).astype(int) + args.nb_data//2)%args.nb_data   
    print("center_info :", center_info.T)    
    obs_size = size_range[obs_index_info]       
    
    ##### context_info ##### 
    obs_info[:, 0] = (np.cos(angle_info) * args.obs_radius).flatten()        
    obs_info[:, 1] = (np.sin(angle_info) * args.obs_radius).flatten()           
    # obs_info[:, 2] = context_info[:, -1] * (context_range[0, -1] - context_range[1, -1]) + context_range[1, -1]    
    obs_info[:, 2] = obs_size      
    print("obs_info :", obs_info)       
    
    state_info = np.hstack((context_info, para_info)).reshape(-1, args.context_dim + args.para_dim)     
    
    ###### save info #######   
    np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/fixed_context_info_" + args.data_name + ".npz", context_info, para_info, obs_info, state_info, center_info, angle_info) 
    # return context_info, para_info, obs_info, state_info, center_info, angle_info  
    return context_info, obs_info, center_info, angle_info, state_info, para_info      


def sample_context(args=None, num_episodes=None, give_seed=None, 
                   context_range=None, angle_range=None, size_range=None
                ):     
    if give_seed is not None: 
        np.random.seed(give_seed)    
       
    context_info = np.random.random((num_episodes, args.context_dim))       
    obs_info = np.random.random((num_episodes, 3))      
    
    ##### get obs info #####   
    # print("context_info :", context_info)    
    index_info = np.clip(np.rint(context_info[:, 0] * args.nb_data), 0, args.nb_data-1).astype(int)       
    # print("index_info :", index_info)     
    angle_info = angle_range[index_info]         
    size_index_info = np.clip(np.rint(context_info[:, 1] * args.nb_data), 0, args.nb_data-1).astype(int)       
    size_info = size_range[size_index_info]     
    # print("angle_info :", angle_info)        
    center_info = (((angle_info/(2*np.pi) * args.nb_data).reshape(-1, 1)).astype(int) + args.nb_data//2)%args.nb_data   
    # print("center_info :", center_info.T)     
    
    ##### context_info #####   
    obs_info[:, 0] = (np.cos(angle_info) * args.obs_radius).flatten()           
    obs_info[:, 1] = (np.sin(angle_info) * args.obs_radius).flatten()            
    # obs_info[:, 2] = context_info[:, -1] * (context_range[0, -1] - context_range[1, -1]) + context_range[1, -1]    
    obs_info[:, 2] = size_info    
    
    # state_info = np.hstack((context_info, para_info)).reshape(-1, args.context_dim + args.para_dim)   
    # state_info  
    return context_info, obs_info, center_info, angle_info        
      

def get_episode_para(args=None, state_info=None, 
                     angle_range=None, size_range=None
                ):  
    context_info = state_info[0, :args.context_dim][None, :]    
    para_info = state_info[0, args.context_dim:][None, :]    
    obs_info = np.random.random((1, 3))    

    angle_index_info = np.clip(np.rint(context_info[:, 0] * args.nb_data), 0, args.nb_data-1).astype(int)       
    angle_info = angle_range[angle_index_info]   
    size_index_info = np.clip(np.rint(context_info[:, 1] * args.nb_data), 0, args.nb_data-1).astype(int)       
    size_info = size_range[size_index_info]     
 
    center_info = (((angle_info/(2*np.pi) * args.nb_data).reshape(-1, 1)).astype(int) + args.nb_data//2)%args.nb_data  

    obs_info[:, 0] = (np.cos(angle_info) * args.obs_radius).flatten()        
    obs_info[:, 1] = (np.sin(angle_info) * args.obs_radius).flatten()            
    obs_info[:, 2] = size_info   
    return context_info, para_info, obs_info, center_info, angle_info   


def get_para_list(args=None, state_info=None, 
                  angle_range=None, size_range=None  
                ):   
    context_info = np.zeros((state_info.shape[0], args.context_dim))    
    para_info = np.zeros((state_info.shape[0], args.para_dim))    
    obs_info = np.zeros((state_info.shape[0], 3))   
    
    context_info = state_info[:, :args.context_dim]    
    para_info = state_info[:, args.context_dim:]

    angle_index_info = np.clip(np.rint(context_info[:, 0] * args.nb_data), 0, args.nb_data-1).astype(int)       
    angle_info = angle_range[angle_index_info]   
    size_index_info = np.clip(np.rint(context_info[:, 1] * args.nb_data), 0, args.nb_data-1).astype(int)       
    size_info = size_range[size_index_info]     

    center_info = (((angle_info/(2*np.pi) * args.nb_data).reshape(-1, 1)).astype(int) + args.nb_data//2)%args.nb_data  

    obs_info[:, 0] = (np.cos(angle_info) * args.obs_radius).flatten()        
    obs_info[:, 1] = (np.sin(angle_info) * args.obs_radius).flatten()            
    obs_info[:, 2] = size_info   

    angle_info = angle_info.reshape(-1, 1)  
    return context_info, para_info, obs_info, center_info, angle_info   


def get_state_para_range(args=None, given_seed=None, context_info=None): 
    if given_seed is not None:  
        np.random.seed(given_seed)    
    context_len = context_info.shape[0]
    para_space = np.random.random((args.num_sim_bo, args.para_dim))   
    state_space = np.random.random((args.num_sim_bo * context_len, args.context_dim + args.para_dim))    
    for i in range(context_len):  
        state_space[i*args.num_sim_bo:(i+1)*args.num_sim_bo, :2] = np.tile(context_info[i, :], (args.num_sim_bo, 1)) 
        state_space[i*args.num_sim_bo:(i+1)*args.num_sim_bo, 2:] = para_space  
    return state_space  


def get_reward_list(
        args=None,  
        waypoints_list=None,  
        force_waypoints_list=None,  
        obs_info=None,  
        range_list=None,  
        center_list=None  
    ):               
    sum_reward_list = []       
    success_list = []      
    force_list = []      
    for k in range(len(waypoints_list)):    
        range_index = range_list[k, :]  
        sum_reward, success, reward_force = get_reward()  
        # sum_reward, success, reward_force = get_reward(
        #     deform_waypoints=waypoints_list[k, int(range_index[0]/args.resample):int(range_index[1]/args.resample), :], 
        #     force_waypoints=force_waypoints_list[k, int(range_index[0]/args.resample):int(range_index[1]/args.resample), :],  
        #     obs_center=obs_info[k, :2], 
        #     obs_r=obs_info[k, 2], 
        #     center_index=center_list[k], 
        #     start_index=int(range_list[k, 0]/args.resample),  
        #     end_index=int(range_list[k, 1]/args.resample), 
        #     max_force=args.max_F,   
        #     beta_1=0.5,  
        #     beta_2=0.5,
        #     circle_radius=args.obs_radius  
        # )   
        
        sum_reward_list.append(sum_reward)     
        success_list.append(success)   
        force_list.append(reward_force)       
    return sum_reward_list, success_list, force_list  


def get_reward(
        deform_waypoints=None,    
        force_waypoints=None,    
        obs_center=None,   
        obs_r=None,    
        center_index=None,       
        start_index=None,   
        end_index=None,     
        max_force=None,  
        beta_1=0.5, 
        beta_2=0.5, 
        circle_radius=None       
    ):              
    sum_reward = 0.0        
    sum_obs = 0.0    
    sum_force = 0.0        
    success = 0     
    reward_force = -1.0      
    success_list = []     
    if start_index > 0 and start_index < center_index and end_index > center_index:    
        for i in range(deform_waypoints.shape[0]):   
            deform_waypoint = deform_waypoints[i, :]  
            # print("deform_waypoint :", deform_waypoint, obs_center)       
            dist = np.linalg.norm(deform_waypoint - obs_center)        
            # print("dist :", dist)  
            if dist > obs_r:   
                sum_obs += 1    
                success = 1   
            else: 
                sum_obs += 0.0   
                success = 0    
            
            success_list.append(success)          
            norm_force = force_waypoints[i, 0]          
            sum_force += norm_force/max_force          
        
        success = 1 - np.any(np.array(success_list)==0).astype(int)      
        # print("success obs :", beta_1 * sum_obs/deform_waypoints.shape[0])        
        # print("sum force :", beta_2 * sum_force/deform_waypoints.shape[0])     
          
        sum_reward = beta_1 * sum_obs - beta_2 * sum_force              
        sum_reward = sum_reward/deform_waypoints.shape[0]      
        # print("sum_reward :", sum_reward)    
        # print("deform length :", (2 * obs_r/(2 * np.pi * circle_radius) * args.nb_data * 1.1 - deform_waypoints.shape[0])/args.nb_data)  
        sum_reward += (2 * obs_r/(2 * np.pi * circle_radius) * args.nb_data * 1.1 - deform_waypoints.shape[0])/args.nb_data        
        reward_force = - 1 * sum_force/deform_waypoints.shape[0]  
    else:
        sum_reward = -1.0     
        success = 0    
        reward_force = -1.0       

    return sum_reward, success, reward_force     
    

def get_initial_data(args=None, tele_exp=None, ref_data=None, context_range=None, para_range=None, angle_range=None, size_range=None):    
    ################### get initial context data ##################
    result_info = generate_initial_context_para(args=args, given_seed=100, context_range=context_range, para_range=para_range, angle_range=angle_range, size_range=size_range)     
    context_info, obs_info, center_info, angle_info, state_info, para_info = result_info[0], result_info[1], result_info[2], result_info[3], result_info[4], result_info[5]  
    desire_start_list = np.ones(args.num_initial) * 100     

    # ####################### control test //////////////////////// 
    assert context_info.shape[0] == args.num_initial      
    assert para_info.shape[0] == args.num_initial      
    # assert desire_start_list.shape[0] == args.num_initial       
    
    if args.move_target:    
        joint_target = ref_data[0, :3]        
        tele_exp.moveToZero(
            joint_target,   
            args.speed,   
            args.delay_time   
        )   
    
    if args.run_ctl:    
        print("ref_data :", ref_data.shape)       
        K_p_re = np.array([1.0, 1.0, 1.0])      
        K_d_re = 0.01 * K_p_re     
        results = tele_exp.control(
            ref_data,   
            obs_info,  # information of obstacles     
            para_info,   
            desire_start_list,   
            K_p_re,   
            K_d_re,   
            5000,     
            2000,     
            0.4,    
            args.ee_force_amp,     
            args.method_index, 
            0   
        )   
        
        traj_info, range_info = results[0], results[1]       

        all_waypoints = traj_info[::args.resample, 3:5]      
        waypoints_list = all_waypoints.reshape((args.num_initial, args.nb_data, 2))     
        force_waypoints_list = traj_info[::args.resample, 9].reshape((args.num_initial, args.nb_data, 1))                 
        # print("waypoints_list :", waypoints_list.shape)    
        # print("force_waypoints_list :", force_waypoints_list.shape)    
        # reward_list = np.random.random((args.num_initial, 1))   
        # print("reward_list :", reward_list)   
        
        # ########## cal reward ###########     
        reward_list, success_list, force_list = get_reward_list(
            args=args, 
            waypoints_list=waypoints_list,  
            force_waypoints_list=force_waypoints_list,   
            obs_info=obs_info,  
            range_list=range_info,   
            center_list=center_info   
        )   
        print("============== evaluation info ============")   
        print("reward list :\n", reward_list)    
        print("success list :\n", success_list)         
        print("force list :\n", force_list)    
        reward_info = np.array(reward_list).reshape(-1, 1)     
    
    if args.move_zero:    
        tele_exp.moveToZero(
            np.array([-54.735610,-54.735610,-54.735610]),   
            args.speed,   
            args.delay_time   
        )   
    
    np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_traj_" + args.initial_data_name + ".npz", traj_info, range_info)    
    np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_random_training_data_" + args.initial_data_name + ".npz", context_info, para_info, obs_info, state_info, center_info, angle_info)   
    
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_reward_list_" + args.initial_data_name + ".npy", reward_info)     
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_state_list_" + args.initial_data_name + ".npy", state_info)      
         
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_success_list_" + args.initial_data_name + ".npy", success_list)     
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/initial_force_list_" + args.initial_data_name + ".npy", force_list)    
    return state_info, reward_info     


def load_initial_data(
        args=None, 
        data_name=None  
    ):  
    file_path_1_list=["./data/aim_data/Subject_" + str(args.subject_index) + "/initial_reward_list_" + data_name + ".npy"]
    file_path_2_list=["./data/aim_data/Subject_" + str(args.subject_index) + "/initial_random_training_data_" + data_name + ".npz"]
    file_path_3_list=["./data/aim_data/Subject_" + str(args.subject_index) + "/initial_state_list_" + data_name + ".npy"]
    file_path_4_list=["./data/aim_data/Subject_" + str(args.subject_index) + "/initial_traj_" + data_name + ".npz"]

    file_path_1, file_path_2, file_path_3, file_path_4 = file_path_1_list[0], file_path_2_list[0], file_path_3_list[0], file_path_4_list[0]
    
    reward_info = np.load(file_path_1)      
    initial_training_data = np.load(file_path_2)     
    state_info = np.load(file_path_3)     
    traj_results = np.load(file_path_4)      
    
    context_info = initial_training_data['arr_0']      
    para_info = initial_training_data['arr_1']     
    obs_info = initial_training_data['arr_2']    
    state_info = initial_training_data['arr_3']   
    center_info = initial_training_data['arr_4']    
    angle_info = initial_training_data['arr_5'].reshape(-1, 1)       
    print("================ saved info ==============")   
    print("reward info :\n", np.mean(reward_info, axis=0), np.std(reward_info, axis=0))    
    # print("success info :\n", success_info)         
    # print("force info :\n", force_info)    

    # # ############ cal reward ###########     
    # traj_info, range_info = traj_results["arr_0"], traj_results["arr_1"]      
    # real_waypoints = traj_info[::args.resample, :2]        
    # ref_waypoints = traj_info[::args.resample, 3:5]        
    # real_waypoints = real_waypoints.reshape((args.num_initial, args.nb_data, 2))       
    # ref_waypoints = ref_waypoints.reshape((args.num_initial, args.nb_data, 2))     
    # force_waypoints = traj_info[::args.resample, 9].reshape((args.num_initial, args.nb_data, 1))                 

    # reward_info, success_info, force_info = get_reward_list(
    #     args=args,   
    #     waypoints_list=ref_waypoints,   
    #     force_waypoints_list=force_waypoints,    
    #     obs_info=obs_info,  
    #     range_list=range_info,   
    #     center_list=center_info   
    # )   
    # print("============== evaluation info ============")   
    # print("reward info :\n", np.mean(reward_info, axis=0), np.std(reward_info, axis=0))     
    # print("success info :\n", np.sum(np.array(success_info))/np.array(success_info).shape[0])         
    # print("force info :\n", np.mean(force_info, axis=0), np.std(force_info, axis=0))    

    # np.save(file_path_1, np.array(reward_info).reshape(-1, 1))     
    return context_info, obs_info, center_info, state_info, reward_info, angle_info     


#### need to be revised ####
def get_episode_data(args=None, tele_exp=None, ref_data=None, data_name='epi_', 
                     context_info=None, obs_info=None, para_info=None, 
                     state_info=None, center_info=None, angle_info=None  
                ):  
    args.num = ref_data.shape[0]     
    K_p_re = np.array([1.0, 1.0, 1.0])      
    K_d_re = 0.01 * K_p_re    

    num_episodes = context_info.shape[0]    
    desire_start_list = 100 * np.ones(num_episodes)    

    ########### control method ######
    results = tele_exp.control(
        ref_data,   
        obs_info,       
        para_info,   
        desire_start_list,  
        K_p_re,   
        K_d_re,   
        args.num,      
        2000,    
        0.4, 
        args.ee_force_amp,       
        args.method_index, 
        0      
    )   
    traj_info, range_info = results[0], results[1]       
    
    real_waypoints = traj_info[::args.resample, :2]     
    real_waypoints = real_waypoints.reshape((num_episodes, args.nb_data, 2))      
    
    ref_waypoints = traj_info[::args.resample, 3:5]         
    ref_waypoints = ref_waypoints.reshape((num_episodes, args.nb_data, 2))     

    force_waypoints = traj_info[::args.resample, 9].reshape((num_episodes, args.nb_data, 1))                  

    reward_list, success_list, force_list = get_reward_list(
        args=args,   
        waypoints_list=ref_waypoints,   
        force_waypoints_list=force_waypoints,     
        obs_info=obs_info,  
        range_list=range_info,  
        center_list=center_info   
    )   
    # print("waypoints_list :", real_waypoints.shape)      
    # print("force_waypoints_list :", force_waypoints.shape)     
    print("success list :\n", success_list)  
    success_rate = np.array(success_list)  
    print("success rate :\n", np.sum(success_rate)/success_rate.shape[0])         
    print("reward list :\n", reward_list)      
    reward_info = np.array(reward_list).reshape(-1, 1)          
    
    ########### save data ##########     
    np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/traj_" + data_name + ".npz", traj_info, range_info)        
    np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_context_" + data_name + ".npz", context_info, para_info, obs_info, state_info, center_info, angle_info)   
    
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/state_list_" + data_name + ".npy", state_info)        
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/reward_list_" + data_name + ".npy", reward_info)             
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/success_list_" + data_name + ".npy", success_list)             
    np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/force_list_" + data_name + ".npy", force_list)     
    return reward_info    


def cps_learning(
        args=None, ref_data=None, tele_exp=None,   
        context_range=None, para_range=None,    
        angle_range=None, size_range=None   
    ):  
    
    ####### intial evalualtion for each subject ######   
    if args.load_subject_initial: 
        context_info, obs_info, center_info, state_info, reward_info, angle_info = load_initial_data(
            args=args, 
            data_name=args.initial_data_name  
        )   
    else:  
        state_info, reward_info = get_initial_data(
            args=args, 
            tele_exp=tele_exp, ref_data=ref_data, 
            context_range=context_range, para_range=para_range, size_range=size_range
        )   
    ####### intial evalualtion for each subject ######   

    
    if args.training_mode == 'BO':        
        # Gaussian process regressor with an RBF kernel    
        # kernel = RBF(length_scale=2.0)    
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))    
        gp_model = GaussianProcessRegressor(kernel=kernel)     
        
        sample_x = state_info   
        sample_y = reward_info   

        ############# construct state range or state space #################  
        state_space = get_state_para_range(args=args, given_seed=100, context_info=state_info[:, :2])  
        
        if args.load_policy:  
            ######################## load previous sample ############  
            sample_x_initial = np.load("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_x_" + str(args.initial_k) + ".npy")   
            sample_y_initial = np.load("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_y_" + str(args.initial_k) + ".npy")   
            args.initial_k += 1   
            sample_x = np.append(sample_x, sample_x_initial, axis=0)       
            sample_y = np.append(sample_y, sample_y_initial, axis=0)        

        for k in range(args.initial_k, args.initial_k+args.num_iters):   
            print("\\\\\\\\\\\\\\\\ REPS ITERATION INDEX \\\\\\\\\\\\\\\\", k)   

            # Fit the Gaussian process model to the sampled points   
            gp_model.fit(sample_x, sample_y)    
            args.beta = args.beta * pow(0.98, args.initial_k)   

            ############################ evaluation ################
            sample_y_pred = gp_model.predict(sample_x, return_std=False)
            print("mse :", mean_squared_error(sample_y_pred, sample_y))    

            # state_info_real = np.zeros((5, args.context_dim + args.para_dim))  
            # for context_index in range(5):    
            state_info_real = np.zeros((context_info.shape[0], args.context_dim + args.para_dim))   
            for context_index in range(context_info.shape[0]):    
                # Get part state space of the whole state space   
                part_state_space = state_space[context_index*args.num_sim_bo:(context_index+1)*args.num_sim_bo, :]    

                # Generate the Upper Confidence Bound (UCB) using the Gaussian process model
                ucb = upper_confidence_bound(part_state_space, gp_model, args.beta)    
                
                # Select the next point based on UCB    
                state_info_real[context_index, :] = part_state_space[np.argmax(ucb)][None, :]                
                # print("index :", np.argmax(ucb))      
                # print("state_info_real :", state_info_real.shape)     

            context_info_real, para_info_real, obs_info_real, center_info_real, angle_info_real = get_para_list(
                args=args,   
                state_info=state_info_real,    
                angle_range=angle_range,    
                size_range=size_range    
            )   
                
            ############ implementation by robot ################   
            reward_info_real= get_episode_data(args=args, tele_exp=tele_exp, ref_data=ref_data, data_name='epi_' + str(k),  
                    context_info=context_info_real, obs_info=obs_info_real, para_info=para_info_real, 
                    center_info=center_info_real, angle_info=angle_info_real
                )   
            #####################################################        
                
            sample_x = np.append(sample_x, state_info_real, axis=0)       
            sample_y = np.append(sample_y, reward_info_real, axis=0)        
            np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_x_" + str(k) + ".npy", sample_x)        
            np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_y_" + str(k) + ".npy", sample_y)   
        
        ############# evaluation implement sample contexts #################   
        if args.evaluate_policy:   
            # Fit the Gaussian process model to the sampled points   
            gp_model.fit(sample_x, sample_y)    
            # args.beta = args.beta * pow(0.98, args.initial_k)   

            ############################ evaluation ################
            sample_y_pred = gp_model.predict(sample_x, return_std=False)
            print("mse :", mean_squared_error(sample_y_pred, sample_y))    

            context_info_evaluation = context_info[:args.num_eva, :]  
            obs_info_evaluation = obs_info[:args.num_eva, :]  
            center_info_evaluation = center_info[:args.num_eva, :]   
            angle_info_evaluation = angle_info[:args.num_eva, :]
            para_info_evaluation = np.zeros((context_info.shape[0], args.para_dim))  
            for i in range(context_info_evaluation.shape[0]):  
                single_context_space = state_space[i * args.num_sim_bo:(i+1)* args.num_sim_bo, :]    
                ucb_context = upper_confidence_bound(single_context_space, gp_model, 0.0)   
                para_info_evaluation[i, :] = single_context_space[np.argmax(ucb_context)][None, args.context_dim:]     

            reward_info_evaluation = get_episode_data(args=args, tele_exp=tele_exp, ref_data=ref_data, data_name='eva_end',  
                            context_info=context_info_evaluation, obs_info=obs_info_evaluation, para_info=para_info_evaluation, 
                            center_info=center_info_evaluation, angle_info=angle_info_evaluation 
                        )   
            print("evaluation reward mean std :", np.mean(reward_info_evaluation, axis=0), np.std(reward_info_evaluation, axis=0))
        ############# evaluation implement sample contexts #################      

    if args.training_mode == "REPS":     
        # Gaussian process regressor with an RBF kernel    
        # kernel = RBF(length_scale=2.0)    
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))     
        gp_model = GaussianProcessRegressor(kernel=kernel)        
        
        reps_obs = GPREPS(
            context_dim=args.context_dim,       
            para_dim=args.para_dim,       
            para_lower_bound=0.0,    
            para_upper_bound=1.0,     
            eps=args.epi    
        )   
        
        if args.load_policy:   
            ######################## load policy parameter ###########   
            policy_para = np.load("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/policy_parameter_" + str(args.initial_k) + ".npz")  
            reps_obs_a, reps_obs_A, reps_obs_sigma = policy_para['arr_0'], policy_para['arr_1'], policy_para['arr_2'] 
            reps_obs.set_para(reps_obs_a, reps_obs_A, reps_obs_sigma)    
            
            ######################## load previous sample ############  
            sample_x = np.load("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_x_" + str(args.initial_k) + ".npy")   
            sample_y = np.load("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_y_" + str(args.initial_k) + ".npy")   
            args.initial_k += 1   
        else:  
            ############## implement sample contexts #################   
            # context_info, obs_info, center_info, state_info = sample_context(args=args, 
            #                                                 num_episodes=args.num_real, 
            #                                                 context_range=context_range,  
            #                                                 angle_range=angle_range, 
            #                                                 size_range=size_range   
            #                                             )   
            # para_info = reps_obs.get_para(context_info)   
            # reward_info = get_episode_data(args=args, tele_exp=tele_exp, ref_data=ref_data, data_name='eva_start',  
            #                 context_info=context_info, obs_info=obs_info, para_info=para_info, center_list=center_info
            #             )    
            # context_info, obs_info, center_info, state_info, reward_info = load_initial_data(args=args, data_name=args.data_name)     
            sample_x = state_info     
            sample_y = reward_info    
            args.initial_k = 0  
        
        for k in range(args.initial_k, args.initial_k + args.num_iters):     
            print("\\\\\\\\\\\\\\\\ REPS ITERATION INDEX \\\\\\\\\\\\\\\\", k)   
            # Fit the Gaussian process model to the sampled points
            gp_model.fit(sample_x, sample_y)               
            
            ################ for traing with artificial samples ######    
            context_info_sim, obs_info_sim, center_info_sim, angle_info_sim = sample_context(args=args, 
                                                num_episodes=args.num_sim,   
                                                context_range=context_range,      
                                                angle_range=angle_range,   
                                                size_range=size_range   
                                            )   
            para_info_sim = reps_obs.get_para(context_info_sim)      
            state_info_sim = np.hstack((context_info_sim, para_info_sim)).reshape(-1, args.context_dim + args.para_dim) 
            reward_info_sim, _ = gp_model.predict(state_info_sim, return_std=True)   
            reward_info_sim = reward_info_sim.reshape(-1, 1)  

            ########## update policy with artifical samples ##########  
            reps_obs.update_policy(context_info_sim, para_info_sim, reward_info_sim)     
            
            ####### random sample context from context space #########
            context_info_real, obs_info_real, center_info_real, angle_info_real = sample_context(args=args, 
                                                num_episodes=args.num_real,   
                                                context_range=context_range,      
                                                angle_range=angle_range,  
                                                size_range=size_range    
                                            )   
            para_info_real = reps_obs.get_para(context_info_real)    
            state_info_real = np.hstack((context_info_real, para_info_real)).reshape(-1, args.context_dim + args.para_dim) 
            ############ implementation by real robot ################   
            print("get real experimental data !!!")     
            reward_info_real= get_episode_data(args=args, tele_exp=tele_exp, ref_data=ref_data, data_name='epi_' + str(k),  
                        context_info=context_info_real, obs_info=obs_info_real, para_info=para_info_real, center_info=center_info_real, angle_info=angle_info_real
                    )   
            reward_info_real = np.random.random((state_info_real.shape[0], 1))   
            ##########################################################
            
            sample_x = np.append(sample_x, state_info_real, axis=0)    
            sample_y = np.append(sample_y, reward_info_real, axis=0)     
            
            ######################## save for offline training #######   
            print("save policy parameters !!!")     
            np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_x_" + str(k) + ".npy", sample_x)     
            np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/training_sample_y_" + str(k) + ".npy", sample_y)   
        
            ######################## save policy parameter ###########   
            np.savez("./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode + "/policy_parameter_" + str(k) + ".npz", reps_obs.a, reps_obs.A, reps_obs.sigma)     
        
        ############# evaluation implement sample contexts #################   
        if args.evaluate_policy:    
            context_info_evaluation = context_info[:args.num_eva, :]  
            obs_info_evaluation = obs_info[:args.num_eva, :]  
            center_info_evaluation = center_info[:args.num_eva, :]  
            angle_info_evaluation = angle_info[:args.num_eva, :]  
            para_info_evaluation = reps_obs.get_para(context_info_evaluation)        
            reward_info_evaluation = get_episode_data(args=args, tele_exp=tele_exp, ref_data=ref_data, data_name='eva_end',  
                            context_info=context_info_evaluation, obs_info=obs_info_evaluation, para_info=para_info_evaluation, center_info=center_info_evaluation, angle_info=angle_info_evaluation
                        )   
        ############# evaluation implement sample contexts #################      



if __name__ == "__main__":       
    parser = argparse.ArgumentParser()           

    # //////// mode /////////////  
    parser.add_argument('--ctl_mode', type=str, default="motion", help='choose mode first !!!!')      
    parser.add_argument('--bilateral', type=int, default=0, help='choose mode first !!!!')   
    
    parser.add_argument('--num', type=int, default=5000, help='choose index first !!!!')     
    parser.add_argument('--num_epi', type=int, default=1, help='give numn epi first !!!!')            
    
    parser.add_argument('--num_initial', type=int, default=20, help='give numn initial first !!!!')       
    parser.add_argument('--beta', type=float, default=2, help='beta !!!')           
    parser.add_argument('--num_iters', type=int, default=1, help='give numn initial first !!!!')          
    parser.add_argument('--num_sim_bo', type=int, default=200, help='give numn initial first !!!!')           
    parser.add_argument('--num_eva', type=int, default=20, help='numn evaluation !!!!')       
    parser.add_argument('--num_save', type=int, default=5, help='numn evaluation !!!!')       
    
    parser.add_argument('--num_pre', type=int, default=5, help='give numn episodes for preference learning !!!!')   
    
    parser.add_argument('--traj_deform', type=int, default=0, help='move to initial point !!!!')      
    parser.add_argument('--policy_learn', type=int, default=0, help='policy learning or not !!!!')    
    parser.add_argument('--context_dim', type=int, default=2, help='policy learning or not !!!!')       
    parser.add_argument('--obs_dim', type=int, default=3, help='policy learning or not !!!!')      
    parser.add_argument('--para_dim', type=int, default=2, help='policy learning or not !!!!')    
    parser.add_argument('--rotate_angle', type=float, default=0.0, help='radius of robot motion !!!!')    
    parser.add_argument('--num_real', type=int, default=2, help='policy learning or not !!!!')       
    parser.add_argument('--num_sim', type=int, default=100, help='policy learning or not !!!!')        
    parser.add_argument('--num_K', type=int, default=10, help='policy learning or not !!!!')       
    
    parser.add_argument('--initial_k', type=int, default=0, help='policy learning or not !!!!')   
    
    parser.add_argument('--load_policy', type=int, default=0, help='low new policy or not !!!!')          
    parser.add_argument('--evaluate_policy', type=int, default=1, help='evaluate policy or not !!!!')      
    parser.add_argument('--load_subject_initial', type=int, default=1, help='evaluate policy or not !!!!')          
    
    # //////// basics ///////////        
    parser.add_argument('--speed', type=int, default=10, help='select from {1, 2, 3}')      
    
    # //// path ////////////////      
    parser.add_argument('--file_name', type=str, default='x_p', help='load reference trajectory !!!')       
    parser.add_argument('--root_path', type=str, default='./data/tro_data', help='choose index first !!!!')      
    parser.add_argument('--load_data', type=int, default=0, help='choose index first !!!!')    
    parser.add_argument('--subject_index', type=int, default=0, help='subject index !!!!')             
    parser.add_argument('--method_index', type=int, default=1, help='method index !!!!')         
    parser.add_argument('--training_mode', type=str, default='REPS', help='training index !!!!')        
    
    # //// learning /////////////       
    parser.add_argument('--nb_data', type=int, default=200, help='choose index first !!!!')      
    parser.add_argument('--N_I', type=int, default=20, help='choose index first !!!!')     
    parser.add_argument('--nb_samples', type=int, default=5, help='load reference trajectory !!!')       
    parser.add_argument('--nb_states', type=int, default=25, help='choose index first !!!!')      
    parser.add_argument('--sample_num', type=int, default=5000, help='choose index first !!!!')      
    parser.add_argument('--input_dim', type=int, default=1, help='choose index first !!!!')      
    parser.add_argument('--output_dim', type=int, default=2, help='load reference trajectory !!!')     
    parser.add_argument('--dim', type=int, default=1, help='via num !!!')     
    parser.add_argument('--via_num', type=int, default=5, help='via num !!!')     
    parser.add_argument('--dt', type=float, default=0.01, help='choose mode first !!!!')       
    parser.add_argument('--lambda_1', type=float, default=0.001, help='lambda_1 !!!')    
    parser.add_argument('--lambda_2', type=float, default=10.0, help='lambda_2 !!!')      
    parser.add_argument('--kh', type=float, default=15.0, help='lambda_2 !!!')    
    parser.add_argument('--nb_dim', type=int, default=1, help='autoregression !!!')     
    parser.add_argument('--center_index', type=int, default=110, help='autoregression !!!')     
    parser.add_argument('--beta_t', type=float, default=0.3, help='beta !!!')       
    
    parser.add_argument('--max_iter', type=int, default=200, help='maximal iteration !!!')        
    parser.add_argument('--max_F', type=float, default=1.8, help='maximal iteration !!!')        
    
    parser.add_argument('--T', type=float, default=5, help='choose index first !!!!')      
    parser.add_argument('--T_s', type=float, default=0.001, help='load reference trajectory !!!')            
    
    parser.add_argument('--prior_flag', type=str, default='tele', help='choose index first !!!!')    
    parser.add_argument('--initial_data_name', type=str, default='fixed_initial_second', help='initial data name !!!!')    
    parser.add_argument('--data_name', type=str, default='episode_', help='data name !!!!')    
    parser.add_argument('--resample', type=int, default=25, help='resample index !!!')    
    
    args = parser.parse_args()    
    para_input = obtain_para_input(args)         
    
    #################################################    
    context_range = np.array([[1.0, 1.0, 7.0], [0.0, 0.0, 3.0]])                   
    para_range = np.array([[0.3, 0.5], [0.1, 0.1]])   
    angle_range = np.linspace(-0.7 * np.pi, 0.7 * np.pi, args.nb_data)   
    size_range = np.linspace(context_range[1, 2], context_range[0, 2], args.nb_data)     
    
    # ############ get subject data ################   
    # sample_x, sample_y = get_initial_data(
    #     args=args, tele_exp=tele_exp, ref_data=ref_data,  
    #     context_range=context_range, para_range=para_range, 
    #     angle_range=angle_range, size_range=size_range      
    # )    
    # context_info, obs_info, center_info, state_info, reward_info, angle_info = load_initial_data(args=args, data_name=args.initial_data_name)   
    # print("reward_info :", reward_info)  
    ########### get subject data ################   
        

    # ########### cps learning #########################  
    # cps_learning(
    #     args=args,  
    #     tele_exp=tele_exp,   
    #     ref_data=ref_data,  
    #     context_range=context_range,    
    #     para_range=para_range,   
    #     angle_range=angle_range,   
    #     size_range=size_range    
    # )   
    # ########### cps learning #########################   


    ############ get preference data ################## 
    get_preference_data(args=args, ref_data=ref_data, tele_exp=tele_exp)     
    ############ get preference data ################## 

    ###################################################
    # context_info, obs_info, center_info, state_info, reward_info, angle_info = load_initial_data(
    #         args=args, 
    #         data_name=args.initial_data_name  
    #     )   
    # print("state_info :", state_info.shape, reward_info.shape, angle_info.shape, obs_info.shape, center_info.shape)    
    # sample_x = state_info   
    # sample_y = reward_info  
    # print("sample_y :", sample_y)  
    # # Gaussian process regressor with an RBF kernel    
    # # kernel = RBF(length_scale=1000)  
    # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))   
    # # kernel = DotProduct() + WhiteKernel(noise_level=0.5)  
    # gp_model = GaussianProcessRegressor(kernel=kernel, random_state=0)   
    
    # # Fit the Gaussian process model to the sampled points   
    # gp_model.fit(sample_x, sample_y)   
    # print(gp_model.score(sample_x, sample_y))  

    # sample_y_pred, sample_y_std = gp_model.predict(sample_x, return_std=True)
    # print("y_pred :", sample_y_pred.T, sample_y_std.T)    
    # print("mse :", mean_squared_error(sample_y_pred, sample_y))      

    # state_space = get_state_para_range(args=args, context_info=state_info[:, :2])  
    # # print("sample_x :", sample_x)     
    # # print("state_space :", state_space)     
    # # y_pred, y_std = gp_model.predict(state_space, return_std=True)   
    # # print("y_pred :", y_pred, y_std)  

    # ######### construct state range or state space #################   
    # context_info, obs_info, center_info, state_info, reward_info, angle_info = load_initial_data(
    #     args=args,   
    #     data_name=args.initial_data_name    
    # )   
    
    # evaluate_data(args=args, data_name_list = ["epi_3"]) 

    # ######### construct state range or state space #################  

    # # # Generate the Upper Confidence Bound (UCB) using the Gaussian process model
    # ucb = upper_confidence_bound(state_space, gp_model, args.beta)    
    # # sample_x = np.append(state_info, state_info, axis=0)  
    # sample_x = np.load("data/aim_data/Subject_1/method_1/BO/training_sample_x_4.npy")
    # sample_y = np.load("data/aim_data/Subject_1/method_1/BO/training_sample_y_4.npy")   
    # print(sample_x.shape)    
    # print(sample_y.shape)      
    # state_space = get_state_para_range(args=args, context_info=np.array([[2.0, 2.0], [3.0, 3.0]]))  
    # print(state_space)     

    #########################################     
    # initial_data = generate_initial_context_para(
    #     args=args, 
    #     context_range=context_range, 
    #     para_range=para_range,
    #     data_name='fixed_context' 
    # )   
    
    # context_info, obs_info, center_info = sample_context(args=args, 
    #                                                      num_episodes=args.num_real, 
    #                                                      context_range=context_range,  
    #                                                      angle_range=angle_range  
    #                                                     )    
    
    # para_info = reps_obs.get_para(context_info)  
    # print(para_info.shape)    
    # true_reward = context_info.dot(np.array([0.5, 0.5])) + para_info.dot(np.array([0.5, 0.5]))
    # print("reward shape :", true_reward)   
    # sample_x = np.hstack((context_info, para_info)).reshape(-1, args.context_dim + args.para_dim)   
    # sample_y = true_reward.reshape(-1, 1)  
    # # Gaussian process regressor with an RBF kernel    
    # kernel = RBF(length_scale=1.0)     
    # gp_model = GaussianProcessRegressor(kernel=kernel)     
    # context_sim, obs_sim, center_sim = sample_context(args=args, 
    #                                                   num_episodes=args.num_sim,   
    #                                                   context_range=context_range,    
    #                                                   angle_range=angle_range  
    #                                                 )   
    # print("context_sim :", context_sim.shape)     
    # para_sim = reps_obs.get_para(context_sim)     
    # print("para_sim", para_sim.shape)    
    # true_reward_sim = context_sim.dot(np.array([0.5, 0.5])) + para_sim.dot(np.array([0.5, 0.5]))
    # print("reward_sim", true_reward_sim.shape)     
    # sample_x_sim = np.hstack((context_sim, para_sim)).reshape(-1, args.context_dim + args.para_dim)  
    # sample_y_sim, y_std = gp_model.predict(sample_x_sim, return_std=True)  
    # error = np.sqrt(mean_squared_error(true_reward_sim, sample_y_sim)) 
    # print("rmse error :", error)    
    # reward_sim = true_reward_sim.reshape(-1, 1)  
    # ### update policy 
    # reps_obs.update_policy(context_sim, para_sim, reward_sim)       
    
    # plot_figure_all_info(
    #     args=args,    
    #     x_e=None,   
    #     y_e=None,       
    #     x_t=None,     
    #     y_t=None,              
    #     range_list=None,  
    #     ee_force_t=None,    
    #     obs_info=None,   
    #     mu_gmr=None,   
    #     sigma_gmr=None,     
    #     mu_kmp=pred_mu_kmp,     
    #     sigma_kmp=pred_sigma_kmp,           
    #     via_points=via_points,      
    #     fig_name='actual_kmp_latest'               
    # )   
    # ########### evaluate gmr and kmp results ############    

    # generate_context_para(args=args, context_range=context_range, para_range=para_range)  
    # ########### preference learning 
    # get_preference_data(args=args, ref_data=ref_data, tele_exp=tele_exp, data_name='pre', pre_idx=0)    
    # ########### evaluate gmr and kmp results ############   