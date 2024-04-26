import numpy as np
import os
import sys
import argparse 

sys.path.append("/home/yuxuan/Project/HPS_Perception/map_ws/src/HPS_Perception/hps_control/scripts")
from learning.mini_core.utils import *
from learning.mini_core.gait_recorder import GaitRecorder
from learning.mini_core.learning_cost import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor   
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct 
from sklearn.metrics import mean_squared_error




def upper_confidence_bound(x, gp_model, beta, state_dim=4):   
    y_pred, y_std = gp_model.predict(x.reshape(-1, state_dim), return_std=True)
    print("y_pred :", y_pred, y_std)  
    ucb = y_pred + beta * y_std    
    ### output y predict ###    
    return ucb    


def get_state_para_range(args=None):  
    kp1_x = np.linspace(0,1,args.state_space_grid_width)
    kp2_x = np.linspace(0,1,args.state_space_grid_width)
    kp1_x, kp2_x = np.meshgrid(kp1_x, kp2_x)
    kp1_x, kp2_x = kp1_x.flatten(), kp2_x.flatten()

    state_space = np.zeros((args.state_space_grid_width**2,4))
    state_space[:,0] = normalize(1,lim_v) 
    state_space[:,1] = normalize(0,lim_s)
    state_space[:,2] = kp1_x
    state_space[:,3] = kp2_x
    return state_space  


def sample_context(num_real=10, num_sim=2000, give_seed=None, give_index=None):     
    if give_seed is not None: 
        np.random.seed(give_seed)    
    

    #### real context #####
    # context_v = np.array([0.8, 1.0, 1.1, 1.2, 1.3, 1.4])  
    # context_s = np.array([-8,-4,-2,0,2,4,8]) 
    context_v = np.array([0.8, 1.0, 1.1, 1.1, 1.2, 1.3, 0.8, 0.8, 0.8, 0.8])  
 
    context_s = np.array([0,0,0,0,0,0,2,4,6,8]) 
    # V,S = np.meshgrid(context_v, context_s)
    # V,S = V.flatten(),S.flatten()
    # idx_chosen = np.random.randint(0,np.shape(V)[0],size=(num_real,))
    context_real = np.zeros((num_real,2))  

    context_real[:, 0] = context_v[give_index]
    context_real[:, 1] = context_s[give_index]  

    # context_real[:, 0] = V[idx_chosen]
    # context_real[:, 1] = S[idx_chosen]
    print("Con_real", context_real)
    
    ##### sim context #####   
    context_sim = np.random.random(size=(num_sim, 2))
    # context_sim[:,0] = scaling(context_sim[:,0], lim_v)
    # context_sim[:,1] = scaling(context_sim[:,1], lim_s)

    return context_real, context_sim   


if __name__ == "__main__":       
    parser = argparse.ArgumentParser()           

    # //////// mode /////////////  
    parser.add_argument('--ctl_mode', type=str, default="motion", help='choose mode first !!!!')      
    parser.add_argument('--bilateral', type=int, default=0, help='choose mode first !!!!')   
    
    parser.add_argument('--num', type=int, default=5000, help='choose index first !!!!')     
    parser.add_argument('--num_epi', type=int, default=1, help='give numn epi first !!!!')            
    
    parser.add_argument('--num_initial', type=int, default=20, help='give numn initial first !!!!')       
    parser.add_argument('--beta', type=float, default=1, help='beta !!!')           
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
    parser.add_argument('--num_real', type=int, default=10, help='policy learning or not !!!!')       
    parser.add_argument('--num_sim', type=int, default=2000, help='policy learning or not !!!!')        
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
    
    parser.add_argument("--state_space_grid_width", type=int, default=50, help="state space grid width!!!")
    parser.add_argument("--ucb_sample_num", type=int, default=10, help="new ucb generated sample!!!")
    parser.add_argument("--eps", type=int, default=3.5, help="")
    parser.add_argument("--training_data_path", type=str, default="/media/yuxuan/BackUp/HPS_Data/Debug_Ba/")
    parser.add_argument("--reps_para_path", type=str, default="/media/yuxuan/BackUp/HPS_Data/Debug_REPS/")
    
    args = parser.parse_args()       
    
    
    ##########training
    evaluation = 9 
    training_mode = "REPS" 

    state_info = None
    reward_info = None
    for i in range(evaluation+1):
        saving_pair_path = args.training_data_path+"e{}/pair/".format(int(evaluation))
        if i == 0:
            state_info, reward_info = load_state_and_cost_list(saving_pair_path)
            reward_info = 1-reward_info
        else:
            state_info_temp, reward_info_temp = load_state_and_cost_list(saving_pair_path)
            reward_info_temp = 1-reward_info_temp
            state_info = np.vstack([state_info, state_info_temp])
            reward_info = np.hstack([reward_info, reward_info_temp])

    reward_info = np.reshape(reward_info, (-1,1))

    if training_mode == "BO": 
        state_space = get_state_para_range(args=args)    

        ## TODO 可能需要改length_scale的大小
        kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 10.0)) 
        gp_model = GaussianProcessRegressor(kernel=kernel)

        sample_x = state_info        
        sample_y = reward_info

        ### fit  
        # Fit the Gaussian process model to the sampled points   
        gp_model.fit(sample_x, sample_y)
        args.beta = args.beta * pow(0.98, evaluation)
   
        print("Score", gp_model.score(sample_x, sample_y))    
        
        predict_sample_y = gp_model.predict(sample_x)
    

        state_info_real = np.zeros((args.ucb_sample_num, args.context_dim + args.para_dim))
        state_info_real[:,0] = 1 # 固定v
        state_info_real[:,1] = 0 # 固定s
        
        # Generate the Upper Confidence Bound (UCB) using the Gaussian process model
        # TODO beta 大探索范围大，从大到小

        ucb = upper_confidence_bound(state_space, gp_model, 0)
        idx_ucb_chosen = np.argsort(ucb)[-1:-args.ucb_sample_num-1:-1] #argsort 从小到大，选择最大的ucb_sample_num个
    
        for i in range(args.ucb_sample_num):    
            state_info_real[i, :] = state_space[idx_ucb_chosen[i]][None, :]     

        # 5个随机，5个策略
        idx_chosen_grid = np.random.randint(low=100,high=2400,size=(5,))
        state_info_real[:5,-2:] = state_space[idx_chosen_grid, -2:]

    if training_mode == "REPS":
         # Gaussian process regressor with an RBF kernel    
        # kernel = RBF(length_scale=2.0)    
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))     
        gp_model = GaussianProcessRegressor(kernel=kernel)        
        from reps import GPREPS
        reps_obs = GPREPS(
            context_dim=args.context_dim,     
            para_dim=args.para_dim,       
            para_lower_bound=0.0,    
            para_upper_bound=1.0,    
            eps=1.0   
        )   
        
        if args.load_policy:   
            ######################## load policy parameter ########### 
            if evaluation == 3:
                reps_obs_a = np.zeros_like(reps_obs.a)
                reps_obs_A = np.zeros_like(reps_obs.A)
                reps_obs_sigma = np.zeros_like(reps_obs.sigma)
            elif evaluation > 3:
                policy_para = np.load(args.reps_para_path+"policy_parameter_" + str(args.initial_k) + ".npz")  
                reps_obs_a, reps_obs_A, reps_obs_sigma = policy_para['arr_0'], policy_para['arr_1'], policy_para['arr_2'] 
            
            reps_obs.set_para(reps_obs_a, reps_obs_A, reps_obs_sigma)
           
        sample_x = state_info     
        sample_y = reward_info    

        
        for k in range(args.initial_k, args.initial_k + args.num_iters):     
            print("\\\\\\\\\\\\\\\\ REPS ITERATION INDEX \\\\\\\\\\\\\\\\", k)   
            # Fit the Gaussian process model to the sampled points
            gp_model.fit(sample_x, sample_y)               
            
            ################ for traing with artificial samples ######    
            _, context_info_sim = sample_context(num_sim=2000, give_index=evaluation)
            
            # for c in context_info_sim:
            #     context_info_sim[c, 0] = normalize(context_info_real[c,0], lim_v)
            #     context_info_sim[c, 1] = normalize(context_info_real[c,1], lim_s)  
            para_info_sim = reps_obs.get_para(context_info_sim)      
            state_info_sim = np.hstack((context_info_sim, para_info_sim)).reshape(-1, args.context_dim + args.para_dim) 
            reward_info_sim, _ = gp_model.predict(state_info_sim, return_std=True)   
            reward_info_sim = reward_info_sim.reshape(-1, 1)

            ########## update policy with artifical samples ##########  
            reps_obs.update_policy(context_info_sim, para_info_sim, reward_info_sim)     
            
            ####### random sample context from context space #########
            context_info_real, _ = sample_context(num_real=args.num_real,give_index=evaluation)
            for c in range(np.shape(context_info_real)[0]):
                context_info_real[c, 0] = normalize(context_info_real[c,0], lim_v)
                context_info_real[c, 1] = normalize(context_info_real[c,1], lim_s)
            # context_info_real[:,0] = normalize(context_info_real[:,0], lim_v)
            # context_info_real[:,1] = normalize(context_info_real[:,1], lim_s)
            para_info_real = reps_obs.get_para(context_info_real)    
            state_info_real = np.hstack((context_info_real, para_info_real)).reshape(-1, args.context_dim + args.para_dim) 
            np.savez(args.reps_para_path+"e{}/".format(evaluation)+"policy_parameter_" + str(k) + ".npz", reps_obs.a, reps_obs.A, reps_obs.sigma)     


    
    # TODO 生成新的impedance
    total_num = args.ucb_sample_num
    idx_knee = 0
    idx_ankle = 1
    new_kp1 = state_info_real[:,2]
    new_kp1 = scaling(new_kp1, lim_kp[idx_knee][1])
    print("New Kp1", new_kp1)
    new_kp2 = state_info_real[:,3]
    new_kp2 = scaling(new_kp2, lim_kp[idx_knee][2])
    print("New Kp2", new_kp2)
    imp_total = np.zeros((total_num,24))
    imp_total[:,0:8] = np.array(k0[idx_knee]+k0[idx_ankle])
    imp_total[:,8:16] = np.array(b0[idx_knee]+b0[idx_ankle])
    imp_total[:,16:] = np.array(q_e0[idx_knee]+q_e0[idx_ankle])
    imp_total[:, 1] = new_kp1[:]
    imp_total[:, 2] = new_kp2[:]  
    
    if check_generated_imp(imp_total):
        print("Save New Parameters in evaluation {}".format(evaluation+1))
        np.save("/media/yuxuan/BackUp/HPS_Data/Debug_REPS/e{}/state_info_real.npy".format(evaluation+1), state_info_real)
        np.save("/home/yuxuan/Project/HPS_Perception/map_ws/src/HPS_Perception/hps_control/scripts/learning/eval_para/eval_{}.npy".format(evaluation+1),imp_total)
