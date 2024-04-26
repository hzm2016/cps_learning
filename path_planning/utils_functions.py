from matplotlib.pyplot import title   
from numpy.lib import NumpyVersion   
import numpy as np  
import math  
import os  
import ctypes  
import time   
import glob   
import scipy  
import argparse   
from sklearn.metrics import mean_squared_error    
from path_planning.kmp.demo_GMR import *    
# from plot_figures_main import *   
import scipy.io as scio    

import seaborn as sns    
# sns.set(font_scale=1.5)    
np.set_printoptions(precision=4)     



def cal_evaluation_cartesian(
    real_data=None, ref_data=None, args=None   
    ):  
    #  ////////////////////// sample data /////////////////////  
    nb_data = args.nb_data       
    nb_samples = args.nb_samples       
    
    nb_data_sup = 0          
    dt = args.dt          
    T = dt * nb_data        
    print("duration :", T)          
    
    # model parameter     
    input_dim = args.input_dim           
    output_dim = args.output_dim     

    demos = real_data     
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]       
    print("demos_t :", np.array(demos_t[0]).shape)       

    # Stack time and position data      
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]

    all_real_data = [np.hstack([-1 * demos[i*nb_data:(i+1)*nb_data, 0][:, None], -1 * demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("all_real_data :", np.array(all_real_data).shape)      
    
    # Stack demos    
    demos_np = demos_tx[0]     
    print("demos_np :", demos_np.shape)     

    for i in range(1, nb_samples):    
        demos_np = np.vstack([demos_np, demos_tx[i]])    
    print("demos_np :", demos_np.shape)     
    
    X = demos_np[:, 0][:, None]    
    Y = demos_np[:, 1:]    
    print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]   
    
    mu_gmr, sigma_gmr, gmr_model = GMR_pred(  
        demos_np=demos_np,     
        X=X,  
        Xt=Xt,     
        Y=Y,     
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,     
        output_dim=output_dim,     
    )   

    # mu_gmr, sigma_gmr, gmr_model_c_5 = GMR_pred(  
    #     demos_np=demos_np,     
    #     X=X,  
    #     Xt=Xt,     
    #     Y=Y,     
    #     nb_data=args.nb_data,    
    #     nb_samples=args.nb_samples,     
    #     nb_states=5,     
    #     input_dim=input_dim,     
    #     output_dim=output_dim,     
    # )   
    
    # plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    
    plot_GMM_raw_data(  
        nb_samples=args.nb_samples, nb_data=args.nb_data, Y=Y, ref_data=ref_data, 
        gmr_model=gmr_model, gmr_model_second=gmr_model_c_5, 
        font_name=args.data_name, fig_path=args.fig_path, save_fig=args.save_fig
    )    
    
    plot_mean_var(   
        nb_samples=args.nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, 
        mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data, 
        font_name=args.data_name, fig_path=args.fig_path, save_fig=args.save_fig   
    )   
    
    real_error = all_real_data - ref_data.T       
    print("real_error :", real_error.shape)      
    return mu_gmr, ref_data, np.mean(real_error**2), np.std(np.array(real_error))    


def cal_adaptation_cartesian(
    real_data=None,   
    ref_data=None,   
    obs_center=None,    
    viaNum=None,   
    viaFlag=None,   
    via_time=None,   
    via_points=None,   
    via_var_list=None,   
    ori_data_list=None,  
    start=None,    
    end=None,    
    start_index=None,    
    end_index=None,    
    args=None   
):     
    #### ///////////////// model parameter //////////////////// ###  
    nb_data = args.nb_data       
    nb_samples = args.nb_samples       
    nb_data_sup = 0       
    dt = 0.01      
    T = dt * nb_data       
    print("T :", T)         
    input_dim = args.input_dim        
    output_dim = args.output_dim      
    #### ///////////////// model parameter //////////////////// ###    

    demos = real_data   

    print("demos :", demos.shape)    
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]      
    print("demos_t :", np.array(demos_t[0]).shape)      

    # # Stack time and position data
    # demos_tx = [np.hstack([demos_t[i] * dt, demos[i, 1, :][:, None], demos[i, 0, :][:, None]]) for i in range(nb_samples)]
    # print("demos_tx :", np.array(demos_tx).shape)      
    
    # Stack time and position data      
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    # , demos[i*nb_data:(i+1)*nb_data, 2][:, None]   
    # print("demos_tx :", np.array(demos_tx).shape)     

    real_real_data = [np.hstack([-1 * demos[i*nb_data:(i+1)*nb_data, 0][:, None], -1 * demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_real_data :", np.array(real_real_data).shape)   
    
    real_joint_error = real_real_data - ref_data.T       
    print("real_joint_error :", real_joint_error.shape)         
    
    # Stack demos    
    demos_np = demos_tx[0]     
    print("demos_np :", demos_np.shape)     

    for i in range(1, nb_samples):    
        demos_np = np.vstack([demos_np, demos_tx[i]])    
    print("demos_np :", demos_np.shape)     
    
    X = demos_np[:, 0][:, None]     
    Y = demos_np[:, 1:]    
    print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    # Test data     
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]     
    
    mu_gmr, sigma_gmr, gmr_model = GMR_pred(  
        demos_np=demos_np,     
        X=X,    
        Xt=Xt,     
        Y=Y,     
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim   
    )   

    # plot_mean_var(
    #     font_name=args.data_name + '_' + str(args.iter), 
    #     nb_samples=args.nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, 
    #     mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data, 
    #     fig_path=args.fig_path, save_fig=args.save_fig
    # )   
    
    # //////////////////////////////////////////////////// 
    # print("ref_data :", ref_data.shape)   
    # viaNum = 2   
    # viaFlag = np.ones(viaNum)    
    # via_time = np.zeros(viaNum)         
    # via_points = np.zeros((viaNum, output_dim))    
    
    # via_time[0] = dt    
    # via_points[0, :] = np.array([0.0, -23.0])        
    # via_time[1] = 1.    
    # via_points[1, :] = np.array([0.0, 10.0])       
    # # via_points[1, :] = np.array([0.0, 28.0])    

    # # via_time[0] = dt    
    # # via_points[0, :] = np.array([0.0, -23.0])     
    # index_list = [start_index, (start_index+end_index)//2, end_index]   
    # print("index_list :", index_list, args.nb_data, T)
    # # index_list = [end_index, (start_index+end_index)//2, start_index]   
    # viaNum = len(index_list)     
    # viaFlag = np.ones(viaNum)    
    # via_time = np.zeros(viaNum)         
    # via_points = np.zeros((viaNum, output_dim))    
    # for i, index in enumerate(index_list):    
    #     via_time[i] = index/args.nb_data * T     
    #     via_time[i] = (index - 50)/args.nb_data * T  
    #     via_points[i, :] = ref_data[:2, index]  
    #     # via_points[i, :] = np.array([ref_data[1, index], ref_data[0, index]])       
    #     via_var = 1E-6 * np.eye(output_dim)    
    #     # via_time[0] = 1.   
    #     # via_points[0, :] = np.array([0.0, 24.0])    
    #     # via_var = 1E-6 * np.eye(output_dim)    
    # print("time :", via_time, "via_points :", via_points)  

    # via_points[0, :] = np.array([0.0, 15.0])     
    # # via_var = 1E-6 * np.eye(4)    
    # # via_var[2, 2] = 1000    
    # # via_var[3, 3] = 1000    
    # ////////////////////////////////////////////////////
           
    ori_refTraj, refTraj, kmpPredTraj = KMP_pred(
        Xt=Xt,  
        mu_gmr=mu_gmr,  #  ref_data.T[:, :2]  
        sigma_gmr=sigma_gmr,   
        viaNum=viaNum,    
        viaFlag=viaFlag,      
        via_time=via_time,     
        via_points=via_points,      
        via_var_list=via_var_list,     
        dt=dt,   
        len=args.nb_data,       
        lamda_1=1,      
        lamda_2=20,          
        kh=20,     
        output_dim=output_dim,    
        dim=1        
    )   
    

    plot_via_points(   
        nb_posterior_samples=viaNum,     
        via_points=via_points,     
        mu_gmr=ori_refTraj['mu'],       
        pred_gmr=kmpPredTraj['mu'],      
        sigma_gmr=refTraj['sigma'],      
        sigma_kmp=kmpPredTraj['sigma'],       
        ref_data=ref_data,     
        obs_center=obs_center,    
        real_data=ori_data_list,    
        start=start, end=end,  
        font_name=args.data_name,     
        save_fig=args.save_fig,    
        fig_path=args.fig_path     
    )   
     
    # plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    # plot_GMM_raw_data(font_name=args.data_name + '_' + str(args.iter), nb_samples=5, nb_data=200, Y=Y, gmr_model=gmr_model)   
    # plot_mean_var(font_name=args.data_name + '_' + str(args.iter), nb_samples=args.nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)
    # plot_mean_var_error(font_name=args.data_name, real_data=real_real_data, ref_data=ref_data.T)  
    return np.mean(real_joint_error**2), np.std(np.array(real_joint_error))   


def cal_iteration_evaluation_joint(joint_data=None, args=None):  
    print("Cal interation evaluation !!!")
    nb_data = args.nb_data     
    nb_samples = args.nb_samples     
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    # print("demodura :", demodura)       
    
    # model parameter 
    input_dim = 1   
    # output_dim = 2     
    output_dim = 3     
       
    # index_num = iter * 200   
    resample_index = int(joint_data.shape[0]/1000)     

    # /// velocity  
    start_1 = 25     
    start_2 = 13       
    demos_vel = joint_data[::resample_index, start_2:start_2+3]      
    ref_data_vel = joint_data[::resample_index, start_1:start_1+3]   
    
    # /// position 
    start_3 = 1      
    start_4 = 7      
    demos_pos = joint_data[::resample_index, start_4:start_4+3]        
    ref_data_pos = joint_data[::resample_index, start_3:start_3+3]        
    
    # # /// ori pos
    # ori_tau_list = ori_tau_list[::resample_index, :]      
    # ori_tau_list = ori_tau_list[:nb_data, :]      
    
    # ori_stiff_list = ori_stiff_list[::resample_index, :]      
    # ori_stiff_list = ori_stiff_list[:nb_data, :]    
    
    # ori_damping_list = ori_damping_list[::resample_index, :]      
    # ori_damping_list = ori_damping_list[:nb_data, :]      
    
    # ori_stiff_data = ori_stiff_data[::resample_index, :]      
    # ori_stiff_data = ori_stiff_data[:nb_data, :]      
    ref_data_pos = ref_data_pos[:nb_data, :]      
    ref_data_vel = ref_data_vel[:nb_data, :]      
    
    # ////
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]       
    # print("demos_t :", np.array(demos_t[0]).shape)     

    # # Stack time and position data
    # Stack time and position data  
    demos_tx_pos = [np.hstack([demos_t[i] * dt, demos_pos[i*nb_data:(i+1)*nb_data, 0][:, None], demos_pos[i*nb_data:(i+1)*nb_data, 1][:, None], demos_pos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    
    # Stack time and position data  
    demos_tx_vel = [np.hstack([demos_t[i] * dt, demos_vel[i*nb_data:(i+1)*nb_data, 0][:, None], demos_vel[i*nb_data:(i+1)*nb_data, 1][:, None], demos_vel[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    
    # print("demos_tx :", np.array(demos_tx_pos).shape, np.array(demos_tx_vel).shape)      

    # real_joint_data = [np.hstack([demos[i*nb_data:(i+1)*nb_data, 0][:, None], demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_joint_data :", np.array(real_joint_data).shape)     
    
    # Stack demos    
    demos_np_pos = demos_tx_pos[0]     
    demos_np_vel = demos_tx_vel[0]            

    for i in range(1, nb_samples):     
        demos_np_pos = np.vstack([demos_np_pos, demos_tx_pos[i]])      
        demos_np_vel = np.vstack([demos_np_vel, demos_tx_vel[i]])       
    # print("demos_np :", demos_np_pos.shape)     
    
    X_p = demos_np_pos[:, 0][:, None]    
    Y_p = demos_np_pos[:, 1:]      
    # print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    X_v = demos_np_vel[:, 0][:, None]    
    Y_v = demos_np_vel[:, 1:]      
    
    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]    
    
    # //// pos
    mu_gmr_pos, sigma_gmr_pos, gmr_model_pos = GMR_pred(  
        demos_np=demos_np_pos,     
        X=X_p,  
        Xt=Xt,     
        Y=Y_p,      
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim,    
        data_name=args.data_name + "_pos"
    )   
    
    # //// vel  
    mu_gmr_vel, sigma_gmr_vel, gmr_model_vel = GMR_pred(   
        demos_np=demos_np_vel,     
        X=X_v,    
        Xt=Xt,      
        Y=Y_v,       
        nb_data=args.nb_data,     
        nb_samples=args.nb_samples,      
        nb_states=args.nb_states,      
        input_dim=input_dim,     
        output_dim=output_dim,      
        data_name=args.data_name + "_vel"      
    )    
    
    plot_raw_data(font_name=args.data_name + '_pos_ours', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X_p, Y=Y_p, mu_gmr=mu_gmr_pos, sigma_gmr=sigma_gmr_pos, ref_data=ref_data_pos)    
    plot_raw_data(font_name=args.data_name + '_vel_ours', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X_v, Y=Y_v, mu_gmr=mu_gmr_vel, sigma_gmr=sigma_gmr_vel, ref_data=ref_data_vel)
    
    error_q, sigma_q = np.array(mu_gmr_pos) - ref_data_pos, sigma_gmr_pos     
    error_d_q, sigma_d_q = np.array(mu_gmr_vel) - ref_data_vel, sigma_gmr_vel    
    
    print("error_q :", np.mean(error_q), "error_d_q :", np.mean(error_d_q))    
    return np.mean(error_q), np.mean(error_d_q), np.mean(sigma_q), np.mean(sigma_d_q)    


def generate_single_path(angle_ampl=30, angle_fre=0.15, num=5000, force_ampl=0.5, force_fre=0.15, plot=False, Tf=0.001):      
    T = 1/angle_fre      
    index_list = np.linspace(0, T, num)      
    q_t = angle_ampl * np.sin(2 * np.pi * angle_fre * index_list + 3/2 * np.pi) + angle_ampl      
    d_q_t = angle_ampl * 2 * np.pi * angle_fre * np.cos(2 * np.pi * angle_fre * index_list + 3/2 * np.pi)  
    ref_data = np.vstack((q_t, d_q_t)).transpose()      
    print("Ref data :", ref_data.shape) 
    if (plot): 
        # Plot the curve
        plt.plot(index_list, q_t, label='q_t')    
        plt.plot(index_list, d_q_t, label='d_q_t')       

        # Add labels and title
        plt.xlabel('x')   
        plt.ylabel('y')  
        plt.title('y = 5 * sin(2 * pi * 0.1 * x + 3/2 *pi) + 5')
        plt.legend()

        # Display the plot
        plt.show()
        # final_data = np.vstack((all_data, all_data[::-1, :]))
        # print(final_data.shape)  
            
        plt.savefig('single_joint.png', bbox_inches='tight', dpi=300, pad_inches=0.0) 
        # np.savetxt("./data/wrist_demo/demo_data/trajectory_theta_list_" + args.flag + ".txt", final_data, fmt='%f', delimiter=',')  
        
    return ref_data  


def smooth(y, box_pts):  
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')  
    return y_smooth   


def GMR_path(
    args=None
):     
    for iter in range(args.num):                   
        # ///////////////// reference path ////////////////////   
        file_path = args.root_path + '/demo_data/trajectory_theta_list_' + args.file_name + '.txt'        
        ref_data = np.loadtxt(file_path, dtype=float, delimiter=',')         
        # initial_joint = ref_data[0, :]      
        initial_joint = np.array([-54.73, -54.73, -54.73])         
        print("initial_joint :", initial_joint)          
        args.iter = iter     
        
        # stiff_data = np.zeros((args.num, 3))        
        # damping_data = np.zeros((args.num, 3))        
        # iter_tau_data = np.zeros((args.num, 3))         
        # iter_stiff_data = np.zeros((args.num, 3))          
        # iter_damping_data = np.zeros((args.num, 3))          
    
        save_path = args.root_path + '/save_data/' + args.folder_name       
         
        # # # ///////////// plot data ///////////  
        # flag = "_baletral_" + control_mode + "_demo_" + flag + "_" + str(iter)     
        save_flag = "_baletral_" + args.control_mode + "_demo_" + args.flag + "_" + str(iter)           
    
        trajectory_theta_t_list = np.loadtxt(args.root_path + "/" + "save_data/" + args.folder_name + "/wrist_encoder" + save_flag + ".txt",  delimiter=',', skiprows=1)  
        trajectory_kinematics_t_list = np.loadtxt(args.root_path + "/" + "save_data/" + args.folder_name + "/wrist_kinematics" + save_flag + ".txt",  delimiter=',', skiprows=1) 
        trajectory_iteration_t_list = np.loadtxt(args.root_path + "/" + "save_data/" + args.folder_name + "/wrist_iteration" + save_flag + ".txt",  delimiter=',', skiprows=1) 
        
        mean_value, std_value = cal_adaptation_cartesian(trajectory_kinematics_t_list, args=args)    


def ori_tro_paper():
    # // revise tro paper   
    ori_tau_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_" + str(iter-1) + ".txt")     
    ori_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_" + str(iter-1) + ".txt")     
    ori_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_" + str(iter-1) + ".txt")    
    
    stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/stiff_baletral_motion_demo_iteration_learning_0.txt")      
    damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/damping_baletral_motion_demo_iteration_learning_0.txt")       

    iter_tau_data, iter_stiff_data, iter_damping_data = cal_iteration_speed(trajectory_theta_t_list, ori_tau_list=ori_tau_data, ori_stiff_list=ori_stiff_data, ori_damping_list=ori_damping_data, args=args, iter_value=iter_value)  
    
    print(iter_tau_data.shape, iter_stiff_data.shape)   
    np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_" + str(iter) + ".txt",  cp.deepcopy(iter_tau_data))      
    np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_" + str(iter) + ".txt",  cp.deepcopy(iter_stiff_data))       
    np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_" + str(iter) + ".txt",  cp.deepcopy(iter_damping_data))        
    
    # plot_stiff_damping(font_name=args.data_name, stiff_data=iter_stiff_data, damping_data=iter_damping_data, tau_data=iter_tau_data, stiff_scale=None) 

    # # ///// mean variance   
    # last_flag = "_baletral_" + control_mode + "_demo_" + args.exp_flag + "_" + str(iter)    
    # trajectory_theta_t_list = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/wrist_encoder" + last_flag + ".txt",  delimiter=',', skiprows=1)    
    
    # ori_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/stiff" + last_flag + ".txt")     
    # ori_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/damping" + last_flag + ".txt")       
    
    # damping_data = np.tile(K_d, (args.num, 1))   
    # stiff_data, damping_data, error_data, mean_value, std_value = cal_iteration_stiffness(trajectory_theta_t_list, ori_stiff_data=ori_stiff_data, args=args)   
    
    # ori_tau_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_" + str(iter-1) + ".txt")     
    # ori_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_" + str(iter-1) + ".txt")    
    # ori_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_" + str(iter-1) + ".txt")  
    
    # stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/stiff_baletral_motion_demo_iteration_learning_0.txt")    
    # damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/damping_baletral_motion_demo_iteration_learning_0.txt")      

    # iter_tau_data, iter_stiff_data, iter_damping_data = cal_iteration_speed(trajectory_theta_t_list, ori_tau_list=ori_tau_data, ori_stiff_list=ori_stiff_data, ori_damping_list=ori_damping_data, args=args) 
    # plot_stiff_damping(font_name=args.data_name, stiff_data=iter_stiff_data, damping_data=iter_damping_data, tau_data=iter_tau_data, stiff_scale=None)  
 

def new_tro_paper():   
    # cal_iteration_speed(trajectory_theta_t_list, args=args)      
    # cal_iteration_speed(trajectory_theta_t_list, ori_tau_list=None, ori_stiff_list=None, ori_damping_list=None, args=args)  
    mean_value_q, mean_value_d_q, std_value_q, std_value_d_q = cal_iteration_evaluation_joint(joint_data=trajectory_theta_t_list)  
    
    iter_tau_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_0.txt")     
    iter_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_0.txt")    
    iter_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_0.txt")     
        
    if iter==0:    
        stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/stiff" + flag + ".txt")   
        damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data/damping" + flag + ".txt")   
        
        iter_tau_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/tau_epi_" + str(iter) + ".txt")     
        iter_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/stiff_epi_" + str(iter) + ".txt")    
        iter_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro/damping_epi_" + str(iter) + ".txt")        
    else:    
        last_flag = "_baletral_" + control_mode + "_demo_" + args.exp_flag + "_" + str(iter-1)   
        trajectory_theta_t_list = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro_ours_tmech/wrist_encoder" + last_flag + ".txt",  delimiter=',', skiprows=1) 
        
        # // t-mech paper   
        ori_stiff_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro_ours_tmech/stiff" + last_flag + ".txt")    
        ori_damping_data = np.loadtxt(args.root_path + "/" + "save_data/ilc_data_tro_ours_tmech/damping" + last_flag + ".txt")      
        
        # damping_data = np.tile(K_d, (args.num, 1))   
        stiff_data, damping_data, error_data, mean_value, std_value = cal_iteration_stiffness(trajectory_theta_t_list, ori_stiff_data=ori_stiff_data, args=args)   
    
        # last_flag = "_baletral_" + control_mode + "_demo_" + args.exp_flag + "_" + str(iter)    
        np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro_ours_tmech/stiff" + flag + ".txt",  stiff_data)       
        np.savetxt(args.root_path + "/" + "save_data/ilc_data_tro_ours_tmech/damping" + flag + ".txt",  damping_data)        
        
        # mean_list.append(mean_value)    
        # std_list.append(std_value)    

    
def partial_deform_trajectory(
    ref_data=None,   
    inter_force=None,   
    beta=0.1   
):  
    ref_data_deformed = ref_data     
    N = inter_force.shape[0]  
    
    G = np.zeros((N, N))        
    I = np.identity(N)     
    Z = np.zeros((4, N))  # for smoothness   
    Z[0, 0] = 1     
    Z[1, 1] = 1    
    Z[2, N-2] = 1       
    Z[3, N-1] = 1     
    
    Q = np.zeros((N+3,N))  
    for i in range(N): 
        Q[i, i] = 1 
        Q[i+1, i] = -3
        Q[i+2, i] = 3
        Q[i+3, i] = -1 
    
    R = np.zeros((N, N))    
    R = Q.T.dot(Q)    
    # print("R :", R)   
    G = (I - np.linalg.inv(R).dot(Z.T).dot(np.linalg.inv(Z.dot(np.linalg.inv(R)).dot(Z.T))).dot(Z)).dot(np.linalg.inv(R))
    G = G/np.linalg.norm(G)  
    # print("G :", G)   
    # print("inter_force", inter_force)     
    ref_data_deformed = ref_data + np.sqrt(N) * beta * G.dot(inter_force)    
    # print("ref_deformed :", G.dot(inter_force))    
    return ref_data_deformed, G   


def obstacle_avoidance(
    ):      
    fig = plt.figure(figsize=(6, 4))     
    ax = fig.add_subplot(111) 
    data_list_x = [] 
    data_list_y = [] 
    for i in range(1,4): 
        path_x = './data/Obstacle_DATA/H30D30_' + str(i) + '_x.mat'    
        path_y = './data/Obstacle_DATA/H30D30_' + str(i) + '_y.mat'    
        data_x = list(scio.loadmat(path_x).values())[-1]     
        data_y = list(scio.loadmat(path_y).values())[-1]      
        # print("shape_y :", list(data_y.keys())[3])          
        # print("shape_x :", list(data_x.keys())[3])         
        # print("shape_y :", np.array(data_y['toe_p3']).shape)       
        # print("shape_x :", np.array(data_x['toe_p1']).shape)        

        # ax.plot(data_x, label='x')   
        # ax.plot(data_y, label='y')   
        #  ///////////////////////////////////////////////////     
        data_list_x.append(np.array(data_x)[::2][50:])   
        data_list_y.append(np.array(data_y)[::2][50:])    

    data_list_x = np.array(data_list_x) 
    data_list_y = np.array(data_list_y) 
    print("data_list_x :", np.array(data_list_x).shape)  
    nb_data = 200       
    nb_samples = 3   
    nb_states = 5      
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    # print("demodura :", demodura)     
    
    # model parameter     
    input_dim = 1       
    output_dim = 2        

    # resampled_theta_t = joint_data[::resample_index, :]     
    # demos = resampled_theta_t[:1000, :]       
    # ref_data = joint_data[::resample_index, start_1:start_1+3]       
    # ref_data = np.load('./data/wrist_demo/save_data/ilc_data/tracking_epi_joint_5_des.npy', allow_pickle=True)
    # print("ref_data :", ref_data)     
    # ori_stiff_data = ori_stiff_data[::resample_index, :]       
    # ori_stiff_data = ori_stiff_data[:nb_data, :]       
    # ref_data = ref_data[:nb_data, :]      
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]      
    print("demos_t :", np.array(demos_t[0]).shape)       

    # # Stack time and position data
    # demos_tx = [np.hstack([demos_t[i] * dt, demos[i, 1, :][:, None], demos[i, 0, :][:, None]]) for i in range(nb_samples)]
    # print("demos_tx :", np.array(demos_tx).shape)      
    
    # Stack time and position data      
    demos_tx = [np.hstack([demos_t[i] * dt, data_list_x[i], data_list_y[i]]) for i in range(nb_samples)]  
    # demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    # , demos[i*nb_data:(i+1)*nb_data, 2][:, None]   
    # print("demos_tx :", np.array(demos_tx).shape)     

    # real_joint_data = [np.hstack([-1 * demos[i*nb_data:(i+1)*nb_data, 0][:, None], -1 * demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_joint_data :", np.array(real_joint_data).shape)   
    
    # real_joint_error = real_joint_data - ref_data.T       
    # print("real_joint_error :", real_joint_error.shape)        
    
    # Stack demos    
    demos_np = demos_tx[0]     
    print("demos_np :", demos_np.shape)     

    for i in range(1, nb_samples):    
        demos_np = np.vstack([demos_np, demos_tx[i]])    
    print("demos_np :", demos_np.shape)     
    
    X = demos_np[:, 0][:, None]    
    Y = demos_np[:, 1:]    
    print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]   
      
    mu_gmr, sigma_gmr, gmr_model = GMR_pred(  
        demos_np=demos_np,     
        X=X,  
        Xt=Xt,     
        Y=Y,     
        nb_data=nb_data,    
        nb_samples=nb_samples,     
        nb_states=nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim,    
        data_name='kevin_obstacle_avoid'     
    )   
    
        # # // wrist robot   
    viaNum = 2   
    viaFlag = np.ones(viaNum)    
    via_time = np.zeros(viaNum)       
    via_points = np.zeros((viaNum, output_dim))   
    
    via_time[0] = dt    
    via_points[0, :] = np.array([0.6, 0.0])     
    via_time[1] = 0.7  
    via_points[1, :] = np.array([1.2, 0.8])  
    
    via_var = 1E-5 * np.eye(output_dim)    
    # # via_var = 1E-6 * np.eye(4)  
    # # via_var[2, 2] = 1000   
    # # via_var[3, 3] = 1000   
    # ///////////////////////////////////////////
           
    ori_refTraj, refTraj, kmpPredTraj = KMP_pred(
        Xt=Xt,  
        mu_gmr=mu_gmr,   
        sigma_gmr=sigma_gmr,   
        viaNum=2,    
        viaFlag=viaFlag,      
        via_time=via_time,     
        via_points=via_points,      
        via_var=via_var,    
        dt=dt,   
        len=args.nb_data,       
        lamda_1=1,      
        lamda_2=60,          
        kh=6,   
        output_dim=output_dim,    
        dim=1,      
        data_name='kevin_obstacle_avoid'       
    )    
    
    plot_via_points(  
        font_name='kevin_obstacle_avoid',   
        nb_posterior_samples=viaNum,   
        via_points=via_points,   
        mu_gmr=ori_refTraj['mu'],   
        pred_gmr=kmpPredTraj['mu'],   
        sigma_gmr=ori_refTraj['sigma'],   
        sigma_kmp=kmpPredTraj['sigma']    
    )   
    
    plot_mean_var_fig(font_name='kevin_obstacle_avoid', nb_samples=nb_samples, nb_data=200, Xt=Xt, Y=Y, mu_gmr=ori_refTraj['mu'], sigma_gmr=ori_refTraj['sigma'], pred_gmr=kmpPredTraj['mu'], pred_sigma=kmpPredTraj['sigma'], ref_data=None, via_points=via_points, via_time=via_time)
    # plot_raw_data(font_name='kevin_obstacle_avoid', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)  
    # plot_GMM_raw_data(font_name='kevin_obstacle_avoid', nb_samples=3, nb_data=200, Y=Y, gmr_model=gmr_model)
        
    plt.ylabel("X[mm]")         
    plt.xlabel("Y[mm]")         
    # ax.set_xlim(-25.0, 25.0)       
    # ax.set_ylim(-25.0, 25.0) 

    plt.tight_layout()       
    # plt.axis('equal')      
    plt.legend(loc="upper right")     

    plt.savefig('reach_targets_circles' + flag + '.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)          
        # np.save('./path_planning/data/wrist/reach_target.npy', np.array(re_sample_data))    
    plt.show()            

    
def cal_stiff(error_data, ori_stiff_data, sigma_data, scale=1/50.0, alpha_t=0.1, beta_t=0.1):     
    stiff_data = []     
    for i in range(error_data.shape[0]):     
        stiff = (1 - math.exp(-alpha_t * np.linalg.norm(sigma_data[i, ::], ord=2))) * beta_t * np.abs(error_data[i, :])    
        stiff_data.append(stiff)     
        # path_name = path_name_list[i]    
        # path_angle = np.loadtxt(path_name, delimiter=',', skiprows=1)     
        # mse = mean_squared_error(path_angle[:, 8], path_angle[:, 10])     
        # print("mse :", math.sqrt(mse))    

    # scale = 1/50.0   
    stiff_data = np.array(stiff_data)    
    stiff_data = stiff_data + ori_stiff_data     
    damping_data = scale * stiff_data     
    
    return stiff_data, damping_data    


def cal_stiff_scale(sigma_demo_data, alpha_t=0.1, beta_t=0.1):     
    stiff_scale = []    
    for i in range(sigma_demo_data.shape[0]):     
        stiff = (1 - math.exp(-alpha_t * np.linalg.norm(sigma_demo_data[i, ::], ord=2))) + beta_t     
        stiff_scale.append(stiff)     
        # path_name = path_name_list[i]     
        # path_angle = np.loadtxt(path_name, delimiter=',', skiprows=1)     
        # mse = mean_squared_error(path_angle[:, 8], path_angle[:, 10])     
        # print("mse :", math.sqrt(mse))    
    
    stiff_scale = np.array(stiff_scale)     
    
    return stiff_scale    


def cal_tau_stiff_damping(error_q, error_d_q, ori_tau_list, ori_stiff_list, ori_damping_list, scale=1/50, decay_factor=np.array([0.8, 0.8, 0.8]), learning_rate=np.array([0.1, 0.1, 0.1])):   
    # ////// calculate s_q  
    s_q = scale * error_d_q + error_q    
    
    tau_list = decay_factor[0] * ori_tau_list + learning_rate[0] * s_q   
    stiff_list = decay_factor[1] * ori_stiff_list + learning_rate[1] * s_q * error_q  
    damping_list = decay_factor[2] * ori_damping_list + learning_rate[2] * s_q * error_d_q   
    
    return tau_list, stiff_list, damping_list    

    
def cal_iteration_stiffness(joint_data, ori_stiff_data=None, args=None):   
    nb_data = args.nb_data     
    nb_samples = args.nb_samples     
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    print("demodura :", demodura)     
    
    # model parameter 
    input_dim = 1   
    # output_dim = 2     
    output_dim = 3     
       
    # index_num = iter * 200   
    resample_index = int(joint_data.shape[0]/1000)     

    start_1 = 1      
    start_2 = 7      
    demos = joint_data[::resample_index, start_2:start_2+3]      
    ref_data = joint_data[::resample_index, start_1:start_1+3]     
    
    ori_stiff_data = ori_stiff_data[::resample_index, :]      
    ori_stiff_data = ori_stiff_data[:nb_data, :]      
    ref_data = ref_data[:nb_data, :]      
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]     
    print("demos_t :", np.array(demos_t[0]).shape)     

    # # Stack time and position data
    # demos_tx = [np.hstack([demos_t[i] * dt, demos[i, 1, :][:, None], demos[i, 0, :][:, None]]) for i in range(nb_samples)]
    # print("demos_tx :", np.array(demos_tx).shape) 
    
    # Stack time and position data  
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 0][:, None], demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    print("demos_tx :", np.array(demos_tx).shape)    

    real_joint_data = [np.hstack([demos[i*nb_data:(i+1)*nb_data, 0][:, None], demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    print("real_joint_data :", np.array(real_joint_data).shape)     
    
    # Stack demos    
    demos_np = demos_tx[0]     
    print("demos_np :", demos_np.shape)     

    for i in range(1, nb_samples):    
        demos_np = np.vstack([demos_np, demos_tx[i]])    
    print("demos_np :", demos_np.shape)     
    
    X = demos_np[:, 0][:, None]    
    Y = demos_np[:, 1:]    
    # print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]   
    
    mu_gmr, sigma_gmr, gmr_model = GMR_pred(  
        demos_np=demos_np,     
        X=X,  
        Xt=Xt,     
        Y=Y,     
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim,    
        data_name=args.data_name     
    )   
    
    plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    # plot_GMM_raw_data(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Y=Y, gmr_model=gmr_model)   
    # plot_mean_var(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_x=ref_x)   
    # plot_mean_var_fig(font_name=data_name, nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_x=ref_x)   
    
    error_data, sigma_data = np.array(mu_gmr) - ref_data, sigma_gmr         
    
    real_joint_error = real_joint_data - ref_data     
    print("real joint error :", np.mean(np.array(real_joint_error)), np.std(np.array(real_joint_error)))    
    # real_error_data = demos.reshape()  
    # print("mse :",  mean_squared_error(ref_data, np.array(mu_gmr)))   
    # print("error_data :", error_data)   
    stiff_data, damping_data = cal_stiff(error_data, ori_stiff_data, sigma_data, scale=1/50.0, alpha_t=0.15, beta_t=20.0)   
    # stiff_data, damping_data = cal_error(error_data, sigma_data, alpha_t=0.1, beta_t=0.1)   
    
    stiff_scale = cal_stiff_scale(sigma_gmr, alpha_t=0.2, beta_t=0.75)    
    
    # ///// data iterpration   
    stiff_interp = cal_iterp_data(stiff_data, N=5000)        
    print("shape :", stiff_interp.shape)     
    
    damping_interp = cal_iterp_data(damping_data, N=5000)    
    print("shape :", damping_interp.shape)    
    
    scale_interp = cal_iterp_scale(stiff_scale, N=5000)      
    print("shape :", damping_interp.shape)    
    
    # plot_stiff_damping(font_name=args.data_name, stiff_data=stiff_interp, damping_data=damping_interp, stiff_scale=scale_interp)   
    
    return stiff_interp.T, damping_interp.T, error_data, np.mean(np.array(real_joint_error)), np.std(np.array(real_joint_error))   


def cal_iteration_speed(joint_data, ori_tau_list=None, ori_stiff_list=None, ori_damping_list=None, args=None, iter_value=None):   
    nb_data = args.nb_data     
    nb_samples = args.nb_samples     
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    print("demodura :", demodura)       
    
    # model parameter 
    input_dim = 1   
    # output_dim = 2     
    output_dim = 3     
       
    # index_num = iter * 200   
    resample_index = int(joint_data.shape[0]/1000)     

    # /// velocity  
    start_1 = 25     
    start_2 = 13       
    demos_vel = joint_data[::resample_index, start_2:start_2+3]      
    ref_data_vel = joint_data[::resample_index, start_1:start_1+3]   
    
    # /// position 
    start_3 = 1      
    start_4 = 7      
    demos_pos = joint_data[::resample_index, start_4:start_4+3]        
    ref_data_pos = joint_data[::resample_index, start_3:start_3+3]       
    
    # /// ori pos
    ori_tau_list = ori_tau_list[::resample_index, :]      
    ori_tau_list = ori_tau_list[:nb_data, :]      
    
    ori_stiff_list = ori_stiff_list[::resample_index, :]      
    ori_stiff_list = ori_stiff_list[:nb_data, :]    
    
    ori_damping_list = ori_damping_list[::resample_index, :]      
    ori_damping_list = ori_damping_list[:nb_data, :]      
    
    # ori_stiff_data = ori_stiff_data[::resample_index, :]      
    # ori_stiff_data = ori_stiff_data[:nb_data, :]      
    ref_data_pos = ref_data_pos[:nb_data, :]      
    ref_data_vel = ref_data_vel[:nb_data, :]      
    
    # ////
    demos_t = [np.arange(nb_data)[:, None] for i in range(nb_samples)]       
    print("demos_t :", np.array(demos_t[0]).shape)     

    # # Stack time and position data
    # Stack time and position data  
    demos_tx_pos = [np.hstack([demos_t[i] * dt, demos_pos[i*nb_data:(i+1)*nb_data, 0][:, None], demos_pos[i*nb_data:(i+1)*nb_data, 1][:, None], demos_pos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    
    # Stack time and position data  
    demos_tx_vel = [np.hstack([demos_t[i] * dt, demos_vel[i*nb_data:(i+1)*nb_data, 0][:, None], demos_vel[i*nb_data:(i+1)*nb_data, 1][:, None], demos_vel[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    
    print("demos_tx :", np.array(demos_tx_pos).shape, np.array(demos_tx_vel).shape)      

    # real_joint_data = [np.hstack([demos[i*nb_data:(i+1)*nb_data, 0][:, None], demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_joint_data :", np.array(real_joint_data).shape)     
    
    # Stack demos    
    demos_np_pos = demos_tx_pos[0]     
    demos_np_vel = demos_tx_vel[0]            

    for i in range(1, nb_samples):     
        demos_np_pos = np.vstack([demos_np_pos, demos_tx_pos[i]])      
        demos_np_vel = np.vstack([demos_np_vel, demos_tx_vel[i]])       
    print("demos_np :", demos_np_pos.shape)     
    
    X_p = demos_np_pos[:, 0][:, None]    
    Y_p = demos_np_pos[:, 1:]      
    # print('X shape: ', X.shape, 'Y shape: ', Y.shape)    
    
    X_v = demos_np_vel[:, 0][:, None]    
    Y_v = demos_np_vel[:, 1:]      
    
    # Test data   
    Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]    
    
    # //// pos
    mu_gmr_pos, sigma_gmr_pos, gmr_model_pos = GMR_pred(  
        demos_np=demos_np_pos,     
        X=X_p,  
        Xt=Xt,     
        Y=Y_p,      
        nb_data=args.nb_data,    
        nb_samples=args.nb_samples,     
        nb_states=args.nb_states,     
        input_dim=input_dim,    
        output_dim=output_dim,    
        data_name=args.data_name + "_pos"
    )   
    
    # //// vel  
    mu_gmr_vel, sigma_gmr_vel, gmr_model_vel = GMR_pred(   
        demos_np=demos_np_vel,     
        X=X_v,    
        Xt=Xt,      
        Y=Y_v,       
        nb_data=args.nb_data,     
        nb_samples=args.nb_samples,      
        nb_states=args.nb_states,      
        input_dim=input_dim,     
        output_dim=output_dim,      
        data_name=args.data_name + "_vel"      
    )   
    
    error_q, sigma_q = np.array(mu_gmr_pos) - ref_data_pos, sigma_gmr_pos    
    print("error_q :", np.max(error_q))    
    error_d_q, sigma_d_q = np.array(mu_gmr_vel) - ref_data_vel, sigma_gmr_vel    
    print("error_d_q :", np.max(error_d_q))           
    
    tau_list, stiff_list, damping_list = cal_tau_stiff_damping(error_q, error_d_q, ori_tau_list, ori_stiff_list, ori_damping_list, scale=iter_value[0, 0],  decay_factor=iter_value[1, :], learning_rate=iter_value[2, :])       
    
    # ///// data iterpration 
    print("stiff shape :", stiff_list.shape)  
    stiff_interp = cal_iterp_data(stiff_list, N=args.num)       
    print("stiff shape :", stiff_interp.shape)    
    
    print("damping shape :", damping_list.shape)   
    damping_interp = cal_iterp_data(damping_list, N=args.num)      
    print("damping shape :", damping_interp.shape)    
    
    print("tau shape :", tau_list.shape)   
    tau_interp = cal_iterp_data(tau_list, N=args.num)      
    print("tau shape :", damping_interp.shape)    
    
    # plot_raw_data(font_name=args.data_name + '_pos', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X_p, Y=Y_p, mu_gmr=mu_gmr_pos, sigma_gmr=sigma_gmr_pos, ref_data=ref_data_pos)    
    
    # plot_raw_data(font_name=args.data_name + '_vel', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X_v, Y=Y_v, mu_gmr=mu_gmr_vel, sigma_gmr=sigma_gmr_vel, ref_data=ref_data_vel)

    return tau_interp.T, stiff_interp.T, damping_interp.T    