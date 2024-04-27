from numpy.lib import NumpyVersion  
import numpy as np  
import math  
import os    
import scipy.io as scio  
import ctypes    
import time   
import glob   
import scipy  
import argparse  
from sklearn.metrics import mean_squared_error       
from path_planning.kmp.demo_GMR import *    
# from plot_figures_main import *    
from path_planning.utils_functions import *       
# from path_planning.spm_path.spm_kinematics import *      
from motor_control import wrist_control   
from matplotlib.pyplot import title      
import seaborn as sns       
# sns.set(font_scale=1.5)       
np.set_printoptions(precision=4)         
from path_planning.cps.reps import *       



def plot_via_points(
    ref_data=None,     
    real_data=None,      
    nb_via_points=None,      
    via_points=None,    
    mu_gmr=None,    
    mu_kmp=None,    
    sigma_gmr=None,     
    sigma_kmp=None,     
    deform_path=None,   
    obs_center=None,    
    obs_r=None,    
    start=None,  
    end=None,   
    font_name='G',    
    save_fig=None,   
    fig_path=None    
):  
    plt.ion()
    plt.figure(figsize=(5, 5))   
    
    ##################  obstacle  ######################
    draw_circle_obs = plt.Circle(obs_center, obs_r, fill=True, color=mycolors['lb'], alpha=0.6, label='c_')      
    # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')   
    plt.gcf().gca().add_artist(draw_circle_obs)     
    # # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')       

    ##################  ref data  ######################
    plt.plot(ref_data[:, 0], ref_data[:, 1], color="black", linewidth=3)     

    #################  real data  ######################    
    if real_data is not None:   
        # for p in range(len(real_data)-1, len(real_data)):       
        #     X = real_data[p][:, 0]       
        #     Y = real_data[p][:, 1]      
        #     plt.plot(X, Y, linewidth=2.5, color=mycolors[color_name[0]])    
        X = real_data[:, 0]       
        Y = real_data[:, 1]       
        plt.plot(X, Y, linewidth=1.5, color=[.7, .7, .7])    
        # for p in range(nb_samples):   
        #     plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
        #     plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')    

    #################  via points  ######################    
    if via_points is not None:  
        plt.scatter(via_points[0, 0], via_points[0, 1], color=[0.64, 0., 0.65], marker='X', s=80, label='via-points')    
        for i in range(1, nb_via_points):  
            # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
            plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)      
        
        plt.text(via_points[1, 0] - 5.0, via_points[1, 1] + 0.5, r'$t_s$', size=font_size)   
        plt.text(via_points[-2, 0] + 2.0, via_points[-2, 1] + 0.5, r'$t_e$', size=font_size)   

    # # ori value    
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80, label='start-points')   
    # plt.scatter(mu_gmr[198, 0], mu_gmr[198, 1], color='green', marker='X', s=80, label='end-points')   

    if mu_gmr is not None: 
        plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.94, 0.54], linewidth=3)  
        plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.1, color=[0.20, 0.93, 0.54])  

        # # pred value  
        # plt.scatter(mu_kmp[0, 0], mu_kmp[0, 1], color='black', marker='X', s=80, label='start-points')   
        # plt.scatter(mu_kmp[196, 0], mu_kmp[196, 1], color='green', marker='X', s=80, label='end-points')   
    
    if mu_kmp is not None:   
        plt.plot(mu_kmp[:, 0], mu_kmp[:, 1], color=[0.20, 0.54, 0.93], linewidth=3, linestyle='--')  
        plot_gmm(mu_kmp[:, :2], sigma_kmp[:, :2, :2], alpha=0.1, color=[0.20, 0.54, 0.93])    

        # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80)   
        # plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.05, color=[0.20, 0.54, 0.93])    
    
    if start != None:   
        # plot force   
        plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], fc='k', ec='k', 
                lw=3, length_includes_head=True, head_width=1.0, head_length=1.0, color='k') # fc='k', ec='k', lw=2, , length_includes_head=True, head_width=0.2, head_length=0.3, color='k'
        # plt.text(end[0] + 1.5, end[1] + 1.2, r'$F_h$', size=font_size)    
        plt.text(end[0] - 5.0, end[1] + 0.2, r'$F_h$', size=font_size)     
        
    if deform_path is not None:   
        plt.plot(deform_path[:, 0], deform_path[:, 1], color="red", linewidth=3)     
        plt.scatter(deform_path[0, 0], deform_path[0, 1], color=[0.64, 0., 0.65], marker='X', s=80)    
        plt.scatter(deform_path[-1, 0], deform_path[-1, 1], color=[0.64, 0.65, 0.], marker='X', s=80)      

    plt.ylabel("Y[mm]", fontsize=font_size)         
    plt.xlabel("X[mm]", fontsize=font_size)        
    plt.xlim(-30.0, 30.0)       
    plt.ylim(-30.0, 30.0)       
    plt.legend()       
    plt.locator_params(nbins=3)      
    plt.tick_params(labelsize=font_size)       
    plt.tight_layout()     

    if save_fig:   
        print(fig_path + font_name + '_kmp.pdf')    
        plt.savefig(fig_path + font_name + '_kmp.pdf', bbox_inches='tight', pad_inches=0.0)    
    # plt.show()    

    plt.pause(10)     
    # plt.close(all)      


def farthest_point_sampling(points, num_points):   
    # Number of points in the input sequence    
    n = points.shape[0]      

    # Initialize the list of selected points with the first point
    selected_indices = [0]    
    selected_points = points[0].reshape(1, -1)     

    # Compute the pairwise Euclidean distance matrix
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - selected_points, axis=-1)

    # Iteratively select the farthest point until reaching the desired number of points
    for _ in range(1, num_points):
        farthest_distances = np.min(dist_matrix, axis=1)  # Minimum distances to the selected points
        farthest_index = np.argmax(farthest_distances)  # Index of the farthest point
        selected_indices.append(farthest_index)
        selected_points = np.vstack((selected_points, points[farthest_index]))
        dist_matrix = np.minimum(dist_matrix, np.linalg.norm(points[farthest_index] - points, axis=-1))

    return selected_points, selected_indices  


# # Test the function with a sample sequence of points
# sequence = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# num_selected_points = 3  
# selected_points, selected_indices = farthest_point_sampling(sequence, num_selected_points)
# print("Selected Points:")    
# print(selected_points)    
# print("Selected Indices:")     
# print(selected_indices)      


def adaptation_learning():  
    # ///////////////////////// ref data //////////////////////
    # phi_range = [0, 2*np.pi]   
    phi_range = [-1/2*np.pi, 3/2*np.pi]   
    # phi_range = [-np.pi/2, 3/2*np.pi]  
    # theta_range = [7/8*np.pi, 7/8*np.pi]     
    theta_range = [8/9*np.pi, 8/9*np.pi]     
    sample_num_1 = [1, 200]    
    theta = np.linspace(theta_range[0], theta_range[1], sample_num_1[0])       
    phi = np.linspace(phi_range[0], phi_range[1], sample_num_1[1])        
    t, p = np.meshgrid(theta, phi)    
    
    r = 70     
    x = r*np.sin(t)*np.cos(p)      
    y = r*np.sin(t)*np.sin(p)     
    z = r*np.cos(t)     
    
    x = x.flatten()      
    y = y.flatten()     
    z = z.flatten()     

    ref_data = np.vstack((x, y, z))      
    ref_data = ref_data[:, ::-1]   

    # middle_ref_data = ref_data
    # middle_ref_data[:, :100] = ref_data[:, 50:150]  
    # middle_ref_data[:, 100:150] = ref_data[:, 150:200] 
    # middle_ref_data[:, 150:] = ref_data[:, :50] 

    # ref_data = middle_ref_data  
    # ///////////////////////////// evaluation //////////////////////////////////////  
    # mean_value_q, mean_value_d_q, std_value_q, std_value_d_q = cal_iteration_evaluation_joint(joint_data=trajectory_theta_t_list, args=args) 
    
    T = args.dt * args.nb_data    
    print("ref_data :", ref_data.shape, T)    

    N_I = 40       
    start_index = 80           
    # N_I = 50   
    # start_index = 75         
    force_dim = 3    
    end_index = start_index + N_I    
    obs_center = np.array([0.0, 21.0])    
    # obs_center = np.array([13.5, 13.5])      
    center_index = start_index + int(N_I/2)     
    # inter_force = np.ones((N, force_dim))    
    inter_force = np.tile(np.array([0, -1, 1]), (N_I, 1))     
    # inter_force = np.tile(np.array([-1, -1, 1]), (N, 1))   
    # print(inter_force.shape)   
    # start_point = ref_data[start_index, :]    
    # end_point = ref_data[end_index, :]  

    ref_data = ref_data.T   
    # beta_list = [0.5, 0.9, 1.2]     
    beta_list = [0.5, 1.0, 1.2]    
    label_list = [r'$\beta=0.5$', r'$\beta=0.9$', r'$\beta=1.2$']      

    real_data_list = []    

    # start = ref_data[start_index + N//2, :2]   
    # end = np.array([0, 8.])   

    start = np.array([0.0, 28.0])  
    end = np.array([0.0, 8.0])   

    # start = np.array([19.0, 19.0])    
    # end = np.array([5.0, 5.0])    
    
    max_via_index = []   
    print("start :", start)     

    for beta in beta_list:    
        partial_ref_data = cp.deepcopy(ref_data[start_index:start_index+N_I, :])    
        inter_force = np.tile(np.array([0, -1, 1]), (N_I, 1))   
        deformed_data, G = partial_deform_trajectory(
            ref_data=partial_ref_data,    
            inter_force=inter_force,    
            beta=beta    
        )   
        real_data = cp.deepcopy(ref_data)     
        real_data[start_index:start_index+N_I] = cp.deepcopy(deformed_data)     
        real_data_list.append(real_data)   

    # start_index_list = [40, 50, 60]  
    # label_list = [r'$N_I=40$', r'$N_I=50$', r'$N_I=60$']   
    # for N_I in start_index_list:    
    #     partial_ref_data = cp.deepcopy(ref_data[start_index:start_index + N_I, :])    
    #     inter_force = np.tile(np.array([0, -1, 1]), (N_I, 1))    
    #     deformed_data, G = partial_deform_trajectory(
    #         ref_data=partial_ref_data,    
    #         inter_force=inter_force,    
    #         beta=0.9    
    #     )   
    #     real_data = cp.deepcopy(ref_data)     
    #     real_data[start_index:start_index+N_I] = cp.deepcopy(deformed_data)     
    #     real_data_list.append(real_data)     
    

    # plot_epi_data(font_name=args.data_name, nb_samples=3, nb_data=200,     
    #               real_data=real_data_list, ref_data=ref_data,   
    #               label_list=label_list,   
    #               start=start, end=end,   
    #               start_index=start_index, end_index=start_index+N_I,     
    #               obs_center=obs_center,   
    #               save_fig=args.save_fig, fig_path=args.fig_path,   
    #               )   
    
    ref_data = ref_data.T   
    print("ref_data :", ref_data.shape)    
    # index_list = [0, start_index, (start_index+end_index)//2, end_index, 199]   
    # index_list = [0, start_index, start_index+15, end_index, 199]   
    index_list = [0, start_index, start_index+3,  (start_index+end_index)//2, end_index-3, end_index, 199]      
    # index_list = [start_index, (start_index+end_index)//2, end_index]   
    print("index_list :", index_list, args.nb_data, T)    
    viaNum = len(index_list)      
    viaFlag = np.ones(viaNum)       
    via_time = np.zeros(viaNum)       
    via_points = np.zeros((viaNum, args.output_dim))       
    deform_ref_data = real_data_list[2].T      
    via_var_list = []    
    for i, index in enumerate(index_list):       
        via_time[i] = index/args.nb_data * T      
        # via_time[i] = (index - 50)/args.nb_data * T     
        # via_points[i, :] = ref_data[:2, index]  
        via_points[i, :] = deform_ref_data[:2, index]    
        # via_points[i, :] = np.array([ref_data[1, index], ref_data[0, index]])       
        if i == 3: 
            via_var = 0.003 * np.eye(args.output_dim)   
            via_var_list.append(via_var)    
        else: 
            via_var = 1E-4 * np.eye(args.output_dim)     
            via_var_list.append(via_var)   

        # via_time[0] = 1.   
        # via_points[0, :] = np.array([0.0, 24.0])      
        # via_var = 1E-6 * np.eye(output_dim)      
    print("time :", via_time, "via_points :", via_points)     

    # ///////////////////////////// cartesian //////////////////////////////////////    
    index_num = args.nb_samples * args.nb_data     
    resample_index = int(trajectory_kinematics_t_list.shape[0]/index_num)       
    start_1 = 1      
    start_2 = 13       
    resampled_theta_t = trajectory_kinematics_t_list[::resample_index, start_1:start_1+3]   
    demos = resampled_theta_t[:index_num, :]     
    
    # # ///////////////////////////// cartesian ////////////////////////////////////// 
    # mu_gmr, ref_mu, _, _ = cal_evaluation_cartesian(
    #     joint_data=demos,      
    #     ref_data=ref_data,   
    #     args=args   
    # )   

    # #######################################################
    mean_value, std_value = cal_iteration_cartesian(
        joint_data=demos,   
        ref_data=ref_data,   
        obs_center=obs_center,   
        viaNum=viaNum,   
        viaFlag=viaFlag,   
        via_time=via_time,   
        via_points=via_points,   
        via_var_list=via_var_list,     
        real_data=real_data_list,   
        start=start, end=end,  
        start_index=start_index, end_index=end_index, args=args   
    )   

    # action_space_scaling(R=3, alpha=0.2, fig_path='./wrist_paper/figures/task_aan/task_performance_index.pdf', save_fig=True)   
 
    comparison_learning(R=3, alpha=0.3, fig_path='./wrist_paper/figures/tase/task_performance_index.pdf', save_fig=False)  

    # ========================== Old Code ============================
    # ref_data = joint_data[::resample_index, start_1:start_1+3]       
    # ref_data = 
    # print("ref_data :", ref_data)     
    # ori_stiff_data = ori_stiff_data[::resample_index, :]       
    # ori_stiff_data = ori_stiff_data[:nb_data, :]       
    # ref_data = ref_data[:nb_data, :]    

    # x_c = -1 * real_list_n[center_index, 0]
    # y_c = -1 * real_list_n[center_index, 1] 
    # f_xc = -1 * real_list_n[center_index, 0]/np.sqrt(real_list_n[center_index, 0]**2 + real_list_n[center_index, 1]**2)   
    # f_yc = -1 * real_list_n[center_index, 1]/np.sqrt(real_list_n[center_index, 0]**2 + real_list_n[center_index, 1]**2)   
    # # force_xy = np.array([f_xc, f_yc])    
    # delta_n  = wrist_control.update_single_path(
    #     500,   
    #     0.1,   
    #     np.array([f_xc, f_yc, 0.0]),  
    #     G    
    # )   
    # print("delta_n :", delta_n)  
    # # real_list[start_index:start_index + N, :] = cp.deepcopy(ref_data[:, :]) + delta_q       
    
    # real_list_n[start_index:start_index + N, :] = cp.deepcopy(ref_data_n[:, :]) + delta_n   
    
    # real_list_n[start_index:start_index + N, 2] = -1 * np.sqrt(70.0*70.0 - real_list_n[start_index:start_index + N, 0]*real_list_n[start_index:start_index + N, 0] - real_list_n[start_index:start_index + N, 1] * real_list_n[start_index:start_index + N, 1])      
    # print(real_list_n[start_index:start_index + N, 2])  
    # # # # np.savetxt("data/wrist_demo/demo_data/trajectory_theta_list_real_circ.txt", real_list, fmt='%f', delimiter=',')   
    
    # # plot_theta_real_trajectory(font_name=args.flag, real_data=real_list, ref_data=q_list)      

    # # plot_theta_real_trajectory(font_name=args.flag, real_data=delta_n, ref_data=delta_n)      
    
    # plot_xy_trajectory(font_name="deform_tra", real_data=real_list_n, ref_data=n_list)       
    
    # plot_ee_pose_deform(      
    #     real_list_n,  
    #     n_list,            
    #     flag="_deform_data",   
    #     save_fig=True   
    # )  
    
    # final_list = cal_iterp_data(mu_gmr, N=5000)  
    # print("Final list :", mu_gmr.shape, ref_mu.shape)    
    # ref_final_list = cal_iterp_data(ref_mu.T, N=5000)     
    # print("Final list :", final_list.shape, ref_final_list.shape)      
    # N = 500  
    # ref_final_list = ref_final_list.T  
    # real_list = cp.deepcopy(ref_final_list)  
    # inter_force = np.ones((N, 2))  
    # ref_data = cp.deepcopy(real_list[:N, :2])   
    # deformed_data = partial_deform_trajectory(
    #     ref_data=ref_data,   
    #     inter_force=inter_force,    
    #     beta=0.1   
    # )
    # real_list[:N, :2] = cp.deepcopy(deformed_data) 
    
    # plot_xy_trajectory(font_name="mu_gmr", real_data=real_list, ref_data=ref_final_list)   
    # plot_xy_trajectory(font_name="mu_gmr", real_data=final_list.T, ref_data=ref_final_list.T)      
    
    # xyz_list = np.zeros((5000, 3)) 
    # xyz_list[:, :2] = final_list.T  
    # xyz_list[:, 2] = -1 * np.sqrt(70*70 - xyz_list[:, 0]*xyz_list[:, 0] - xyz_list[:, 1]*xyz_list[:, 1])    
    # # plot_pose_trajectory(font_name="mu_gmr_3d", point_list_eval=xyz_list, point_list=ref_final_list.T)    
    # ref_final_list = ref_final_list.T
    # # print("ref_final :", ref_final_list[0, :], ref_final_list[-1, :])  
    # # ref_final_list = ref_final_list[::-1, :]   
    # # print("ref_final T:", ref_final_list[0, :], ref_final_list[-1, :])   
    
    # spm_fik_left = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)    
    # # psi=-54.73561     
    # psi=0.0     
    # trajectory_theta_list, _, _, _, _, _ = spm_fik_left.cal_theta_list(xyz_list, sample_num=5000, psi=psi)
    # trajectory_ref_theta_list, _, _, _, _, _ = spm_fik_left.cal_theta_list(ref_final_list, sample_num=5000, psi=psi)      
    
    # np.savetxt("data/wrist_demo/demo_data/trajectory_theta_list_ref_circ.txt", trajectory_ref_theta_list, fmt='%f', delimiter=',')  
    # plot_theta_real_trajectory(font_name=args.flag, real_data=trajectory_theta_list, ref_data=trajectory_ref_theta_list)  

    # trajectory_theta_t_list = np.loadtxt("./data/wrist_demo/demo_data/trajectory_theta_list_circ_5.txt",  delimiter=',') 
    # q_list = trajectory_theta_t_list[:5000, :3]   
    # # n_list = trajectory_theta_t_list[:5000, 3:]    
    # print("q_list :", q_list)  


def load_previous_data(args=None):     
    if args.mode==0:   
        control_mode = "zero_force"    
    elif args.mode==1:  
        control_mode = "motion"  
    elif args.mode==2:   
        control_mode= "assistive_force"  
    else:
        control_mode = "zero_force"    

    flag = "_baletral_motion_demo_" + args.flag 

    trajectory_theta_t_list = np.loadtxt(args.root_path + "/" + args.folder + "/wrist_encoder" + flag + ".txt",  delimiter=',', skiprows=1)  
    trajectory_torque_t_list = np.loadtxt(args.root_path + "/" + args.folder + "/wrist_torque" + flag + ".txt",  delimiter=',', skiprows=1)   
    trajectory_kinematics_t_list = np.loadtxt(args.root_path + "/" + args.folder + "/wrist_kinematics" + flag + ".txt",  delimiter=',', skiprows=1)  
    trajectory_iter_t_list = np.loadtxt(args.root_path + "/" + args.folder + "/wrist_iteration" + flag + ".txt",  delimiter=',', skiprows=1)  
    # trajectory_force_t_list = np.loadtxt(args.root_path + "/" + "save_data/" + args.folder + "/wrist_exp_info" + flag + ".txt",  delimiter=',', skiprows=1)

    # if args.game: 	
    #     trajectory_game_t_list = np.loadtxt(args.root_path + "/" + args.folder + "/wrist_game" + flag + ".txt",  delimiter=',', skiprows=1)  
    # fig_path = args.root_path + '/fig/' + control_mode   

    return trajectory_kinematics_t_list[:, 1:3]      


def gmr_mu_sigma(  
    demos_np=None,     
    Xt=None,        
    nb_samples=5,   
    nb_states=5,   
    input_dim=1,   
    output_dim=2    
):  
    in_idx = list(range(input_dim))   
    out_idx = list(range(1, output_dim+1))      
    print("gmr info :", input_dim, output_dim, in_idx, out_idx)    
        
    # GMM + GMR 
    gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim+output_dim, in_idx=in_idx, out_idx=out_idx)  
    gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)  
    gmr_model.gmm_em(demos_np.T)    

    # GMR   
    mu_gmr = []  
    sigma_gmr = []   
    sigma_gmr_1 = []   
    sigma_gmr_2 = []   
    for i in range(Xt.shape[0]):    
        mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])  
        mu_gmr.append(mu_gmr_tmp)  
        sigma_gmr.append(sigma_gmr_tmp)  
        sigma_gmr_1.append(sigma_gmr_tmp[0, 0])   
        sigma_gmr_2.append(sigma_gmr_tmp[1, 1])   
    
    mu_gmr = np.array(mu_gmr)    
    sigma_gmr = np.array(sigma_gmr)    
    return mu_gmr, sigma_gmr, gmr_model      


def kmp_mu_sigma(
    Xt=None,   
    mu_gmr=None,   
    sigma_gmr=None,    
    via_time_list=None,        
    via_index_list=5,  
    via_point_list=None,      
    via_var_list=None,   
    nb_data=200,    
    dt=0.01,     
    lamda_1=0.01,      
    lamda_2=0.6,       
    kh=6,     
    output_dim=2,     
    dim=2        
):  
    # ////////////////////////////////////////////////  
    All_Xt = dt * np.arange(nb_data)[:, None]       
    refTraj = {}    
    refTraj['t'] = All_Xt    
    refTraj['mu'] = cp.deepcopy(mu_gmr)      
    refTraj['sigma'] = cp.deepcopy(sigma_gmr)         

    #  KMP parameters
    # len = Xt.shape[0]   
    len = nb_data     

    # update reference trajectory using desired points
    newRef = refTraj      
    newLen = len      

    # # print("newLen :", newRef.shape, newLen)    
    # for Index, viaIndex in enumerate(via_index_list):      
    #     print("Index :", Index)     
    #     via_var = via_var_list[Index]      
    #     newRef, newLen = kmp_insertPoint(newRef, newLen, via_time_list[Index], via_point_list[Index, :], via_var)      
    # print("newLen :", newRef['t'].shape, newRef['mu'].shape, newRef['sigma'].shape)   
    
    # newRef, newLen = kmp_insertPoint_test(newData=newRef, newNum=newLen, 
    #                      via_time_list=via_time_list, via_point_list=via_point_list, 
    #                      via_var_list=via_var_list, via_index_list=via_index_list  
    #                     )  
    
    via_num = via_time_list.shape[0] 
    insert_value = wrist_control.kmp_insertPoint(
        All_Xt,     
        newRef['mu'],      
        newRef['sigma'].reshape((nb_data * output_dim, output_dim)),           
        via_num,       
        output_dim,    
        via_index_list,         
        via_time_list,          
        via_point_list,           
        via_var_list.reshape((via_num * output_dim, output_dim))       
    )   

    # # newRef['t'] = refTraj['t']   
    # newRef['mu'] = insert_value[0]         
    # newRef['sigma'] = insert_value[1].reshape((nb_data, output_dim, output_dim))      
    # print("newLen :", newLen, newRef['t'].shape, newRef['mu'].shape, newRef['sigma'].shape)       
    sample_t = newRef['t']  
    # sample_mu = newRef['mu']   
    sample_mu = cp.deepcopy(insert_value[0]) 
    sample_sigma = cp.deepcopy(insert_value[1].reshape((nb_data, output_dim, output_dim)))   
    
    # mu_gmr = insert_value[0]                
    # sigma_gmr = insert_value[1].reshape((nb_data, output_dim, output_dim))     

    # Prediction using kmp   
    # Kinv_1, Kinv_2 = kmp_estimateMatrix_mean_var(newRef, newLen, kh, lamda_1, lamda_2, dim, output_dim)  
         
    Kinv_value = wrist_control.kmp_estimateMatrix_mean_var(
        All_Xt, sample_sigma.reshape((nb_data * output_dim, output_dim)), 
        newLen, kh, lamda_1, lamda_2, dim, output_dim
    )   
    Kinv_1, Kinv_2 = Kinv_value[0], Kinv_value[1]  
    print("Kinv_1 :", Kinv_1.shape)   
    
    # # uncertainLen = 0.0 * len     
    # # totalLen = int(len + uncertainLen)  
    # kmp_mu = newRef['mu']     
    # kmp_sigma = newRef['sigma']       
    totalLen = int(Xt.shape[0])    
    
    new_time_t = np.zeros((totalLen, 1))   
    new_mu_t = np.zeros((totalLen, output_dim))    
    new_sigma_t = np.zeros((totalLen, output_dim, output_dim))     
    
    # t, mu, sigma    
    for index in range(totalLen):     
        # t = index * dt   
        t = Xt[index]    
        # kmp_mu, kmp_sigma = kmp_pred_mean_var(t, newRef, newLen, kh, Kinv_1, Kinv_2, lamda_2, dim, output_dim)    
        kmp_mu, kmp_sigma = kmp_pred_mean_var_test(t, sample_t, sample_mu, newLen, kh, Kinv_1, Kinv_2, lamda_2, dim, output_dim)  
        new_time_t[index, 0] = t   
        new_mu_t[index, :] = kmp_mu.T    
        new_sigma_t[index, :, :] = kmp_sigma         

    kmp_mu = new_mu_t     
    kmp_sigma = new_sigma_t      
          
    return kmp_mu, kmp_sigma, sample_mu, sample_sigma        


class cps_training():    
    def __init__(self, omega_dim=None, context_dim=None, R_min=3.0, R_max=5.0, R=22.5, 
                 nb_data=200, N_I=40, dt=0.01, input_dim=1, output_dim=2, dim=1, nb_samples=5, nb_states=10, 
                 via_num=5, lambda_1=1, lambda_2=20, kh=20, 
                 K=100, M=15, num_dofs=3, num=3000  
                ) -> None:   
        self.context_dim = context_dim     
        self.omega_dim = omega_dim     
        self.R_min = R_min      
        self.R_max = R_max     
        # self.context = np.array([0.0, 21.0, 5.0])     
        self.context = np.array([0.5, 1.0])       
        self.param = np.zeros(omega_dim)   
        self.reward = 0.0       
        self.Omega_min = np.array([0.0, 0.0])  # delat t //// beta_mu  
        self.Omega_max = np.array([1.0, 1.0])  
        self.K = K  # maximal number of iteration learning  
        self.k = 0  # 
        self.m = 0 
        self.M = M   
        self.eps = 0.01   
        self.cp = UpperPolicy(self.context_dim)   
        self.mean_list = []   
        self.std_list = []   
        self.beta_1 = 1.0   
        self.beta_2 = 1.0    

        ############### main tasks ######################
        self.Ori_R = R   
        self.ref_data = None   
        self.nb_data = nb_data     
        self.N_I = N_I
        self.nb_samples = nb_samples  
        self.nb_states = nb_states 
        self.dt = dt   
        self.T = self.nb_data * self.dt     
        self.via_num = via_num   
        self.via_point_list = None     
        self.via_time_list = None    
        self.via_index_list = None      
        self.via_var_list = None   
        self.lambda_1 = lambda_1       
        self.lambda_2 = lambda_2             
        self.kh = kh   
        self.input_dim = input_dim      
        self.output_dim = output_dim       
        self.dim = dim    
        self.obs_info = self.set_obs_info(context=self.context)    

        ############ previous data collection ############
        self.dataset = []     
        self.path_ptr = 0      
        # self.Xt = None      
        #### gmr ####   
        self.demos_t = [np.arange(self.nb_data)[:, None] for i in range(self.nb_samples)]          
        self.Xt = self.dt * np.arange(self.nb_data)[:, None]     
        self.mu_gmr = None       
        self.sigma_gmr = None        
        self.gmr_model = None       
        self.mu_kmp = None      
        self.sigma_kmp = None      
        self.real_data = None      
        self.deformed_energy_path = None       
         
        #################### iterative learning ###########
        self.learning_rate = None     
        self.decay_rate = None   
        self.iter_tau_data = None      
        self.iter_stiff_data = None       
        self.iter_damping_data = None        
        
        ################## real-time control ##############
        self.Ts = 0.001    
        self.real_num = int(self.T/self.Ts)     
        self.num_dofs = num_dofs     
        self.num = num     
        # self.main_task(N=self.nb_data)     

    def main_task(self,  
                  N=None,    
                  circle_center = np.array([0.3, 0.3]),   
                  T=0.1,   
                  R=10.0   
                  ):   
        if self.num_dofs==3:  
            ############### Three-Dof wrist robot ############
            phi_range = [-1/2*np.pi, 3/2*np.pi]       
            theta_range = [8/9*np.pi, 8/9*np.pi]        
            sample_num_1 = [1, N]       
            theta = np.linspace(theta_range[0], theta_range[1], sample_num_1[0])        
            phi = np.linspace(phi_range[0], phi_range[1], sample_num_1[1])        
            t, p = np.meshgrid(theta, phi)    
            
            r = 70     
            x = r*np.sin(t)*np.cos(p)       
            y = r*np.sin(t)*np.sin(p)       
            z = r*np.cos(t)      
            
            x = x.flatten()      
            y = y.flatten()     
            z = z.flatten()     

            ref_data = np.vstack((x, y, z))        
            self.ref_data = ref_data[:, ::-1].T    
            # self.ref_data = ref_data.T   
        else:   
            # T = 1.0/omega_t        
            time_list = np.linspace(0.0, T, N)       
            # print(time_list, circle_center)        
            x = circle_center[0] + R * np.cos(2 * np.pi * 1.0/T * time_list)         
            y = circle_center[1] + R * np.sin(2 * np.pi * 1.0/T * time_list)        
            
            ref_data = np.vstack((x, y))   
            self.ref_data = ref_data.T   
            print("ref data shape:", self.ref_data.shape)      
        
        time_list = np.linspace(0.0, T, N)      
        ctl_time_list = np.linspace(0.0, self.T, self.num)      
        self.waypoints = np.vstack([np.interp(ctl_time_list, time_list, x), np.interp(ctl_time_list, time_list, y)]).T      
        print("waypoints :", self.waypoints.shape)    
        return self.ref_data   
        
    def select_context(self, M=1):   
        context_list = []   
        for m in range(M):    
            context_list.append(np.random.random(self.context_dim))   
         
        self.context = context_list[0]    
        print("current context :", self.context)    
        return context_list     

    def set_obs_info(self, context=None):   
        theta = context[0] * 2 * np.pi   
        obs_center = np.array([self.Ori_R * np.cos(theta), self.Ori_R * np.sin(theta)])  
        obs_r = (self.R_max - self.R_min) * context[1] + self.R_min   
        # print(obs_center, obs_r)  
        self.obs_info = np.hstack([obs_center, [obs_r]])  
        return self.obs_info    

    def reward_cal(self, context=None, waypoints=None, force_list=None):  
        reward_1 = 0.0  
        self.obs_info = self.obs_info(context)   
        self.obs_center = self.obs_info[:2]   
        self.obs_r = self.obs_info[2]     
        for i in range(waypoints.shape[0]):    
            waypoint = waypoints[i]  
            if np.linalg.norm(waypoint - self.obs_center) < self.obs_r:  
                reward_1 += -1 * self.beta_1 * np.linalg.norm(waypoint - self.obs_center)
        reward_2 = self.beta_2 * sum(np.abs(force_list[:, :2]))    
        self.reward = reward_1 + reward_2  
        return self.reward     
    
    def select_action(self, context, mean=True):    
        
        # obtain reference trajectory    

        # implementation   

        return self.param    

    def preference_encoding(self, real_path):    
        # previous data  
        index_num = self.nb_samples * self.nb_data       
        # resample_index = int(real_path.shape[0]/index_num)      
        # demos = real_path[::resample_index, :][:index_num, :]     
        # demos = real_path[:index_num, :]   
        demos = real_path      
        self.real_data = demos       
        # print("demos_t :", np.array(demos_t[0]).shape)     

        # Stack time and position data      
        demos_tx = [np.hstack([self.demos_t[i] * self.dt, demos[i * self.nb_data:(i+1) * self.nb_data, 0][:, None], demos[i*self.nb_data:(i+1)*self.nb_data, 1][:, None]]) for i in range(self.nb_samples)]  
        
        # Stack demos    
        demos_np = demos_tx[0]    
        for i in range(1, self.nb_samples):     
            demos_np = np.vstack([demos_np, demos_tx[i]])     
        print("demos_np :", demos_np.shape)     

        self.mu_gmr, self.sigma_gmr, self.gmr_model = gmr_mu_sigma(  
            demos_np=demos_np,   
            Xt=self.Xt,     
            nb_samples=self.nb_samples,         
            nb_states=self.nb_states,     
            input_dim=self.input_dim,    
            output_dim=self.output_dim     
        )   
        return self.mu_gmr, self.sigma_gmr   
    
    def implementation(self, context=None, param=None):   
        traj_num = len(context)
        
        # context ::: generate obs and vr  
        obs_list = []   
        # param ::: beta and delta_t     
        param_list = []       
        param_value = self.param * (self.Omega_max - self.Omega_min) + self.Omega_min    
           
        multi_traj = self.wrist_control(ref_data, context, param_value)  
        index_num = args.nb_data     
        resample_index = int(multi_traj.shape[0]/index_num)             
        single_waypoint = multi_traj[::resample_index, :3]         
        single_force = multi_traj[::resample_index, 3:]        
        
        reward_list = []   
        for index in range(traj_num):     
            reward = self.reward_cal(context=context, waypoints=single_waypoint, force_list=single_force)    
            reward_list.append(reward)    
        
        # store data :: context_list, param_list, reward_list    
        
        return reward   
    
    def energy_partial_deform(self, start_index=None, N_I=None, beta=0.9, inter_force=None):        
        partial_ref_data = cp.deepcopy(self.ref_data[start_index:start_index + N_I, :])    
        # inter_force = np.tile(np.array([0, -1, 1]), (N_I, 1))    
        self.deformed_energy_path, G_m = partial_deform_trajectory(
            ref_data=partial_ref_data,      
            inter_force=inter_force,      
            beta=beta      
        )    
        return self.deformed_energy_path, G_m        

    def trajectory_deform(self, param=None, t_s=None, F_h=None):       
        param_value = param * (self.Omega_max - self.Omega_min) + self.Omega_min    
        delta_t = param_value[0]    ### duration time      
        beta_t = param_value[1]    ### path scale factor  
        print("delta, beta :", delta_t, beta_t)         
        t_e = t_s + delta_t     

        start_index = int(t_s/self.T * self.nb_data)         
        end_index = int(t_e/self.T * self.nb_data)        

        self.via_index_list = np.array([start_index, start_index+1, start_index+(end_index-start_index)//2, end_index-1, end_index])    
        t_v = int((start_index+end_index)/2) * self.dt    
        
        self.via_time_list = np.array([t_s, t_s + self.dt, t_v, t_e-self.dt, t_e])     
        self.via_point_list = np.zeros((self.via_num, 2))    
        self.via_var_list = np.zeros((self.via_num, self.output_dim, self.output_dim))      

        # for via_index in self.via_index_list:    
        for index in range(self.via_num):       
            via_index = self.via_index_list[index]       
            self.via_point_list[index, :self.output_dim] = self.ref_data[via_index, :self.output_dim]       
            via_var = 1E-4 * np.eye(self.output_dim)       
            # self.via_var_list.append(via_var)    
            self.via_var_list[index, :, :] = via_var
            if index == 2:    
                self.via_point_list[index, :self.output_dim] = self.via_point_list[index, :self.output_dim] + beta_t * F_h[via_index, :self.output_dim]  
                via_var = 0.003 * np.eye(self.output_dim)     
                # self.via_var_list[index, :, :] = self.sigma_gmr[via_index, :, :]         
                self.via_var_list[index, :, :] = via_var          
                # print("via var :\n", via_var)         

        #################################################
        # print("via time list:\n", self.via_time_list)     
        # print("via points list:\n", self.via_point_list)        
        # print("via var list:\n", self.via_var_list)       
        
        # #### gmr ####   
        # self.Xt = self.dt * np.arange(self.nb_data)[:, None]     
        # Xt = self.dt * np.arange(start_index, end_index)[:, None]     
        # print("Xt :", Xt)   

        # #### kmp ####    
        # self.mu_kmp, self.sigma_kmp, self.mu_gmr, self.sigma_gmr  = kmp_mu_sigma(
        #     Xt=self.Xt,     
        #     mu_gmr=self.mu_gmr,      
        #     sigma_gmr=self.sigma_gmr,         
        #     via_time_list=self.via_time_list,          
        #     via_index_list=self.via_index_list,          
        #     via_point_list=self.via_point_list,           
        #     via_var_list=self.via_var_list,       
        #     nb_data=self.nb_data,    
        #     dt=self.dt,  
        #     lamda_1=self.lambda_1,      
        #     lamda_2=self.lambda_2,            
        #     kh=self.kh,    
        #     output_dim=self.output_dim,      
        #     dim=self.dim          
        # )   

        return self.mu_kmp, self.sigma_kmp     
    
    def visualization(self, save_fig=False):  
        
        if self.ref_data is None:   
            self.main_task()   

        plot_via_points(  
            ref_data=self.ref_data,     
            real_data=self.real_data,      
            nb_via_points=self.via_num,      
            via_points=self.via_point_list,     
            mu_gmr=self.mu_gmr,     
            sigma_gmr=self.sigma_gmr,    
            mu_kmp=self.mu_kmp,   
            sigma_kmp=self.sigma_kmp,    
            deform_path=self.deformed_energy_path,      
            obs_center=self.obs_info[:2],     
            obs_r=self.obs_info[2],     
            start=None,     
            end=None,     
            font_name='G',    
            save_fig=save_fig,    
            fig_path=None     
        )    
        return 1  
    
    def training(self):    
        
        while self.k < self.K:   
            ####### simulation data generation ###########
            context_list_M = self.select_context(M=self.M)      
            parameter_list_M = self.cp.sample(context_list_M)      
            for m in range(self.M):   
                reward_list_M = 1      

            p_list = computeSampleWeighting(reward_list_M, context_list_M, self.eps)   
            self.cp.update(parameter_list_M, context_list_M, p_list)   

            ######## interact with real environment  ########
            sum_reward = []   
            context_list_L = self.select_context(M=self.L)    
            for l in range(self.L):   
                context = context_list_L[l]     
                param = self.mean(context.reshape(1, -1))     
                reward = self.implementation(param)    
                sum_reward.append(reward)     

            sum_reward = np.array(sum_reward)     
            mean_R = np.mean(sum_reward)     
            std_R = np.std(sum_reward)    
            
            self.mean_list.append(mean_R)    
            self.std_list.append(std_R)   
        return self.mean_list, self.std_list    