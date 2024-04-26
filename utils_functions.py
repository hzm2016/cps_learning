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

# from plot_main import *   
# from plot_main.plot_online_tase_main import *    
# iteration learning functions     
# from path_planning.kmp.demo_GMR import *       

import scipy.io as scio  

import seaborn as sns   
# sns.set(font_scale=1.5)   
np.set_printoptions(precision=4) 

import socket   



def robot_ctl_thread(
    online_tase=None, ctl_para=None, iter_para=None,   
    ref_data=None, mu_gmr_input=None, sigma_gmr_input=None, 
    save_data_list=None, iter_data_list=None,  
    run_code=None, ctl_eve=None, pro_eve=None, iter=None    
):  
    start = time.time()
    for index in range(9):    
        print("========================= Iter ============================:", str(iter))  
        print("mu_gmr :", mu_gmr_input[0, :])    
        run_code = 1 
        if (iter == 2) or (iter == 4):         
            print("here iter :", iter)  
            pro_eve.set()      
            
        back_data = online_tase.control(
            ctl_para,      
            iter_para,       
            ref_data,   
            mu_gmr_input,     
            sigma_gmr_input, 
            iter       
        )   
        
        # save_data = back_data[0][::args.resample, :]    
        # iter_data = back_data[1][::args.resample, :]    
        save_data = back_data[0]    
        iter_data = back_data[1]    
        save_data_list.append(save_data)    
        iter_data_list.append(iter_data)    

        iter += 1
        print("=========================== Done ================================")  
    
    online_tase.moveToZero()   
    end = time.time()   
    np.save('./data/exp_tase/' + flag + '_' + str(iter) + '.npy', np.array(save_data_list))       
    np.save('./data/exp_tase/' + flag + '_' + str(iter) +'_iter.npy', np.array(iter_data_list))               
    print("running time :", end - start)  
    run_code = 0   


def data_process_thread(  
    cps_obs=None, save_data_list=None, iter_data_list=None, 
    mu_gmr_input=None, sigma_gmr_input=None,  
    iter=None, args=None, run_code=None, ctl_eve=None, pro_eve=None  
):  
    t_list = np.linspace(0.0, args.T, int(args.T/args.T_s))      
    t_l_list = np.linspace(0.0, args.T, int(args.nb_data))     
    
    # np.save('./data/exp_tase/' + flag + '_' + str(iter) + '_mu.npy', mu_gmr_input)         
    # np.save('./data/exp_tase/' + flag + '_' + str(iter) + '_sigma.npy', sigma_gmr_input)     
    # print("iter", iter)   
    
    while run_code:       
        # if len(save_data_list) == 5:  
        #     start_time = time.time()  
        #     print("data length :", len(save_data_list), len(iter_data_list))
        # print("data length :", len(save_data_list), len(iter_data_list))  
        # print("data shape :", np.array(save_data_list).shape)   
        pro_eve.wait()   
        if pro_eve.is_set():  
            pro_eve.clear()  
            
        print("data shape :", np.array(save_data_list).shape)  
        # save_data = np.array(save_data_list).reshape(3 * args.num, 14)    
        # save_data = save_data[::args.resample, :]        
        # print(save_data.shape)   
        
        # q_t_2 = save_data[:, 1]   
        # d_q_t_2 = save_data[:, 7]    

        # real_data = np.hstack([q_t_2[:args.nb_data*args.nb_samples, None], d_q_t_2[:args.nb_data*args.nb_samples, None]])      
        
        # mu_gmr, sigma_gmr = cps_obs.preference_encoding(real_data)    
        # f_1 = interpolate.interp1d(t_l_list, mu_gmr[:, 0], kind='linear')    
        # f_2 = interpolate.interp1d(t_l_list, mu_gmr[:, 1], kind='linear')    
        # f_3 = interpolate.interp1d(t_l_list, sigma_gmr[:, 0, 0], kind='linear')    
        # f_4 = interpolate.interp1d(t_l_list, sigma_gmr[:, 1, 1], kind='linear')   
        
        # mu_gmr_input[:, 1] = f_1(t_list)      
        # mu_gmr_input[:, 3] = f_2(t_list)      
        # sigma_gmr_input[:, 1] = f_3(t_list)          
        # sigma_gmr_input[:, 3] = f_4(t_list)       
        
        mu_gmr_input[0, :] = mu_gmr_input[0, :] + np.array([1.0, 1.0, 1.0, 1.0])
        
        print("input :")   
        
        # save_data = np.array(save_data_list).reshape(3 * args.num, 14)    
        
        # # # iter_data_list = np.array(iter_data_list)  
        
        # # # iter_data = back_data[1][::args.resample, :]     
        
        # # # mu_gmr_input = np.zeros((int(args.T/args.T_s), 4))   
        # # # sigma_gmr_input = np.zeros((int(args.T/args.T_s), 4))       
        
        # end_time = time.time()  
        # # print("Process Time :", end_time - start_time)  


def server_test(): 
    Udp_Socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    localaddr = ("192.168.1.108", 9090)
    Udp_Socket.bind(localaddr)
    while True: 
        # send_data  =  input("Input data")
        send_data  =  "Input data"
        if send_data == "EXIT":
            break
        Udp_Socket.sendto(send_data.encode("gbk"), ("192.168.1.106",9090))
        send_data = ""
    Udp_Socket.close()  

    
def client_test(): 
    Udp_Socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    Udp_Socket.bind(("127.0.0.1", 9090)) 
    Recv_Data = Udp_Socket.recvfrom(1024)
    Recv_Msg = Recv_Data[0]
    Sender_Addr = Recv_Data[1]
    print("%s:%s"%(str(Sender_Addr),Recv_Msg.decode("gbk")))
   

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


def cal_iteration_cartesian(joint_data, args=None):     
    phi_range = [0, 2*np.pi]   
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
 
    #  ///////////////////////////////////////////////////     
    nb_data = args.nb_data       
    nb_samples = args.nb_samples       
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    # print("demodura :", demodura)       
    
    # model parameter     
    input_dim = 1       
    output_dim = 2       
    # output_dim = 3        
       
    # index_num = iter * 200     
    # resample_index = int(joint_data.shape[0]/1000)      
    # index_num = iter * 200     
    resample_index = int(joint_data.shape[0]/1000)    

    start_1 = 1      
    start_2 = 13       
    
    resampled_theta_t = joint_data[::resample_index, start_1:start_1+3]   
    demos = resampled_theta_t[:1000, :]     
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
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    # , demos[i*nb_data:(i+1)*nb_data, 2][:, None]   
    # print("demos_tx :", np.array(demos_tx).shape)     

    real_joint_data = [np.hstack([-1 * demos[i*nb_data:(i+1)*nb_data, 0][:, None], -1 * demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_joint_data :", np.array(real_joint_data).shape)   
    
    real_joint_error = real_joint_data - ref_data.T       
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
        output_dim=output_dim,    
        data_name=args.data_name     
    )   
    
    # # // wrist robot    
    # viaNum = 2   
    # viaFlag = np.ones(viaNum)    
    # via_time = np.zeros(viaNum)         
    # via_points = np.zeros((viaNum, output_dim))    
    
    # via_time[0] = dt    
    # via_points[0, :] = np.array([0.0, -23.0])     
    # via_time[1] = 1.    
    # via_points[1, :] = np.array([0.0, 10.0])     
    # # via_points[1, :] = np.array([0.0, 28.0])    
    
    viaNum = 1   
    viaFlag = np.ones(viaNum)    
    via_time = np.zeros(viaNum)         
    via_points = np.zeros((viaNum, output_dim))    

    # via_time[0] = dt    
    # via_points[0, :] = np.array([0.0, -23.0])     
    via_time[0] = 1.    
    # via_points[0, :] = np.array([0.0, 15.0])     
    via_points[0, :] = np.array([0.0, 24.0])    
    
    via_var = 1E-6 * np.eye(output_dim)    
    # # via_var = 1E-6 * np.eye(4)  
    # # via_var[2, 2] = 1000   
    # # via_var[3, 3] = 1000   
    # ///////////////////////////////////////////
           
    ori_refTraj, refTraj, kmpPredTraj = KMP_pred(
        Xt=Xt,  
        mu_gmr=mu_gmr,   
        sigma_gmr=sigma_gmr,   
        viaNum=viaNum,    
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
        data_name=args.data_name       
    )   
    
    plot_via_points(  
        font_name=args.data_name + '_' + str(args.iter),    
        nb_posterior_samples=viaNum,    
        via_points=via_points,    
        mu_gmr=ori_refTraj['mu'],    
        pred_gmr=kmpPredTraj['mu'],    
        sigma_gmr=ori_refTraj['sigma'],    
        sigma_kmp=kmpPredTraj['sigma']    
    )   
     
    # plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    # plot_GMM_raw_data(font_name=args.data_name + '_' + str(args.iter), nb_samples=5, nb_data=200, Y=Y, gmr_model=gmr_model)   
    # plot_mean_var(font_name=args.data_name + '_' + str(args.iter), nb_samples=args.nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)
    # plot_mean_var_error(font_name=args.data_name, real_data=real_joint_data, ref_data=ref_data.T)  
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


def cal_evaluation_cartesian(joint_data, args=None):     
    # ///////////////////////// ref data //////////////////////
    phi_range = [0, 2*np.pi]   
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
 
    #  ////////////////////// sample data /////////////////////  
    nb_data = args.nb_data       
    nb_samples = args.nb_samples       
    
    nb_data_sup = 0      
    dt = 0.01     
    demodura = dt * nb_data      
    # print("demodura :", demodura)       
    
    # model parameter     
    input_dim = 1       
    output_dim = 2       
    # output_dim = 3        
       
    # index_num = iter * 200     
    # resample_index = int(joint_data.shape[0]/1000)      
    # index_num = iter * 200     
    resample_index = int(joint_data.shape[0]/1000)    
    start_1 = 1      
    start_2 = 13       
    resampled_theta_t = joint_data[::resample_index, start_1:start_1+3]   
    demos = resampled_theta_t[:1000, :]     
    
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
    demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    #  demos_tx = [np.hstack([demos_t[i] * dt, demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 0][:, None]]) for i in range(nb_samples)]
    # , demos[i*nb_data:(i+1)*nb_data, 2][:, None]     
    # print("demos_tx :", np.array(demos_tx).shape)     

    real_joint_data = [np.hstack([-1 * demos[i*nb_data:(i+1)*nb_data, 0][:, None], -1 * demos[i*nb_data:(i+1)*nb_data, 1][:, None], demos[i*nb_data:(i+1)*nb_data, 2][:, None]]) for i in range(nb_samples)]
    # print("real_joint_data :", np.array(real_joint_data).shape)   
    
    real_joint_error = real_joint_data - ref_data.T       
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
        output_dim=output_dim,    
        data_name=args.data_name     
    )   
    
    # plot_raw_data(font_name=args.data_name, nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=X, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)    
    # plot_GMM_raw_data(font_name=args.data_name + '_' + str(args.iter), nb_samples=5, nb_data=200, Y=Y, gmr_model=gmr_model)   
    plot_mean_var(font_name=args.data_name + '_' + str(args.iter), nb_samples=args.nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr, ref_data=ref_data)
    # plot_mean_var_error(font_name=args.data_name, real_data=real_joint_data, ref_data=ref_data.T)  
    return mu_gmr, ref_data, np.mean(real_joint_error**2), np.std(np.array(real_joint_error))    


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
        
        mean_value, std_value = cal_iteration_cartesian(trajectory_kinematics_t_list, args=args)  


def evaluate_data(args=None, data_name_list = ["eva_end"]):   
    root_path = "./data/aim_data/Subject_" + str(args.subject_index) + "/method_" + str(args.method_index) + "/" + args.training_mode 
    for data_name in data_name_list: 
        file_path_1 = root_path + "/reward_list_" + data_name + ".npy"  
        file_path_2 = root_path + "/training_context_" + data_name + ".npz"   
        file_path_3 = root_path + "/state_list_" + data_name + ".npy"  
        file_path_4 = root_path + "/traj_" + data_name + ".npz"  
    
        reward_info = np.load(file_path_1)      
        initial_training_data = np.load(file_path_2)     
        # state_info = np.load(file_path_3)     
        traj_results = np.load(file_path_4)      
        print(reward_info) 
        num_episodes = reward_info.shape[0]   
        context_info = initial_training_data['arr_0']      
        para_info = initial_training_data['arr_1']     
        obs_info = initial_training_data['arr_2']    
        # state_info = initial_training_data['arr_3']   
        center_info = initial_training_data['arr_4']    
        angle_info = initial_training_data['arr_5'].reshape(-1, 1)       
        # print("================ saved info ===============")   
        # print("reward info :\n", np.mean(reward_info, axis=0), np.std(reward_info, axis=0))    
        # print("success info :\n", success_info)         
        # print("force info :\n", force_info)      

        # # ############ cal reward ###########     
        traj_info, range_info = traj_results["arr_0"], traj_results["arr_1"]      
        real_waypoints = traj_info[::args.resample, :2]        
        ref_waypoints = traj_info[::args.resample, 3:5]        
        real_waypoints = real_waypoints.reshape((num_episodes, args.nb_data, 2))       
        ref_waypoints = ref_waypoints.reshape((num_episodes, args.nb_data, 2))     
        force_waypoints = traj_info[::args.resample, 9].reshape((num_episodes, args.nb_data, 1))                 

        reward_info, success_info, force_info = get_reward_list(
            args=args,   
            waypoints_list=ref_waypoints,   
            force_waypoints_list=force_waypoints,    
            obs_info=obs_info,  
            range_list=range_info,   
            center_list=center_info   
        )   
        print("============== evaluation info ============")   
        print("reward info :\n", np.mean(reward_info, axis=0), np.std(reward_info, axis=0))     
        print("success info :\n", np.sum(np.array(success_info))/np.array(success_info).shape[0])         
        print("force info :\n", np.mean(force_info, axis=0), np.std(force_info, axis=0))    
          

def get_preference_data(args=None, ref_data=None, tele_exp=None):      
    context_info = np.tile(np.array([0.0,0.0,5.0]), (args.num_pre, 1))     
    para_info = np.tile(np.array([1.0, 1.0]), (args.num_pre, 1))   
    desire_start_list = 200 * np.ones(args.num_pre)     
    args.traj_deform = 0    
    args.policy_learn = 0    
    # args.data_name = "first_collect_low_stiff"  
    print("ref_data :", ref_data.shape)      
    
    # ///////////////////// tele rehabilitation ////////////////
    hardware_setting = np.array([args.tele_master, args.tele_slave, args.master_version, args.slave_version]) 
        
    # //// sea_ctl_cmd, iter_cmd, record_force, iter_num, use_pred or not      
    initilize_setting = np.array([args.sea_ctl_cmd, args.iter_cmd, args.ee_force, args.iter_num, args.use_pred, args.traj_deform, args.policy_learn])          
    
    # //// use_vr // game_index, 1, reach target// motion_mode: 1, mirror, 0 /// target_mode // target_index   
    game_setting = np.array([args.use_vr, 2.0, 1.0, 1.0, 0.0])         
    vr_info = np.array([25, 12.0, 0.0, -10.0, 3.0])           
    
    tele_exp.begin(
        hardware_setting,  
        save_root_path,       
        flag,       
        para_input,       
        initilize_setting,             
        game_setting,        
        vr_info         
    )   
    
    ############### get data for reference learning ################
    if args.move_target:          
        tele_exp.moveToZero(
            ref_data[0, :3],     
            args.speed,   
            args.delay_time    
        )   
    
    
    if args.run_ctl:  
        K_p_re = np.array([1.0, 1.0, 1.0])      
        K_d_re = 0.01 * K_p_re     
        results = tele_exp.control(
            ref_data,   
            context_info,       
            para_info,   
            desire_start_list,  
            K_p_re,   
            K_d_re,   
            5000,    
            2000,    
            0.4, 
            20, 
            args.method_index, 
            1     
        )   
        
        traj_info, range_info = results[0], results[1]       
    
        # # ########## cal reward ##########        
        np.save("./data/aim_data/Subject_" + str(args.subject_index) + "/traj_reference_" + args.data_name + ".npy", traj_info)             
        # # ########## cal reward ##########   
    
    if args.move_zero:   
        tele_exp.moveToZero(
            np.array([-54.735610,-54.735610,-54.735610]),   
            args.speed,   
            args.delay_time   
        )   
    

def upper_confidence_bound(x, gp_model, beta, state_dim=4):   
    y_pred, y_std = gp_model.predict(x.reshape(-1, state_dim), return_std=True)
    print("y_pred :", y_pred, y_std)  
    ucb = y_pred + beta * y_std    
    ### output y predict ###    
    return ucb    


def obtain_ctl_para(num_dofs=None):      
    K_p = np.array([15.0, 15.0])    
    K_d = np.array([0.1, 0.1])    
    K_v = np.array([0.002, 0.002])           
    ctl_para = np.zeros((3, num_dofs))        
    ctl_para[0, :] = cp.deepcopy(K_p)        
    ctl_para[1, :] = cp.deepcopy(K_d)     
    ctl_para[2, :] = cp.deepcopy(K_v)     
    
    iter_para = np.zeros((4, 3))       
    iter_para[0, :] = np.array([5.0, 5.0, 0.0])                    # alpha_q      
    iter_para[1, :] = np.array([1.5, 3.5, 0.0]) * angle_to_radian   # learning rate tau    
    iter_para[2, :] = np.array([0.25, 0.8, 0.0]) * angle_to_radian   # learning rate stiff      
    iter_para[3, :] = np.array([0.03, 0.03, 0.0]) * angle_to_radian   # learning rate damping    
    
    return ctl_para, iter_para     


def obtain_para_input(args):   
    # args.ctl_mode = "zero_force"   
    ones = np.array([1.0, 1.0, 1.0])     
    fric_1 = 0.8 * ones;     
    fric_2 = 2.5 * ones;     
    if args.ctl_mode == "zero" :   
        # K_p = np.array([1850.0, 1850.0, 1850.0])           
        # K_d = np.array([85.5, 85.5, 85.5])       
        # K_p = np.array([205.0,205.0,205.0])           
        # K_d = np.array([5.55, 5.55, 5.55])     
        K_p = 2250.0 * ones            
        K_d = 2 * np.sqrt(K_p)         
        # K_p = 850.0 * ones            
        # K_d = 55.0 * ones       
        # K_p = 1550.0 * ones            
        # K_d = 50.0 * ones       
        # K_p = 2250.0 * ones            
        # K_d = 105.5 * ones       
        # K_i = np.array([0.0,0.0,0.0])      
        # K_v = np.array([0.008,0.008,0.008])        
        # K_p_theta = np.array([10,10,10])        
        # K_d_theta = K_p_theta * 1.0/70.0        
        # K_p = np.array([1100.0, 1100.0, 1100.0])         
        # K_d = np.array([20.5, 20.5, 20.5])         
        K_i = np.array([0.0,0.0,0.0])      
        K_v = np.array([0.002,0.002,0.002])        
        # K_p_theta = np.array([10,10,10])        
        # K_d_theta = K_p_theta * 1.0/70.0     
        
        K_p_theta = np.array([55, 55, 55])    
        K_d_theta = K_p_theta * 1.0/65.0        
        
        # # ///// friction calculation motion 
        fric_1 = 0.8 * ones;     
        fric_2 = 1.0 * ones;     
    elif args.ctl_mode == "motion":     
        # K_p = np.array([45,45,45])     
        # K_d = K_p * 1.0/50.0     
        K_p = np.array([95, 95, 95])       
        K_d = K_p * 1.0/60.0       
        # K_p = np.array([30,30,30])         
        # K_d = K_p * 1.0/50.0       
        # K_p = np.array([20,20,20])      
        # K_d = K_p * 1.0/50.0      
        # K_d = np.array([0.06,0.06,0.06])      
        K_i = np.array([0.0,0.0,0.0])     
        # K_v = np.array([0.005,0.005,0.005])      
        K_v = 0.002 * ones   
        # /////// parameters    
        # K_p = np.array([0.50,0.50,0.50])     
        # K_d = np.array([0.01,0.01,0.01])     
        # K_i = np.array([0.0,0.0,0.0])    
        # K_v = np.array([0.01,0.01,0.01])    
        # K_p_theta = np.array([10,10,10])      
        # K_d_theta = K_p_theta * 1.0/70.0       
        # K_p_theta = np.array([80,80,80])      
        # K_d_theta = K_p_theta * 1.0/220.0       
        # K_p_theta = 140 * ones  
        # K_d_theta = K_p_theta * 1.0/700.0   
        # K_p_theta = np.array([250, 250, 250])
        # K_d_theta = K_p_theta * 1.0/600.0   
        # blue spring   
        K_p_theta = np.array([100, 100, 100])     
        # K_p_theta = np.array([120, 120, 120])            
        # K_p_theta = np.array([85, 85, 85])            
        # K_p_theta = np.array([20, 20, 20])            
        # K_d_theta = K_p_theta * 1.0/50.0      
        K_d_theta = np.sqrt(K_p_theta) * 0.4    
        # K_d_theta = K_p_theta * 1.0/55.0     
        # K_d_theta = K_p_theta * 1.0/35.0     
        # K_p_theta = np.array([95, 95, 95])               
        # K_d_theta = K_p_theta * 1.0/65.0       
        # ///////////////////////// normal sea control    
        # // K_p_test<<30,30,30;     
        # // K_d_test<<1.0,1.0,1.0;    
        # // K_v<<0.012,0.012,0.012;     
        # // K_p_test<<35,35,35;     
        # // K_d_test<<1.2,1.2,1.2;      
        # // K_v<<0.012,0.012,0.012;     
        # ///// friction calculation force   
        fric_1 = 0.04 * ones;     
        fric_2 = 2.5 * ones;    
    elif args.ctl_mode == "assistive":   
        # K_p = np.array([3000.0,2200.0,2200.0])     
        K_p = np.array([1800.0,1800.0,1800.0])    
        # K_p = np.array([1000.0,1000.0,1000.0])    
        # K_d = K_p * 1.0/100.0   
        K_d = K_p * 1.0/25.0        
        # K_p = np.array([500.0,500.0,500.0])     
        # K_d = K_p * 1.0/60.0   
        # K_d = K_p * 1.0/30.0       
        # K_d = np.array([0.2,0.2,0.2])        
        K_i = np.array([0.000,0.000,0.000])       
        K_v = np.array([0.008,0.008,0.008])       
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0       
        fric_1 = 0.8 * ones;      
        fric_2 = 1.0 * ones;       
    elif args.ctl_mode == "force_fre":       
        K_p = np.array([2200.0,2200.0,2200.0])     
        K_d = K_p * 1.0/40.0   
        # K_d = K_p * 1.0/25.0        
        # K_p = np.array([900.0,900.0,900.0])       
        # # K_d = K_p * 1.0/60.0     
        # K_d = K_p * 1.0/30.0        
        # K_d = np.array([0.2,0.2,0.2])        
        K_i = np.array([0.000,0.000,0.000])       
        K_v = np.array([0.008,0.008,0.008])       
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0       
    else:    
        K_p = np.array([45.0,45.0,45.0])      
        K_d = np.array([0.55,0.55,0.55])        
        K_i = np.array([0.0,0.0,0.0])     
        K_v = np.array([0.008,0.008,0.008])      
        K_p_theta = np.array([10,10,10])      
        K_d_theta = K_p_theta * 1.0/70.0              
    para_input = np.zeros((8, 3))     
    para_input[0, :] = cp.deepcopy(K_p)     
    para_input[1, :] = cp.deepcopy(K_d)     
    para_input[2, :] = cp.deepcopy(K_i)     
    para_input[3, :] = cp.deepcopy(K_v)     
    para_input[4, :] = cp.deepcopy(K_p_theta)       
    para_input[5, :] = cp.deepcopy(K_d_theta)       
    para_input[6, :] = cp.deepcopy(fric_1)       
    para_input[7, :] = cp.deepcopy(fric_2)  
    
    return para_input


def get_online_gmr_update():    
    # ############################################################################
    # tele_exp.gmr_begin(
    #     demos_np.T,      
    #     args.nb_data,      
    #     args.nb_samples,        
    #     args.nb_states,       
    #     args.nb_dim,       
    #     args.max_online_data,       
    #     args.order,     
    #     in_idx,      
    #     out_idx       
    # )   
    
    # for i in range(2):    
    #     # print("state :", demos_np[i, :12].shape, demos_np[i, :].shape)  
    #     # tele_exp.gmr_online_predict(
    #     #     demos_np[i, :12],
    #     #     demos_np[i, :]    
    #     #     )       
    #     print("index :", i, demos_np[i, :][None, :].shape)      
    #     tele_exp.gmr_online_update(demos_np[i, :][None, :])     

    # tele_exp.gmr_offline_learning(
    #     100, 
    #     1e-5  
    # )   

    # # T = 5.0    
    # # time_list = np.linspace(0.0, T, args.nb_data)[:, None]    
    # mu_gmr = []         
    # sigma_gmr = []                         
    # for i in range(args.nb_data):          
    #     # online_gmr.update_dataset(demos_np_online[i, :][None, :])        
    #     # print(demos_np_online[150 + 1, :][None, :].shape)         
    #     # online_gmr.online_update(demos_np_online[i, :][None, :], 30)        
    #     # mu_gmr_tmp = tele_exp.gmr_online_predict(time_list[i])      
    #     mu_gmr_tmp = tele_exp.gmr_online_predict(demos_np[i, :args.input_dim])      
    #     mu_gmr.append(mu_gmr_tmp)      
    #     sigma_gmr_tmp = np.zeros((args.output_dim, args.output_dim))     
    #     sigma_gmr.append(sigma_gmr_tmp)     
    
    # mu_gmr = np.array(mu_gmr)    
    # sigma_gmr = np.array(sigma_gmr)     
    # print("mu_gmr :", mu_gmr.shape, sigma_gmr.shape)       
    # # from paper_fig_main import *  
    # # from paper_tele_fig_main import *  
    # # # plot_paper_tele_tracking_list(
    # # #     exp_info_list[0],   
    # # #     theta_list[0],       
    # # #     flag=args.flag,    
    # # #     save_fig=True,     
    # # #     save_root=''    
    # # # )   
    
    # plot_multiple_motion(  
    #     # demos_np[:args.nb_data, 1:],        
    #     demos_np[:args.nb_data, args.input_dim:],        
    #     pred_motion=mu_gmr,          
    #     mu_gmr=mu_gmr,         
    #     sigma_gmr=sigma_gmr,          
    #     save_fig=True,        
    #     flag="tele_validation_offline"          
    # )   
    return None  
    
    
def get_gmr_training_data(args=None, training_data=None):     
    in_idx = np.array(range(args.input_dim))       
    out_idx = np.array(range(args.input_dim, args.input_dim + args.output_dim))       
    args.nb_dim = args.input_dim + args.output_dim       
    print("in_idx :", in_idx)      
    print("out_idx :", out_idx)      
    
    time_list = np.linspace(0.0, args.T, args.nb_data)       
    demos_t = [time_list[:, None] for i in range(args.nb_samples)]      
    d_t = np.array(demos_t).reshape(args.nb_data * args.nb_samples)         
    print("demos_t :", d_t[:, None].shape)    
        
    demos_np = np.hstack((d_t[:, None], training_data))   
    print("demos_np :", demos_np.shape)      
    
    return demos_np, in_idx, out_idx     


def get_gmr_update(args=None, tele_exp=None, traj_data=None):                
    demos_np, in_idx, out_idx = get_gmr_training_data(args=args, training_data=traj_data)          
    
    tele_exp.gmr_begin(
        demos_np.T,      
        args.nb_data,      
        args.nb_samples,        
        args.nb_states,       
        args.nb_dim,       
        args.max_online_data,       
        args.order,     
        in_idx,      
        out_idx       
    )   
    
    start = time.time()    
    sample_time_test = np.linspace(0.0, args.T, args.nb_data)[:, None]     
    print("sample_time_test :", sample_time_test.shape)  
    back_gmr = tele_exp.gmr_update(
        demos_np.T,           
        args.max_iter,       
        sample_time_test          
    )   
    end = time.time()    
    print("delta time :", end - start)    

    pred_mu_gmr, pred_sigma_gmr = np.array(back_gmr[0]), np.array(back_gmr[1])    
    pred_sigma_gmr = np.reshape(pred_sigma_gmr, (pred_mu_gmr.shape[0], args.output_dim, args.output_dim))  
    print("mu_gmr :", pred_mu_gmr.shape)     
    print("sigma_gmr :", pred_sigma_gmr.shape)    
    
    return pred_mu_gmr, pred_sigma_gmr * 2       


def get_kmp_partial_update(args=None, tele_exp=None, mu_gmr=None, sigma_gmr=None, 
                           F_h=None, t_s=None, delta_t=None, beta_t=None
                        ):               
    start = time.time()   
    sigma_gmr = sigma_gmr.reshape((args.nb_data * args.output_dim, args.output_dim))    
    tele_exp.preference_update(
        mu_gmr, 
        sigma_gmr  
    )   
    
    sample_time_test = np.linspace(t_s, t_s + delta_t, int(delta_t/args.T*args.nb_data))[:, None]     
    mu_kmp, sigma_kmp, via_points = tele_exp.kmp_update(
        F_h, 
        delta_t,  
        beta_t,   
        t_s,  
        1,   
        sample_time_test, 
        10.0   
    )   
    sigma_kmp = sigma_kmp.reshape((-1, args.output_dim, args.output_dim))      
    
    end = time.time()    
    print("delta time :", end - start)    
    
    plot_figure_all_info(
        args=args,    
        x_e=None,   
        y_e=None,      
        x_t=traj_data[:, 0],   
        y_t=traj_data[:, 1],           
        range_list=None,  
        ee_force_t=None,    
        obs_info=None,   
        mu_gmr=mu_gmr, 
        sigma_gmr=sigma_gmr.reshape((args.nb_data, args.output_dim, args.output_dim)),   
        mu_kmp=mu_kmp,     
        sigma_kmp=sigma_kmp,           
        via_points=via_points,      
        fig_name='actual_kmp_latest'               
    )   
        
    return mu_kmp, sigma_kmp, via_points    


def get_kmp_update(args=None, tele_exp=None, mu_gmr=None, sigma_gmr=None, 
                   ref_data=None, center_index=100, beta_t=2.0):                   
    start = time.time()    
    # sample_time_test = np.linspace(0.0, args.T, args.num)[:, None]     
    # print(sample_time_test.shape)   
    # back_gmr = tele_exp.gmr_update(
    #     demos_np.T,           
    #     args.max_iter,         
    #     sample_time_test          
    # )   
    # print("sigma_gmr :", sigma_gmr.shape)    
    
    sigma_gmr = sigma_gmr.reshape((args.nb_data * args.output_dim, args.output_dim))    
    tele_exp.preference_update(
        mu_gmr, 
        sigma_gmr  
    )   
    
    # F_h = np.array([-5.0, 0.0])       
    # delta_t = args.T * 0.2        
    # beta_t = 2.0      
    # t_s = 100/args.nb_data * args.T     
    
    angle_list = np.linspace(-np.pi, np.pi, args.nb_data)
    # print("angle_list :", angle_list)      
    angle_center = angle_list[args.center_index]         
    F_bar = 10   
    F_t = -1 * np.array([np.cos(angle_center) * F_bar, np.sin(angle_center) * F_bar])   
    print("angle_center :", angle_center, "F_t :", F_t)     
    t_s = (args.center_index - args.N_I//2)/args.nb_data * args.T  
    delta_t = args.N_I/args.nb_data * args.T   
    
    # sample_time_test = np.linspace(t_s, t_s + delta_t, int(delta_t/args.T*args.nb_data))[:, None]   
    sample_time_test = np.linspace(0.0, args.T, args.nb_data)[:, None]     
    mu_kmp, sigma_kmp, via_points = tele_exp.kmp_update(
        F_t, 
        delta_t,    
        beta_t,   
        t_s,  
        0,   
        sample_time_test    
    )   
    sigma_kmp = sigma_kmp.reshape((-1, args.output_dim, args.output_dim))      
    
    end = time.time()    
    print("delta time :", end - start)    
    return mu_kmp, sigma_kmp, via_points      
    
    
def data_processing(training_data, mode="offline", args=None):   
    demos_np = None     
    in_idx = None     
    out_idx = None     
    args.nb_samples = 5      
    # args.nb_data = int(training_data.shape[0]/args.nb_samples)      
    # # /////////// time-driven //////////////
    # args.input_dim = 1   
    # args.output_dim = 3 
    # # args.input_dim = int(args.order * args.output_dim)   
    # args.nb_dim = args.input_dim + args.output_dim    
    # T = 5.0    
    # time_list = np.linspace(0.0, T, args.nb_data)     
    # demos_t = [time_list[:, None] for i in range(args.nb_samples)]      
    # d_t = np.array(demos_t).reshape(args.nb_data * args.nb_samples)       
    # print("demos_t :", d_t[:, None].shape)       
    # # Stack time and position data       
    # # demos_tx = [np.hstack([self.demos_t[i] * self.dt, demos[i * nb_data:(i+1) * nb_data, 0][:, None], demos[i * nb_data : (i+1) * nb_data, 1][:, None]]) for i in range(nb_samples)]  
    # # # Stack demos    
    # # demos_np = demos_tx[0]    
    # # for i in range(1, nb_samples):         
    # demos_np = np.hstack((d_t[:, None], training_data))      
    # print("demos_np :", demos_np.shape)      
    # in_idx = np.array(range(args.input_dim))         
    # out_idx = np.array(range(args.input_dim, args.input_dim + args.output_dim))      
    # print("in_idx :", in_idx, "out_idx :", out_idx)         
    # print("nb_samples :", args.nb_samples, "nb_data :", args.nb_data, "nb_dim", args.nb_dim)     
    if mode == "offline":   
        args.nb_data = int(training_data.shape[0]/args.nb_samples)   
        # print("demos_t :", np.array(demos_t[0]).shape)      
        # /////////// time-driven //////////////  
        args.input_dim = 1   
        # args.output_dim = 3    
        args.nb_dim = args.input_dim + args.output_dim    
        
        T = 5.0   
        time_list = np.linspace(0.0, T, args.nb_data)     
        demos_t = [time_list[:, None] for i in range(args.nb_samples)]      
        d_t = np.array(demos_t).reshape(args.nb_data * args.nb_samples)        
        print("demos_t :", d_t[:, None].shape)    
            
        demos_np = np.hstack((d_t[:, None], training_data))       
    else:   
        # /////////// time-driven //////////////  
        args.input_dim = args.order * args.output_dim    
        # args.output_dim = 3   
        # args.input_dim = int(args.order * args.output_dim)     
        args.nb_dim = args.input_dim + args.output_dim    

        input_dataset = []    
        output_dataset = []    
        for i in range(args.order, training_data.shape[0]):     
            input_state = training_data[i-args.order:i, :].reshape(args.input_dim)    
            output_state = training_data[i, :]      
            input_dataset.append(input_state)        
            output_dataset.append(output_state)    
        
        # print(np.array(input_dataset).shape, np.array(output_dataset).shape)  
        demos_np = np.hstack((np.array(input_dataset), np.array(output_dataset)))     
        print("dataset :", demos_np.shape)        

        # training_data = demos_np[:training_index, :]     
        # test_data = demos_np[training_index:, :input_dim]     
        # true_data = demos_np[training_index:, input_dim:]     
        # print("data seperate :", training_data.shape, test_data.shape, true_data.shape)           
        # print("demos_t :", np.array(demos_t[0]).shape)      
        # T = 5.0   
        # time_list = np.linspace(0.0, T, args.nb_data)     
        # demos_t = [time_list[:, None] for i in range(args.nb_samples)]      
        # d_t = np.array(demos_t).reshape(args.nb_data * args.nb_samples)       
        # print("demos_t :", d_t[:, None].shape)     
        # demos_np = np.hstack((d_t[:, None], training_data))      
        # print("demos_np :", demos_np.shape)    

    print("demos_np :", demos_np.shape)     
    args.nb_data = int(training_data.shape[0]/args.nb_samples)      
    in_idx = np.array(range(args.input_dim))     
    out_idx = np.array(range(args.input_dim, args.input_dim + args.output_dim))       
    print("nb_dim", args.nb_dim, "in_idx :", in_idx, "out_idx :", out_idx)     
    print("nb_samples :", args.nb_samples, "nb_data :", args.nb_data)       
    return demos_np, in_idx, out_idx, args    



if __name__ == "__main__":  
    
    for iter in range(1):              
        # args.data_name = "iteration_learning_tro_" + str(iter)   
        
        # ///////////////// reference path ////////////////////   
        file_path = args.root_path + '/demo_data/trajectory_theta_list_' + args.file_name + '.txt'       
        ref_data = np.loadtxt(file_path, dtype=float, delimiter=',')        
        initial_joint = ref_data[0, :]       
        # initial_joint = np.array([-54.73, -54.73, -54.73])         
        # print("initial_joint :", initial_joint)          
        
        # /////////////////////// iteration ////////////////////
        stiff_data = np.zeros((args.num, 3))        
        damping_data = np.zeros((args.num, 3))        
        iter_tau_data = np.zeros((args.num, 3))         
        iter_stiff_data = np.zeros((args.num, 3))          
        iter_damping_data = np.zeros((args.num, 3))          
    
        save_path = args.root_path + '/save_data/' + args.ctl_mode    
        
        wrist_control.tele_rehabilitation_test(    
            args.ctl_mode,        
            flag,       
            save_path,       
            para_input,        
            ref_data,       
            iter_tau_data,         
            iter_stiff_data,          
            iter_damping_data,          
            args.ee_force,         
            args.speed,        
            args.delay_time,         
            args.iter,        
            args.num,        
            initilize_setting,       
            args.slave,      
            args.master,         
            args.tele_rehab,   
            int(1)      
        )             