from tkinter import X  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.io import loadmat   
from GMRbasedGP.utils.gmr import plot_gmm, Gmr 
import seaborn as sns   

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", font_scale=1.2, rc=custom_params)

R_3 = 140    

angle_to_radian = np.pi/180     

font_size = 15     

wrist_joint_label_list = [r"$\phi_{PS}[\circ]$", r"$\phi_{RU}[\circ]$", r"$\phi_{FE}[\circ]$"]    
wrist_joint_vel_label_list = [r"$\dot{\phi}_{PS}[\circ/s]$", r"$\dot{\phi}_{RU}[\circ/s]$", r"$\dot{\phi}_{FE}[\circ/s]$"]  
wrist_tau_label_list = [r"$\tau_{PS}$[Nm]", r"$\tau_{RU}$[Nm]", r"$\tau_{FE}$[Nm]"]    
real_ee_effector_label_list = [r"$\psi_t^x$", r"$\psi_t^y$", r"$\psi_t^z$"]     
des_ee_effector_label_list = [r"$\psi_e^x$", r"$\psi_e^y$", r"$\psi_e^z$"]       
des_ee_tau_label_list = [r"$\tau_e^x$", r"$\tau_e^y$", r"$\tau_e^z$"]        
real_ee_tau_label_list = [r"$\tau_t^x$", r"$\tau_t^y$", r"$\tau_t^z$"]     
delta_ee_tau_label_list = [r"$\delta{\tau}_t^x$", r"$\delta{\tau}_t^y$", r"$\delta{\tau}_t^{FE}$"]        

des_robot_joint_label_list = [r"$q_e^1$", r"$q_e^2$", r"$q_e^3$"]    
# des_robot_joint_label_list = [r"$q_e^1$", r"$q_e^2$", r"$q_e^3$"]   
real_robot_joint_label_list = [r"$q_t^1[\circ]$", r"$q_t^2[\circ]$", r"$q_t^3[\circ]$"]     
des_robot_joint_vel_label_list = [r"$\dot{q}_e^1$", r"$\dot{q}_e^2$", r"$\dot{q}_e^3$"]     
real_robot_joint_vel_label_list = [r"$\dot{q}_t^1$", r"$\dot{q}_t^2$", r"$\dot{q}_t^3$"]     
real_robot_joint_torque_label_list = [r"$\tau_t^1$[Nm]", r"$\tau_t^2$[Nm]", r"$\tau_t^3$[Nm]"]    
delta_robot_joint_label_list = [r"$\delta q^1$", r"$\delta q^2$", r"$\delta q^3$"]     

colors = np.array(["#0072B2", "#F0E442", "#D55E00"])   
random_state, n_components, n_features = 2, 3, 2   

font_size = 15 

mycolors = {
    'nr': [213/255,15/255,37/255],  # new red
    'ng': [0/255,153/255,37/255],  # new green
    'nb': [51/255,105/255,232/255],  # new blue  
    'ny': [238/255,178/255,17/255],  # new yellow   
    'r': [180/255,20/255,47/255],  # red  
    'b': [0,114/255,189/255],  # blue  
    'db': [0,100/255,200/255],  # blue  
    'g': [119/255,172/255,48/255],  # green  
	'o': [217/255,83/255,25/255],  # orange  
	'y': [237/255,177/255,32/255], 
	'p': [126/255,47/255,142/255], 
	'pi': [204/255,102/255,102/255], 
	'lb': [77/255,190/255,238/255], 
	'li': [164/255,196/255,0], 
	'lr': [229/255,20/255,0], 
	'lg': [220/255,220/255,220/255], 
	'dr': [102/255,0,0],  
	'em': [0,138/255,0],  
    'br': [0.6510, 0.5725, 0.3412],  
    'gy': [0.6, 0.6, 0.6],  
    'vgy': [160/255, 160/255, 160/255],  
    'm': [1, 0, 1],  
    'c':  [0, 1, 1], 
    'rr': [1.0, 0.4, 0.4],  
    'gl': [0.8314, 0.7020, 0.7843]   
}   

color_name = ['nr', 'ng', 'nb', 'ny', 'r', 'b', 'db', 'g', 'o', 'y', 
	'p', 'pi', 'lb', 'li', 'lr', 'lg', 'dr', 'em', 'br', 'gy', 'vgy', 'm', 'c', 'rr', 'gl']   


def plot_epi_data(font_name='G', nb_samples=5, nb_dim=5, nb_data=200, X=None, Y=None):	
    plt.figure(figsize=(5, 5))   
    X_t = np.array(X[0:nb_data])   

    for p in range(nb_samples):    
        plt.plot(Y[p * nb_data : (p+1) * nb_data, 1], Y[p * nb_data : (p+1) * nb_data, 0], color=mycolors[color_name[p]], label='Epi_' + str(p))    
        # print("X_t :", X_t)   
        # for j in range(4):   
        #     plt.plot(X_t, Y[p * nb_data:(p + 1) * nb_data, j], color=mycolors[color_name[j+7]])    
        #     # plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')    

    # plot_gmm(np.array(gmr_model.mu)[:, 1:3], np.array(gmr_model.sigma)[:, 1:3, 1:3], alpha=0.6, color=[0.1, 0.34, 0.73])

    ax = plt.gca()  
    plt.ylabel("X[mm]")         
    plt.xlabel("Y[mm]")         
    ax.set_xlim(-30.0, 30.0)       
    ax.set_ylim(-30.0, 30.0) 
    ax.set_aspect(1)
    plt.tight_layout()       
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=20)    

    plt.legend() 
    # plt.savefig('figures/GMM_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  


def plot_raw_data(font_name='G', nb_samples=5, nb_dim=5, nb_data=200, X=None, Y=None, mu_gmr=None, sigma_gmr=None, ref_data=None):  	 
    plt.figure(figsize=(10, 5))   
    X_t = np.array(X[0:nb_data])   
    
    for p in range(nb_samples):   
        for j in range(nb_dim):    
            plt.plot(X_t, Y[p * nb_data:(p + 1) * nb_data, j], color=mycolors[color_name[j+7]])    
            # plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')    
    # print("mu :", gmr_model.mu)    
    
    for i in range(3):  
        plt.plot(X_t, ref_data[:, i], color='black', linewidth=3)  
        plt.plot(X_t, mu_gmr[:, i], color=[0.20, 0.54, 0.93], linewidth=3)   
        # plt.plot(X_t, mu_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3)    
        miny = mu_gmr[:, i] - np.sqrt(sigma_gmr[:, i, i])    
        maxy = mu_gmr[:, i] + np.sqrt(sigma_gmr[:, i, i])     
        # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
        plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.20, 0.54, 0.93], alpha=0.3)     
    
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])    
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])     
    # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
    # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.93, 0.54, 0.20], alpha=0.3)     
    
    axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    plt.xlabel('Time[s]', fontsize=font_size)      
    plt.ylabel('Output', fontsize=font_size)     
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/GMM_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  


def plot_GMM_raw_data(font_name='G', nb_samples=5, nb_data=200, Y=None, gmr_model=None):	
    plt.figure(figsize=(5, 5))    
    font_size = 20 
    for p in range(nb_samples):    
        plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
        plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')    

    plot_gmm(np.array(gmr_model.mu)[:, 1:3], np.array(gmr_model.sigma)[:, 1:3, 1:3], alpha=0.6, color=[0.1, 0.34, 0.73])

    ax = plt.gca()     
    # axes.set_xlim([-8., 8.])     
    # axes.set_ylim([-4., 14.])     
    # plt.xlabel('$y_1$', fontsize=30)      
    # plt.ylabel('$y_2$', fontsize=30)      
    plt.ylabel("X[mm]", fontsize=font_size)  # fontsize=25        
    plt.xlabel("Y[mm]", fontsize=font_size)  # fontsize=25        
    ax.set_xlim(-30.0, 30.0)        
    ax.set_ylim(-30.0, 30.0)         
    plt.locator_params(nbins=3)           
    plt.tick_params(labelsize=font_size)              
    plt.tight_layout()  
    print('figures/GMM_' + font_name +'.png')
    # plt.savefig('figures/GMM_' + font_name +'.pdf', bbox_inches='tight', pad_inches=0.0)    
    plt.savefig('./wrist_paper/figures/GMM_' + font_name + '.pdf', bbox_inches='tight', pad_inches=0.0)  
    plt.show()  
    
    
def plot_mean_var_fig(font_name='G', nb_samples=5, nb_data=200, Xt=None, Y=None, mu_gmr=None, sigma_gmr=None, pred_gmr=None, pred_sigma=None, ref_data=None, via_points=None, via_time=None):
    plt.figure(figsize=(14, 5))  
    font_size = 15 
    plt.subplot(1,3,1) 
    for p in range(nb_samples):   
        plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])   
        plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)    
    
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)   
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    
    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.05, color=[0.20, 0.54, 0.93])    
    
    # plt.plot(ref_data[1, :], ref_data[0, :], color="black", linewidth=3)   
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)  

        # pred value  
    plt.plot(pred_gmr[:, 0], pred_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3, linestyle='--')  
    
    ax = plt.gca()    
    
    plt.ylabel("X[mm]", fontsize=15)         
    plt.xlabel("Y[mm]", fontsize=15)          
    # ax.set_xlim(-30.0, 30.0)         
    # ax.set_ylim(-30.0, 30.0)      
    plt.locator_params(nbins=3)        
    plt.tick_params(labelsize=15)        
    plt.tight_layout()       

    ax_2 = plt.subplot(1,3,2) 
    # plt.figure(figsize=(5, 4))  
    for p in range(nb_samples):   
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7]) 
        
    plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3) 
    plt.plot(Xt[:, 0], pred_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3, linestyle='--') 
    miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])  
    maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])  
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)  
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_time[i], via_points[i, 0], color=[0.64, 0., 0.65], marker='X', s=80)  

    
    # plt.plot(Xt[:, 0], mu_gmr[:, 0] + ref_data[1, :], color="green", linewidth=3)   
    # plt.plot(Xt[:, 0], -1 * ref_data[1, :], color="black", linewidth=3)   
    
    # ax_2 = plt.gca()
    # ax_2.set_ylim([-30., 30.]) 
    plt.xlabel('$t$', fontsize=font_size) 
    plt.ylabel('$y_1$', fontsize=font_size) 
    plt.tick_params(labelsize=font_size)   
    plt.tight_layout()
    # plt.savefig('figures/GMR_' + font_name + '_1.png', bbox_inches='tight', pad_inches=0.0)  

    ax_3 = plt.subplot(1,3,3)  
    # plt.figure(figsize=(5, 4))   
    for p in range(nb_samples):   
        plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])  
    
    plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3) 
    plt.plot(Xt[:, 0], pred_gmr[:, 1], color=[0.93, 0.54, 0.20], linewidth=3, linestyle='--') 
    miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])  
    maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])   
    plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)  
    for i in range(via_points.shape[0]):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_time[i], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)  

    # plt.plot(Xt[:, 0], mu_gmr[:, 1] + ref_data[0, :], color="green", linewidth=3)   
    # plt.plot(Xt[:, 0], -1 * ref_data[0, :], color="black", linewidth=3) 
       
    # ax_3 = plt.gca()
    # ax_3.set_ylim([-30., 30.])
    plt.xlabel('$t$', fontsize=font_size)
    plt.ylabel('$y_2$', fontsize=font_size)
    plt.tick_params(labelsize=font_size)    
    plt.tight_layout()  
    # plt.savefig('figures/GMR_' + font_name + '_2.png', bbox_inches='tight', pad_inches=0.0)  
    
    plt.savefig('figures/GMR_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)  
    plt.show()   


def plot_mean_var(font_name='G', nb_samples=5, nb_data=200, Xt=None, Y=None, mu_gmr=None, sigma_gmr=None, ref_data=None): 
    plt.figure(figsize=(5, 5))   
    for p in range(nb_samples):   
        plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])   
        plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)    
    
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3)   
    plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)    
    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.15, color=[0.20, 0.54, 0.93])    
    
    plt.plot(ref_data[1, :], ref_data[0, :], color="black", linewidth=3)   
    
    ax = plt.gca()    
    # axes.set_xlim([-8., 8.])       
    # axes.set_ylim([-4., 14.])     
    # axes.set_xlim([-14, 14.])  
    # axes.set_ylim([-14., 14.])     
    # plt.xlabel('$y_1$', fontsize=30)     
    # plt.ylabel('$y_2$', fontsize=30)     
    plt.ylabel("X[mm]")   #, fontsize=15
    plt.xlabel("Y[mm]")   #, fontsize=15
    ax.set_xlim(-30.0, 30.0)         
    ax.set_ylim(-30.0, 30.0)      
    plt.locator_params(nbins=3)      
    plt.tick_params(labelsize=20)             
    plt.tight_layout()     
    print('figures/GMR_' + font_name + '.png') 
    # plt.savefig('figures/GMR_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)  
    plt.savefig('./wrist_paper/figures/GMR_' + font_name + '.pdf', bbox_inches='tight', pad_inches=0.0)  

    # plt.figure(figsize=(5, 4))  
    # for p in range(nb_samples):   
    #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7]) 
        
    # plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])  
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])  
    # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    # axes = plt.gca()
    # axes.set_ylim([-14., 14.]) 
    # plt.xlabel('$t$', fontsize=30) 
    # plt.ylabel('$y_1$', fontsize=30) 
    # plt.tick_params(labelsize=20)  
    # plt.tight_layout()
    # plt.savefig('figures/GMR_' + font_name + '_1.png', bbox_inches='tight', pad_inches=0.0)  

    # plt.figure(figsize=(5, 4))   
    # for p in range(nb_samples):   
    #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])  
    
    # plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3) 
    # miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])
    # maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])  
    # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    # axes = plt.gca()
    # axes.set_ylim([-14., 14.])
    # plt.xlabel('$t$', fontsize=30)
    # plt.ylabel('$y_2$', fontsize=30)
    # plt.tick_params(labelsize=20)  
    # plt.tight_layout()  
    
    # plt.savefig('figures/GMR_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)  
    
    # plt.show()   


def plot_via_points(
    font_name='G',   
    nb_posterior_samples=None,    
    via_points=None,  
    mu_gmr=None,  
    pred_gmr=None,  
    sigma_gmr=None,  
    sigma_kmp=None  
):  
    plt.figure(figsize=(5, 5))   
    # font_size = 15 
    # for p in range(nb_samples):  
    #     plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
    #     plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)

    plt.scatter(via_points[0, 0], via_points[0, 1], color='red', marker='X', s=80, label='start-points')   
    for i in range(1, nb_posterior_samples):  
        # plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)  
        plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80, label='via-points')  

    # ori value 
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.94, 0.54], linewidth=3)  
    plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.1, color=[0.20, 0.93, 0.54])   

    # pred value  
    plt.plot(pred_gmr[:, 0], pred_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3, linestyle='--')  
    plot_gmm(pred_gmr[:, :2], sigma_kmp[:, :2, :2]*0.5, alpha=0.1, color=[0.20, 0.54, 0.93])  
    
    # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80)  
    # print(mu_gmr.shape) 
    # print("mu_gmr : ", mu_gmr)
    # print("pred_gmr : ", pred_gmr)    
    # plot_gmm(mu_gmr[:, :2], sigma_gmr[:, :2, :2], alpha=0.05, color=[0.20, 0.54, 0.93])    
    
    obs_center = np.array([0, 18])  
    draw_circle_obs = plt.Circle(obs_center, 5, fill=True, color='black')     
    # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')     
    plt.gcf().gca().add_artist(draw_circle_obs)   

    ax = plt.gca()   
    plt.ylabel("X[mm]", fontsize=font_size)         
    plt.xlabel("Y[mm]", fontsize=font_size)    
    plt.legend()       
    # ax.set_xlim(-20.0, 20.0)       
    # ax.set_ylim(-20.0, 20.0)  
    # ax.set_xlim(-30.0, 30.0)       
    # ax.set_ylim(-30.0, 30.0)  
    # axes.set_xlim([-14, 14.])   
    # axes.set_ylim([-14., 14.])   
    # plt.xlabel('$y_1$', fontsize=30)   
    # plt.ylabel('$y_2$', fontsize=30)   
    plt.locator_params(nbins=3)  
    plt.tick_params(labelsize=font_size)   
    plt.tight_layout()  
    # plt.savefig('wrist_paper/figure/sec_5/GMR_via_points_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)   
    # plt.savefig('wrist_paper/figure/sec_5/GMR_via_points_' + font_name + '.pdf', bbox_inches='tight', pad_inches=0.0)  
    plt.savefig('./figures/GMR_via_points_' + font_name + '.png', bbox_inches='tight', pad_inches=0.0)       
    plt.show()  
    
    # plt.figure(figsize=(5, 4))  
    # for p in range(nb_samples): 
    #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 0], color=[.7, .7, .7]) 
      
    # plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0]) 
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0]) 
    # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    # axes = plt.gca()
    # # axes.set_ylim([-14., 14.]) 
    # plt.xlabel('$t$', fontsize=30) 
    # plt.ylabel('$y_1$', fontsize=30) 
    # # plt.tick_params(labelsize=20)  
    # plt.tight_layout()
    # plt.savefig('figures/GMR_' + font_name + '_1.png')  

    # plt.figure(figsize=(5, 4))   
    # for p in range(nb_samples):   
    #     plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])  
    
    # plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3) 
    # miny = mu_gmr[:, 1] - np.sqrt(sigma_gmr[:, 1, 1])
    # maxy = mu_gmr[:, 1] + np.sqrt(sigma_gmr[:, 1, 1])  
    # plt.fill_between(Xt[:, 0], miny, maxy, color=[0.20, 0.54, 0.93], alpha=0.3)
    # axes = plt.gca()
    # # axes.set_ylim([-14., 14.])
    # plt.xlabel('$t$', fontsize=30)
    # plt.ylabel('$y_2$', fontsize=30)  
    # plt.tick_params(labelsize=20)  
    # plt.tight_layout()  
    # plt.savefig('figures/GMR_' + font_name + '_2.png')  
    # plt.show() 
    

def plot_poster_samples(
    mu_gmr=None,  
    mu_gp_rshp=None,   
    sigma_gp_rshp=None,     
    Y_obs=None,  
    nb_posterior_samples=None,   
    mu_posterior=None   
):  
    # Posterior
    plt.figure(figsize=(5, 5))  
    plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
    plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])

    for i in range(nb_posterior_samples):
        plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
        plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)

    plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)
    plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)
    plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)

    ax = plt.gca()  
    plt.ylabel("X[mm]")         
    plt.xlabel("Y[mm]")         
    ax.set_xlim(-30.0, 30.0)       
    ax.set_ylim(-30.0, 30.0) 
    ax.set_aspect(1)   
    plt.tight_layout()        
    plt.locator_params(nbins=3)     
    plt.tick_params(labelsize=20)    
    plt.tight_layout()
    # plt.savefig('figures/GMRbGP_B_posterior_datasup.png')  
    plt.show()  


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1], 180 + angle, edgecolor="black"
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor("#56B4E9")
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker="o", color=colors[y], alpha=0.8)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-3.0, 3.0)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction="out")
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(
            k,
            w,
            width=0.9,
            color="#56B4E9",
            zorder=3,
            align="center",
            edgecolor="black",
        )
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.0), horizontalalignment="center")
    ax2.set_xlim(-0.6, 2 * n_components - 0.4)
    ax2.set_ylim(0.0, 1.1)
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax2.tick_params(axis="x", which="both", top=False)

    if plot_title:
        ax1.set_ylabel("Estimated Mixtures")
        ax2.set_ylabel("Weight of each component") 
        
        
def plot_error_data(font_name=None, X=None, nb_data=200, error_data=None): 
    plt.figure(figsize=(10, 5))   
    X_t = np.array(X[0:nb_data])   
    
    for i in range(3):  
        plt.plot(X_t, error_data[:, i], linewidth=3, label='state_'+str(i))    
        # plt.plot(X_t, mu_gmr[:, i], color=[0.20, 0.54, 0.93], linewidth=3)   
        # # plt.plot(X_t, mu_gmr[:, 0], color=[0.93, 0.54, 0.20], linewidth=3)    
        # miny = mu_gmr[:, i] - np.sqrt(sigma_gmr[:, i, i])    
        # maxy = mu_gmr[:, i] + np.sqrt(sigma_gmr[:, i, i])     
        # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
        # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.20, 0.54, 0.93], alpha=0.3)     
    
    # miny = mu_gmr[:, 0] - np.sqrt(sigma_gmr[:, 0, 0])    
    # maxy = mu_gmr[:, 0] + np.sqrt(sigma_gmr[:, 0, 0])     
    # # print(X_t.shape, np.array(miny[:, None]).shape, maxy.shape)   
    # plt.fill_between(np.squeeze(X_t), np.array(miny), np.array(maxy), color=[0.93, 0.54, 0.20], alpha=0.3)     
    
    axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    plt.xlabel('Time[s]', fontsize=font_size)      
    plt.ylabel('Output', fontsize=font_size)     
    plt.locator_params(nbins=3)   
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/GMR_error_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()   
    
    
def plot_mean_var_error(font_name=None, real_data=None, ref_data=None):   
    plt.figure(figsize=(10, 5))   
    # axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    label_xyz_list = ['X', 'Y', 'Z']  
    label_xyz_t_list = ['X_t', 'Y_t', 'Z_t']    
    for i in range(3): 
        plt.plot(ref_data[:, i], label=label_xyz_list[i])   
        
    for j in range(1):  
        for i in range(3): 
            plt.plot(real_data[j][:, i], label=label_xyz_t_list[i])    
            
    plt.xlabel('Time[s]', fontsize=font_size)        
    plt.ylabel('Output', fontsize=font_size)      
    plt.locator_params(nbins=3)    
    plt.tick_params(labelsize=font_size)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/Mean_std_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()  
    
    
def plot_xy_trajectory(font_name=None, real_data=None, ref_data=None):   
    plt.figure(figsize=(5, 5))   
    # axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])    
    label_xyz_list = ['X', 'Y', 'Z']  
    label_xyz_t_list = ['X_t', 'Y_t', 'Z_t']    
    # for i in range(3): 
    #     plt.plot(ref_data[:, i], label=label_xyz_list[i])   
        
    # for j in range(1):   
    #     for i in range(3):   
    #         plt.plot(real_data[j][:, i], label=label_xyz_t_list[i])    
    
    plt.plot(ref_data[:, 0], ref_data[:, 1], linewidth=3)         
    plt.scatter(real_data[0, 0], real_data[0, 1])   
     
    plt.plot(real_data[:, 0], real_data[:, 1])     

    plt.xlabel('X[mm]', fontsize=20)          
    plt.ylabel('Y[mm]', fontsize=20)         
    plt.locator_params(nbins=3)        
    plt.tick_params(labelsize=20)     
    plt.tight_layout()   
    plt.legend() 
    plt.savefig('figures/real_trajectory_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()   
    
    
def plot_pose_trajectory(font_name=None, point_list_eval=None, point_list=None):   
    fig = plt.figure(figsize=(8, 8))     
    ax = fig.gca(projection='3d')                                                                                                                     
    ax.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], color='b', linewidth=3, label='ref')   
    ax.plot3D(point_list_eval[:, 0], point_list_eval[:, 1], point_list_eval[:, 2], color='g', linewidth=3, linestyle=':', label='eval')           

    # plt.xlabel('time($t$)')    
    # plt.tight_layout()   
    # plt.legend(loc="upper right")    
    # plt.savefig('./data/wrist/ee_force_zero_force.png')    
    ax.set_xlim3d(-50, 50)       
    ax.set_ylim3d(-50, 50)       
    ax.set_zlim3d(-100, 0)       

    FONT_SIZE = 15
    ax.set_xlabel(r'$X(mm)$', fontsize=FONT_SIZE, labelpad=10)    
    ax.set_ylabel(r'$Y(mm)$', fontsize=FONT_SIZE, labelpad=10)     
    ax.set_zlabel(r'$Z(mm)$', fontsize=FONT_SIZE, labelpad=10)       
        
    # plt.tight_layout()    
    plt.legend()  
    plt.savefig('figures/ee_pose_eval.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)    
    
    
def plot_theta_real_trajectory(font_name=None, real_data=None, ref_data=None):   
    plt.figure(figsize=(12, 4))   
    # axes = plt.gca()   
    # axes.set_xlim([-14., 14.])     
    # axes.set_ylim([0., 1.])   

    for i in range(3):   
        plt.plot(ref_data[:, i], linewidth=4.5, label=des_robot_joint_label_list[i])    
        
        plt.plot(real_data[:, i], linewidth=2.5, label=real_robot_joint_label_list[i], linestyle=":")  
        
    # for j in range(1):  
    #     for i in range(3): 
    #         plt.plot(real_data[j][:, i], label=label_xyz_t_list[i])    
    
    # plt.plot(real_data[:, 0], real_data[:, 1])

    plt.xlabel('X[mm]', fontsize=20)        
    plt.ylabel('Y[mm]', fontsize=20)      
    plt.locator_params(nbins=3)    
    plt.tick_params(labelsize=20)     
    plt.tight_layout()    
    plt.legend()   
    plt.savefig('figures/theta_real_trajectory_' + font_name +'.png', bbox_inches='tight', pad_inches=0.0)    
    plt.show()   


def plot_ee_pose_deform(      
	point_list,  
    point_list_ref,         
	flag="_circle", 
 	save_fig=False   
):  
	fig = plt.figure(figsize=(8, 8))     
	ax = fig.gca(projection='3d')    
 
	ax.plot3D(point_list[:, 0], point_list[:, 1], point_list[:, 2], color='r', linewidth=2, label='real')   
	ax.plot3D(point_list_ref[:, 0], point_list_ref[:, 1], point_list_ref[:, 2], color='black', linewidth=3, linestyle=':', label='ref')             

	# plt.xlabel('time($t$)')   
	# plt.tight_layout()   
	# plt.legend(loc="upper right")   
	# plt.savefig('./data/wrist/ee_force_zero_force.png')  
	ax.set_xlim3d(-50, 50)      
	ax.set_ylim3d(-50, 50)      
	ax.set_zlim3d(-100, 0)       
	
	# plt.gca().set_box_aspect((1, 1, 1))    

	FONT_SIZE = 15
	ax.set_xlabel(r'$X(mm)$', fontsize=FONT_SIZE, labelpad=10)    
	ax.set_ylabel(r'$Y(mm)$', fontsize=FONT_SIZE, labelpad=10)     
	ax.set_zlabel(r'$Z(mm)$', fontsize=FONT_SIZE, labelpad=10)       
	  
	# plt.tight_layout()    
	plt.legend() 
	if save_fig: 
		plt.savefig('figures/ee_pose' + flag + '.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)    
	plt.show()   