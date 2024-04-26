import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
# from path_planning.path_generate import *
from matplotlib.animation import FuncAnimation

""" ================================= Plot result ===================================== """
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
""" ================================================================================= """

FONT_SIZE = 28  
linewidth = 4  

Length = [0.30, 0.15, 0.25, 0.125]  
L_1 = Length[0]  
L_2 = Length[2]  

# writing space
WIDTH = 0.370  
HEIGHT = 0.370  

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = FONT_SIZE 
# sns.set_theme()
params = {
'axes.axisbelow': True,
 'axes.edgecolor': 'white',
 'axes.facecolor': '#EAEAF2',
 'axes.grid': True,
 'axes.labelcolor': '.15',
 'axes.linewidth': 0.0,
 'figure.facecolor': 'white',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
'Liberation Sans',
'Bitstream Vera Sans',
  'sans-serif'],
 'grid.color': 'white',
 'grid.linestyle': '-',
 'image.cmap': 'Greys',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': 'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': 'out',
 'xtick.major.size': 1.0,
 'xtick.minor.size': 0.0,
 'ytick.color': '.15',
 'ytick.direction': 'out',
 'ytick.major.size': 1.0,
 'ytick.minor.size': 0.0
 }
# sns.axes_style(rc=params) 
sns.set(font_scale=1.5)


def plot_real_trajectory(
    root_path='./motor_control/bin/data/',  
):
    """ Including angle, velocity and torque"""
    angle_list = np.loadtxt(root_path + 'angle_list.txt', skiprows=1),
    torque_list = np.loadtxt(root_path + 'torque_list.txt', skiprows=1),
    angle_vel_list = np.loadtxt(root_path + 'angle_vel_list.txt', skiprows=1),
    angle_list_e = np.loadtxt('./motor_control/bin/2_font_3_angle_list.txt',
                              delimiter=',', skiprows=1)
    
    angle_list_1 = angle_list[:, 0]
    angle_list_2 = angle_list[:, 1]
    
    torque_list_1 = torque_list[:, 0]
    torque_list_2 = torque_list[:, 1]
    
    angle_vel_list_1 = angle_vel_list[:, 0]
    angle_vel_list_2 = angle_vel_list[:, 1]
    
    angle_list_1_e = angle_list_e[:, 0]
    angle_list_2_e = angle_list_e[:, 1]
    
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(2, 4, 1)
    plt.subplots_adjust(wspace=2, hspace=0)
    
    plt.plot(angle_list_1, linewidth=linewidth)
    # plt.xlim([0, 128])
    # plt.ylim([0, 128])
    plt.xlabel('time($t$)')
    plt.ylabel('$q_1$(rad)')
    # plt.axis('equal')
    plt.tight_layout()
    
    plt.subplot(2, 4, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    plt.plot(angle_list_2, linewidth=linewidth)
    
    # plt.xlim([0., 0.6])
    # plt.ylim([0., 0.6])
    plt.xlabel('time($t$)')
    plt.ylabel('$q_2$(rad)')
    
    plt.subplot(2, 4, 3)
    plt.plot(torque_1_o, linewidth=linewidth-2)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_1$(Nm)')
    plt.legend()
    
    plt.subplot(2, 4, 4)
    plt.plot(torque_list_2, linewidth=linewidth-2)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_2$(Nm)')
    plt.legend()
    
    plt.subplot(2, 4, 5)
    plt.plot(angle_vel_list_1, linewidth=linewidth)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_1$(Nm)')
    plt.legend()
    
    plt.subplot(2, 4, 6)
    plt.plot(angle_vel_list_2, linewidth=linewidth)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_2$(Nm)')
    plt.legend()
    
    plt.subplot(2, 4, 7)
    plt.plot(angle_list_1_e, linewidth=linewidth)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_1$(Nm)')
    plt.legend()
    
    plt.subplot(2, 4, 8)
    plt.plot(angle_list_2_e, linewidth=linewidth)
    
    plt.xlabel('time($t$)')
    plt.ylabel('$\tau_2$(Nm)')
    plt.legend()
    
    plt.show()
    

def plot_real_2d_path(
    root_path='./motor_control/bin/data/',
    file_name='',
    stroke_num=1, 
    epi_time=0,
    delimiter=',',
    skiprows=1
):
    """ 
        plot angle trajectory and cartesian path 
    """
    FONT_SIZE = 28 
    linewidth = 4 

    fig = plt.figure(figsize=(20, 8))
    
    # plt.subplot(1, 2, 1)
    # plt.subplots_adjust(wspace=0, hspace=0)

    # plt.plot(angle_list_1_e, linewidth=linewidth, label='angle_1_e') 
    # plt.plot(angle_list_1_t, linewidth=linewidth, label='angle_1_t')
    # plt.plot(angle_list_2_e, linewidth=linewidth, label='angle_2_e') 
    # plt.plot(angle_list_2_t, linewidth=linewidth, label='angle_2_t')

    # plt.xlabel('time($t$)', fontsize=FONT_SIZE)
    # plt.ylabel('$rad', fontsize=FONT_SIZE)
    # plt.legend()

    
    plt.subplot(1, 1, 1) 
    for i in range(stroke_num): 

        angle_list = np.loadtxt(root_path + file_name + str(i) + '_' + str(epi_time) + '.txt', delimiter=delimiter, skiprows=skiprows)

        angle_list_1_e = angle_list[:, 0] 
        angle_list_2_e = angle_list[:, 3]

        angle_list_1_t = angle_list[:, 1] 
        angle_list_2_t = angle_list[:, 4] 

        # d_angle_list_1_t = angle_list[:, 2] 
        # d_angle_list_2_t = angle_list[:, 5] 

        x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
        y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)

        x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t) 
        y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t) 
    
        # plt.plot(x_e, y_e, linewidth=linewidth, label='desired')  
        # plt.plot(x_t, y_t, linewidth=linewidth, label='real')  

    # plt.subplot(1, 2, 1) 
    # for i in range(stroke_num): 
        # print(x_e - x_t)   
        plt.plot(x_e-x_t, linewidth=linewidth, label='error')  
        plt.plot(y_e-y_t, linewidth=linewidth, label='error')  

    plt.xlabel('x(m)', fontsize=FONT_SIZE)  
    plt.ylabel('y(m)', fontsize=FONT_SIZE)  
    plt.ylim([-WIDTH/2, WIDTH/2])  
    plt.xlim([0.13, 0.13 + WIDTH])  
    # plt.legend()
    plt.savefig('xing' + str(epi_time) + '.png') 
    plt.show() 


def plot_real_error_path(
    root_path='./motor_control/bin/data/', 
    file_name='',
    stroke_num=1,
    epi_num=0,
    delimiter=' ',
    skiprows=1
):
    """ 
        plot angle trajectory and cartesian path 
    """
    FONT_SIZE = 28   
    linewidth = 4

    error_x = np.zeros((stroke_num, epi_num)) 
    error_y = np.zeros((stroke_num, epi_num)) 
    for i in range(stroke_num): 
        for j in range(epi_num):  
            # print(root_path + file_name + str(i) + '_' + str(j) + '.txt')
            angle_list = np.loadtxt(root_path + file_name + str(i) + '_' + str(j) + '.txt', delimiter=delimiter, skiprows=skiprows)
            
            angle_list_1_e = angle_list[:, 0]   
            angle_list_2_e = angle_list[:, 3]   
    
            angle_list_1_t = angle_list[:, 1]   
            angle_list_2_t = angle_list[:, 4]   

            # d_angle_list_1_t = angle_list[:, 2] 
            # d_angle_list_2_t = angle_list[:, 5] 

            x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
            y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)

            x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t) 
            y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t) 


            error_x[i, j] = sum(x_e - x_t)/x_t.shape[0]
            error_y[i, j] = sum(y_e - y_t)/y_t.shape[0]

        # plt.plot(x_e, y_e, linewidth=linewidth, label='desired')  
        # plt.plot(x_t, y_t, linewidth=linewidth, label='real')  
    print(x_e - x_t)
    print("x_t :", x_t.shape)

    # plt.subplot(1, 2, 1) 
    # for i in range(stroke_num): 
        # print(x_e - x_t)   
        # plt.plot(x_e-x_t, linewidth=linewidth, label='error')   
        # plt.plot(y_e-y_t, linewidth=linewidth, label='error')   
    
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)  
    for i in range(stroke_num): 
        plt.plot(error_x[i, :], linewidth=linewidth, label='stroke_' + str(i)) 
    plt.xlabel('train times', fontsize=FONT_SIZE)   
    plt.ylabel('x(m)', fontsize=FONT_SIZE)  
    plt.legend()

    plt.subplot(1, 2, 2) 
    for i in range(stroke_num): 
        plt.plot(error_y[i, :], linewidth=linewidth, label='stroke_' + str(i)) 
    
    plt.xlabel('train times', fontsize=FONT_SIZE)   
    plt.ylabel('y(m)', fontsize=FONT_SIZE)  
    plt.subplots_adjust(wspace=0.2, hspace=0) 

    # plt.plot(angle_list_1_e, linewidth=linewidth, label='angle_1_e') 
    # plt.plot(angle_list_1_t, linewidth=linewidth, label='angle_1_t')
    # plt.plot(angle_list_2_e, linewidth=linewidth, label='angle_2_e') 
    # plt.plot(angle_list_2_t, linewidth=linewidth, label='angle_2_t')

    # plt.xlabel('time($t$)', fontsize=FONT_SIZE)
    # plt.ylabel('$rad', fontsize=FONT_SIZE)
    # plt.legend()
    # plt.xlabel('x(m)', fontsize=FONT_SIZE)   
    # plt.ylabel('y(m)', fontsize=FONT_SIZE)  
    # plt.ylim([-WIDTH/2, WIDTH/2])  
    # plt.xlim([0.13, 0.13 + WIDTH])  
    plt.legend()
    # plt.savefig('xing' + str(epi_time) + '.png') 
    plt.show() 



def plot_real_stroke_2d_path(
        root_path='./motor_control/bin/data/',
        file_name='',
        stroke_num=1,
        delimiter=',',
        skiprows=1
):
    """
        plot angle trajectory and cartesian path
    """
    FONT_SIZE = 28
    linewidth = 4
    
    fig = plt.figure(figsize=(20, 8))
    
    # plt.subplot(1, 2, 1)
    # plt.subplots_adjust(wspace=0, hspace=0)
    
    # plt.plot(angle_list_1_e, linewidth=linewidth, label='angle_1_e')
    # plt.plot(angle_list_1_t, linewidth=linewidth, label='angle_1_t')
    # plt.plot(angle_list_2_e, linewidth=linewidth, label='angle_2_e')
    # plt.plot(angle_list_2_t, linewidth=linewidth, label='angle_2_t')
    
    # plt.xlabel('time($t$)', fontsize=FONT_SIZE)
    # plt.ylabel('$rad', fontsize=FONT_SIZE)
    # plt.legend()
    
    plt.subplot(1, 1, 1)
    
    # for i in range(stroke_num):
    angle_list = np.loadtxt(root_path + file_name + '.txt', delimiter=delimiter, skiprows=skiprows)
    
    angle_list_1_e = angle_list[:, 0]
    angle_list_2_e = angle_list[:, 1]
    
    # angle_list_1_t = angle_list[:, 1]
    # angle_list_2_t = angle_list[:, 4]
    
    # d_angle_list_1_t = angle_list[:, 2]
    # d_angle_list_2_t = angle_list[:, 5]
    
    x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
    y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)
    
    # x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t)
    # y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t)
    
    plt.plot(x_e, y_e, linewidth=linewidth, label='desired')
    # plt.plot(x_t, y_t, linewidth=linewidth, label='real')
    plt.scatter(np.flipud(x_e)[0], np.flipud(y_e)[1], s=100, c='b', marker='o')
    
    plt.xlabel('x(m)', fontsize=FONT_SIZE)
    plt.ylabel('y(m)', fontsize=FONT_SIZE)
    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    # plt.legend()
    
    plt.show()


def plot_torque_path(
        root_path='./motor_control/bin/data/',
        file_angle_name='', 
        file_torque_name='' 
):
    """ plot angle trajectory and cartesian path"""
    FONT_SIZE = 28
    linewidth = 4
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = FONT_SIZE
    # print("angle_list :", root_path + file_angle_name)

    angle_list_e = np.loadtxt(root_path + file_angle_name, delimiter=',', skiprows=1) 
    max_index = angle_list_e.shape[0]

    angle_list_1_e = angle_list_e[:max_index, 0]
    angle_list_2_e = angle_list_e[:max_index, 1]

    torque_list = np.loadtxt(root_path + file_torque_name, delimiter=',', skiprows=1)
    torque_list_1 = torque_list[:max_index, 0]
    torque_list_2 = torque_list[:max_index, 1]
    
    fig = plt.figure(figsize=(24, 8)) 
    
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.plot(angle_list_1_e, linewidth=linewidth, label='angle 1')
    plt.plot(angle_list_2_e, linewidth=linewidth, label='angle 2')
    
    plt.xlabel('time($t$)')  # fontsize=FONT_SIZE
    plt.ylabel('rad')  # fontsize=FONT_SIZE
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(torque_list_1, linewidth=linewidth, label='angle 1')
    plt.plot(torque_list_2, linewidth=linewidth, label='angle 2')
    
    plt.xlabel('time($t$)')  # fontsize=FONT_SIZE
    plt.ylabel('Nm')  # fontsize=FONT_SIZE
    plt.legend()
    
    plt.show()


def plot_real_2d_demo_path(
    root_path='./motor_control/bin/data/',
    file_name='',
    delimiter=',',
    skiprows=1
):
    """ 
        plot angle trajectory and cartesian path 
    """
    FONT_SIZE = 28 
    linewidth = 4 

    fig = plt.figure(figsize=(20, 8))
    
    # plt.subplot(1, 2, 1)
    # plt.subplots_adjust(wspace=0, hspace=0)

    # plt.plot(angle_list_1_e, linewidth=linewidth, label='angle_1_e') 
    # plt.plot(angle_list_1_t, linewidth=linewidth, label='angle_1_t')
    # plt.plot(angle_list_2_e, linewidth=linewidth, label='angle_2_e') 
    # plt.plot(angle_list_2_t, linewidth=linewidth, label='angle_2_t')

    # plt.xlabel('time($t$)', fontsize=FONT_SIZE)
    # plt.ylabel('$rad', fontsize=FONT_SIZE)
    # plt.legend()

    plt.subplot(1, 1, 1) 

    # for i in range(stroke_num): 

    angle_list = np.loadtxt(root_path + file_name + '_demonstrated_angle_list.txt', delimiter=delimiter, skiprows=skiprows)

    angle_list_1_e = angle_list[:, 0]  
    angle_list_2_e = angle_list[:, 2]  

    # angle_list_1_t = angle_list[:, 1]   
    # angle_list_2_t = angle_list[:, 4]   

    # d_angle_list_1_t = angle_list[:, 2] 
    # d_angle_list_2_t = angle_list[:, 5] 

    x_e = L_1 * np.cos(angle_list_1_e) + L_2 * np.cos(angle_list_1_e + angle_list_2_e)
    y_e = L_1 * np.sin(angle_list_1_e) + L_2 * np.sin(angle_list_1_e + angle_list_2_e)

    # x_t = L_1 * np.cos(angle_list_1_t) + L_2 * np.cos(angle_list_1_t + angle_list_2_t) 
    # y_t = L_1 * np.sin(angle_list_1_t) + L_2 * np.sin(angle_list_1_t + angle_list_2_t) 

    plt.plot(x_e, y_e, linewidth=linewidth, label='desired')  
    # plt.plot(x_t, y_t, linewidth=linewidth, label='real')  

    plt.xlabel('x(m)', fontsize=FONT_SIZE)  
    plt.ylabel('y(m)', fontsize=FONT_SIZE)  
    plt.ylim([-WIDTH/2, WIDTH/2])  
    plt.xlim([0.13, 0.13 + WIDTH])   
    # plt.legend()  

    plt.show()  


def plot_txt_data(
    file_angle_name='',   
    save_file_name='demonstration.pdf',  
    high_scale=0.3,  
    wide_scale=0.3  
):
    """ plot angle trajectory and cartesian path"""
    FONT_SIZE = 20
    linewidth = 2.0

    # read angle
    angle_list_e = np.loadtxt(file_angle_name, delimiter=',', skiprows=1)
    # max_index = angle_list_e.shape[0] 
    max_index = 2500;    
    # num_theta = angle_list_e.shape[1]   
    num_theta = 6
    
    fig = plt.figure(figsize=(20, 15))
    
    plt.subplot(1, 1, 1)  
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)
    
    for i in range(6): 
        plt.plot(angle_list_e[10:max_index,i], linewidth=linewidth, label='theta_' + str(i)) 
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  
    plt.ylabel('rad', fontsize=FONT_SIZE)  
    plt.title('All Theta', fontsize=FONT_SIZE) 
    # plt.ylim([3, 4]) 
    plt.yticks(fontsize=FONT_SIZE) 
    plt.xticks(fontsize=FONT_SIZE)   
    # plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)  
    
    # plt.savefig(save_file_name) 
    
    plt.show()   
    
    return angle_list_e[10:max_index, :3]   


def plot_sea_angle_torque_path(
    file_angle_name='',  
    file_torque_name='',  
    file_impedance_name='',   
    save_file_name='demonstration.pdf',  
    high_scale=0.3,  
    wide_scale=0.3  
):
    """ plot angle trajectory and cartesian path"""
    FONT_SIZE = 20
    linewidth = 2.0

    # read angle
    angle_list_e = np.loadtxt(file_angle_name, delimiter=',', skiprows=1)
    max_index = angle_list_e.shape[0] 

    theta_1_t = angle_list_e[:max_index, 0]  # theta_1_t 
    q_1_t = angle_list_e[:max_index, 2]  # q_1_t 
    q_1_e = angle_list_e[:max_index, 4]  # q_1_e 
    theta_2_t = angle_list_e[:max_index, 6]  # theta_2_t 
    q_2_t = angle_list_e[:max_index, 8]  # q_2_t 
    q_2_e = angle_list_e[:max_index, 10]  # q_2_e 
    # position error 
    delta_q_1 = q_1_e - q_1_t
    delta_q_2 = q_2_e - q_2_t 

    d_theta_1_t = angle_list_e[:max_index, 1]  # d_theta_1_t
    d_q_1_t = angle_list_e[:max_index, 3]  # d_q_1_t
    d_q_1_e = angle_list_e[:max_index, 5]  # d_q_1_e
    d_theta_2_t = angle_list_e[:max_index, 7]  # d_theta_2_t
    d_q_2_t = angle_list_e[:max_index, 9]  # d_q_2_t
    d_q_2_e = angle_list_e[:max_index, 11]  # d_q_2_e

    # read torque 
    torque_list = np.loadtxt(file_torque_name, delimiter=',', skiprows=1)
    
    torque_1_o = torque_list[:max_index, 0]  # torque_1_o
    torque_1_t = torque_list[:max_index, 1]  # torque_1_t
    tau_1 = torque_list[:max_index, 2]  # tau_1
    torque_2_o = torque_list[:max_index, 3]  # torque_2_o
    torque_2_t = torque_list[:max_index, 4]  # torque_2_t
    tau_2 = torque_list[:max_index, 5]  # tau_2

    # read impedance 
    impedance_list = np.loadtxt(file_impedance_name, delimiter=',', skiprows=1)
    tau_1_t = impedance_list[:max_index, 0]  # tau_1_t    
    kp_1_t = impedance_list[:max_index, 1]  # kp_1_t    
    kd_1_t = impedance_list[:max_index, 2]  # kd_1_t  
    tau_2_t = impedance_list[:max_index, 3]  # tau_2_t
    kp_2_t = impedance_list[:max_index, 4]  
    kd_2_t = impedance_list[:max_index, 5]  
    
    fig = plt.figure(figsize=(65, 15))
    
    plt.subplot(2, 4, 1)
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)
    
    plt.plot(theta_1_t, linewidth=linewidth, label=r'$\theta_{1t}$')
    plt.plot(q_1_t, linewidth=linewidth, label=r'$q_{1t}$')
    plt.plot(q_1_e, linewidth=linewidth, label=r'$q_{1e}$')
    plt.plot(delta_q_1, linewidth=linewidth, label=r'$\Delta q_1$')
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)
    plt.ylabel('rad', fontsize=FONT_SIZE)
    plt.title('Pos Joint 1', fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)
    
    plt.subplot(2, 4, 2)
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)
    plt.plot(d_theta_1_t, linewidth=linewidth, label=r'$\dot{\theta}_{1t}$') 
    plt.plot(d_q_1_t, linewidth=linewidth, label=r'$\dot{q}_{1t}$') 
    plt.plot(d_q_1_e, linewidth=linewidth, label=r'$\dot{q}_{1e}$')
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('rad/s', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Vel Joint 1', fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)

    plt.subplot(2, 4, 3)
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale) 
    plt.plot(torque_1_o, linewidth=linewidth, label=r'$\tau_{1i}$') 
    plt.plot(torque_1_t, linewidth=linewidth, label=r'$\tau_{1t}$') 
    plt.plot(tau_1, linewidth=linewidth, label=r'$\tau_{1o}$')
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('Nm', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Tor Joint 1', fontsize=FONT_SIZE)   
    plt.yticks(fontsize=FONT_SIZE)   
    plt.xticks(fontsize=FONT_SIZE)   
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)   
    # plt.legend(fontsize=FONT_SIZE)  

    plt.subplot(2, 4, 4) 
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)   
    plt.plot(tau_1_t, linewidth=linewidth, label=r'$\tau_{1t}$')   
    # plt.plot(kp_1_t, linewidth=linewidth, label=r'$k_{p1}$')   
    # plt.plot(kd_1_t, linewidth=linewidth, label=r'$k_{d1}$')   
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('Nm', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Learned Impedance', fontsize=FONT_SIZE)  
    plt.yticks(fontsize=FONT_SIZE)    
    plt.xticks(fontsize=FONT_SIZE)   
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)  
    
    plt.subplot(2, 4, 5)
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)
    plt.plot(theta_2_t, linewidth=linewidth, label=r'$\theta_{2t}$') 
    plt.plot(q_2_t, linewidth=linewidth, label=r'$q_{2t}$') 
    plt.plot(q_2_e, linewidth=linewidth, label=r'$q_{2e}$')
    plt.plot(delta_q_2, linewidth=linewidth, label=r'$\Delta q_2$')
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('rad', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Pos Joint 2', fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)

    plt.subplot(2, 4, 6)
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)
    plt.plot(d_theta_2_t, linewidth=linewidth, label=r'$\dot{\theta}_{2t}$')
    plt.plot(d_q_2_t, linewidth=linewidth, label=r'$\dot{q}_{2t}$')
    plt.plot(d_q_2_e, linewidth=linewidth, label=r'$\dot{q}_{2e}$')
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('rad/s', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Vel Joint 2', fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)
    
    plt.subplot(2, 4, 7)  
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)  
    plt.plot(torque_2_o, linewidth=linewidth, label=r'$\tau_{1i}$')   
    plt.plot(torque_2_t, linewidth=linewidth, label=r'$\tau_{1t}$')   
    plt.plot(tau_2, linewidth=linewidth, label=r'$\tau_{1o}$')   
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('Nm', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Tor Joint 2', fontsize=FONT_SIZE)  
    plt.yticks(fontsize=FONT_SIZE)  
    plt.xticks(fontsize=FONT_SIZE)  
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)

    plt.subplot(2, 4, 8)  
    plt.subplots_adjust(wspace=wide_scale, hspace=high_scale)   
    plt.plot(tau_2_t, linewidth=linewidth, label=r'$\tau_{2t}$')    
    # plt.plot(kp_2_t, linewidth=linewidth, label=r'$k_{p2}$')   
    # plt.plot(kd_2_t, linewidth=linewidth, label=r'$k_{d2}$')   
    
    plt.xlabel('time($t$)', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.ylabel('Nm', fontsize=FONT_SIZE)  # fontsize=FONT_SIZE
    plt.title('Learned Impedance', fontsize=FONT_SIZE)  
    plt.yticks(fontsize=FONT_SIZE)    
    plt.xticks(fontsize=FONT_SIZE)   
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=FONT_SIZE)
    # plt.legend(fontsize=FONT_SIZE)  
    
    plt.savefig(save_file_name) 
    
    # plt.show()

    return q_1_t, q_2_t, delta_q_1, delta_q_2 


def plot_real_osc_2d_demo_path(
    osc_list
):
    """
            plot angle trajectory and cartesian path
        """
    FONT_SIZE = 28
    linewidth = 4
    
    fig = plt.figure(figsize=(10, 10))
    osc_list = np.array(osc_list)
    
    plt.subplot(1, 1, 1)
    for i in range(osc_list.shape[0]):
        # x_e = osc_list[i, :, 0]
        # y_e = osc_list[i, :, 1]
        x_e = osc_list[i][:, 0]
        y_e = osc_list[i][:, 1]
        plt.plot(x_e, y_e, linewidth=linewidth, label='desired')
        # plt.plot(x_t, y_t, linewidth=linewidth, label='real')
    
    plt.xlabel('x(m)', fontsize=FONT_SIZE)
    plt.ylabel('y(m)', fontsize=FONT_SIZE)
    plt.ylim([0, 128])
    plt.xlim([0, 128])
    
    plt.show()
    

if __name__ == "__main__":
    
    
    fig, ax = plt.subplots()
    x, y = [], []
    line, = plt.plot([], [], '.-', color='orange')
    nums = 50  # 需要的帧数 
    
    
    # def init():
    #     ax.set_xlim(-5, 60)
    #     ax.set_ylim(-3, 3)
    #     return line
    #
    #
    # def update(step):
    #     if len(x) >= nums:  # 通过控制帧数来避免不断的绘图
    #         return line
    #     x.append(step)
    #     y.append(np.cos(step / 3) + np.sin(step ** 2))  # 计算y
    #     line.set_data(x, y)
    #     return line
    #
    #
    # ani = FuncAnimation(fig, update, frames=nums,  # nums输入到frames后会使用range(nums)得到一系列step输入到update中去
    #                     init_func=init, interval=500) 

    # stroke_length = 3
    # for i in range(stroke_length): 
    #     angle_list = np.loadtxt('../data/font_data/chuan/angle_list_' + str(i) + '.txt', delimiter=' ')
    #     N_way_points = angle_list.shape[0]
    #     # print("N_way_points :", N_way_points)
    #     # word_path.append(way_points.copy())
    #     # angle_point_1 = way_points[-1, :]
    #     # end_point = forward_ik(angle_point_1)
    #     point_list = forward_ik_path(angle_list)

    #     plt.plot(point_list[:, 0], linewidth=linewidth, label='x_1(m)')
    #     plt.plot(point_list[:, 1], linewidth=linewidth, label='x_2(m)')
        
    #     plt.show()
    #     plt.pause(1)