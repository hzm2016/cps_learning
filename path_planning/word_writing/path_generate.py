import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import math
import seaborn as sns
sns.set_theme()
sns.set(font_scale=0.5)

import copy
import cv2
import PIL.Image as Image

from scipy.signal import savgol_filter
import scipy.interpolate as spi

# writing space
WIDTH = 0.360
HEIGHT = 0.360

# image size
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# joints limits
action_dim = 2
ANGLE_1_RANGE = np.array([-1.90, 1.90])
ANGLE_2_RANGE = np.array([-2.2, 2.5])

center_shift=np.array([0.15, -WIDTH / 2])

Length = [0.30, 0.150, 0.25, 0.125]
L1 = Length[0]
L2 = Length[2]

Ts = 0.001

# plot parameters
linewidth = 3


def initial_parameter_estimate(num_demons_each_style=30):
    """ demonstration for each styles with zero impedance """
    num_demonstrations = 30
    
    writting_vel = 0.0
    impedance_params = np.ones(2)
    
    # captured images
    images_list = []
    distance_list = []
    for i in range(num_demonstrations):
        distance_list.appen
        images_list.append(i.copy())

    return writting_vel, impedance_params


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def calibrate_para():
    ori_point = np.array([])


def IK(point):
    """
        Inverse kinematics
    """
    angle = np.zeros(action_dim)
    x1 = point[0]
    x2 = point[1]
    
    L = x1**2 + x2**2

    gamma = math.atan2(x2, x1)

    cos_belta = (L1 ** 2 + L - L2 ** 2) / (2 * L1 * np.sqrt(L))

    if cos_belta > 1:
        angle_1 = gamma
    elif cos_belta < -1:
        angle_1 = gamma - np.pi
    else:
        angle_1 = gamma - math.acos(cos_belta)

    cos_alpha = (L1 ** 2 - L + L2 ** 2) / (2 * L1 * L2)

    if cos_alpha > 1:
        angle_2 = np.pi
    elif cos_alpha < -1:
        angle_2 = 0
    else:
        angle_2 = np.pi - math.acos(cos_alpha)

    angle[0] = np.round(angle_1, 5).copy()
    angle[1] = np.round(angle_2, 5).copy()
    return angle


def forward_ik(angle):
    """ 
        calculate point
    """
    point = np.zeros_like(angle)
    point[0] = L1 * math.cos(angle[0]) + L2 * math.cos(angle[0] + angle[1])
    point[1] = L1 * math.sin(angle[0]) + L2 * math.sin(angle[0] + angle[1])
    
    return point


def Jacobian(theta):
    """
        calculate Jacobian
    """
    J = np.zeros((action_dim, action_dim))
    
    J[0, 0] = -L1 * math.sin(theta[0]) - L2 * math.sin(theta[0] + theta[1])
    J[0, 1] = -L2 * math.sin(theta[0] + theta[1])
    J[1, 0] = L1 * math.cos(theta[0]) + L2 * math.cos(theta[0] + theta[1])
    J[1, 1] = L2 * math.cos(theta[0] + theta[1])
    
    return J


def Stiff_convert(theta, stiffness, damping):
    """
        convert stiffness from task space to joint space
    """
    J = Jacobian(theta)
    stiff_joint = J.transpose().dot(stiffness).dot(J)
    damping_joint = J.transpose().dot(damping).dot(J)
    
    return stiff_joint, damping_joint


def forward_ik_path(angle_list, transfer_to_img=True):
    """
        calculate osc point
    """
    point_list = np.zeros_like(angle_list)
    
    for i in range(point_list.shape[0]):
        point_list[i, 0] = L1 * math.cos(angle_list[i, 0]) + L2 * math.cos(angle_list[i, 0] + angle_list[i, 1])
        point_list[i, 1] = L1 * math.sin(angle_list[i, 0]) + L2 * math.sin(angle_list[i, 0] + angle_list[i, 1])
    
    point_list = point_list - center_shift
    if transfer_to_img:
        point_list[:, 0] *= 128/WIDTH
        point_list[:, 1] *= 128/HEIGHT
        
    return point_list


def generate_stroke_stiffness_path(angle_list, stiffness_list, damping_list,
                                   save_path=False, save_root='', word_name='yi', stroke_name=0):
    """
    :param angle_list:
    :param stiffness_list:
    :param damping_list:
    :return:
    """
    stiff_joint_list = []
    damping_joint_list = []
    for i in range(angle_list.shape[0]):
        stiff_task = np.diag(stiffness_list[i, :]).copy()
        damping_task = np.diag(damping_list[i, :]).copy()
        stiff_joint, damping_joint = Stiff_convert(angle_list[i, :], stiff_task, damping_task)
        stiff_joint_list.append([stiff_joint[0, 0], stiff_joint[1, 1]])
        damping_joint_list.append([damping_joint[0, 0], damping_joint[1, 1]])
    
    params_list = np.hstack((stiff_joint_list, damping_joint_list))
    print("params_list :", params_list.shape)
    if save_path:
        np.savetxt(save_root + '/' + word_name + '/' + 'params_list_' + str(stroke_name) + '.txt',
                   params_list, fmt='%.05f')
    
    return params_list


def generate_stroke_path(traj, inter_type=1, inverse=True,
                  center_shift=np.array([-WIDTH/2, 0.23]),
                  velocity=0.04, Ts=0.001, filter_size=17, plot_show=False, save_path=False, word_name=None, stroke_name=0):
    """
         generate stroke trajectory from list
         velocity ::: 0.04m/s
    """
    # calculate length of path
    dist = 0.0
    for i in range(len(traj) - 1):
        point_1 = np.array([traj[i, 1], traj[i, 0]])
        point_2 = np.array([traj[i+1, 1], traj[i+1, 0]])
        dist += np.linalg.norm((point_2.copy() - point_1.copy()), ord=2)

    path_data = np.zeros_like(traj)
    
    path_data[:, 0] = savgol_filter(traj[:, 0], filter_size, 3, mode='nearest')
    path_data[:, 1] = savgol_filter(traj[:, 1], filter_size, 3, mode='nearest')
    
    # M = N//len(traj)
    # x_list = []
    # y_list = []
    # for i in range(len(traj)):
    #     # need to check which dimension can be applied for interp
    #     x_list_i = np.linspace(path_data[i, 1], path_data[i+1, 1], M)
    #     y_list_i = np.interp(x_list_i, path_data[i:i+1, 1], path_data[i:i+1, 0])
    #     x_list.append(x_list_i)
    #     y_list.append(y_list_i)

    # # need to check which dimension can be applied for interp
    # x_list = np.linspace(path_data[-1, 1], path_data[0, 1], N)
    # # x_list = path(1, 2):(path(end, 2) - path(1, 2)) / (N - 1): path(end, 2)
    # y_list = np.interp(x_list, path_data[:, 1][::-1], path_data[:, 0][::-1])

    # transform to work space
    ratio = IMAGE_WIDTH / WIDTH
    
    period = dist/ratio/velocity
    print("Distance (mm) :", np.array(dist))
    print("Period (s) :", np.array(period))
    N = np.array(period / Ts).astype(int)
    
    start_point = np.array([path_data[0, 1], path_data[0, 0]])
    end_point = np.array([path_data[-1, 1], path_data[-1, 0]])
    dir = end_point - start_point
    angle = math.atan2(dir[1], dir[0])
    
    if angle > -math.pi/4 and angle < 0:
        inter_type = 2
        inverse = False
    if angle > 3.0 or angle < -3.0:
        inter_type = 2
    if angle > -math.pi/2 and angle < - math.pi/4:
        inter_type = 1
    if angle > math.pi/4 and angle < math.pi *3/4:
        inverse = False
    if angle > math.pi *3/4 and angle < math.pi:
        inter_type = 2
    
    sample_x = []
    sample_y = []
    if inter_type==1:
        if inverse:
            y_list = np.linspace(path_data[-1, 0], path_data[0, 0], N)
            x_list = np.interp(y_list, path_data[:, 0][::-1], path_data[:, 1][::-1])
        else:
            y_list = np.linspace(path_data[0, 0], path_data[-1, 0], N)
            x_list = np.interp(y_list, path_data[:, 0], path_data[:, 1])
            
            # sample_y = np.array(path_data[:, 0])
            # sample_x = np.array(path_data[:, 1])
            #
            # # 进行三次样条拟合
            # ipo3 = spi.splrep(sample_y, sample_x, k=3)  # 样本点导入，生成参数
            # x_list = spi.splev(y_list, ipo3)  # 根据观测点和样条参数，生成插值
    elif inter_type==2:
        if inverse:
            x_list = np.linspace(path_data[-1, 1], path_data[0, 1], N)
            y_list = np.interp(x_list, path_data[:, 1][::-1], path_data[:, 0][::-1])
        else:
            x_list = np.linspace(path_data[0, 1], path_data[-1, 1], N)
            y_list = np.interp(x_list, path_data[:, 1], path_data[:, 0])
    else:
        print("Please check the given stroke path !!!")
    
    image_points = np.vstack((x_list, y_list)).transpose()

    x_1_list = x_list/ratio + center_shift[0]
    x_2_list = y_list/ratio + center_shift[1]

    # print("x_list :", x_list)
    # print("y_list :", y_list)
    
    task_points = np.vstack((x_1_list, x_2_list)).transpose()
    
    # print("x_1_list :::", x_1_list)
    # print("x_2_list :::", x_2_list)
    
    # inverse
    # x_1_list = np.hstack([x_1_list, x_1_list[::-1]])
    # x_2_list = np.hstack([x_2_list, x_2_list[::-1]])
    # x_1_list = np.hstack([x_1_list])
    # x_2_list = np.hstack([x_2_list])

    angle_1_list_e = []
    angle_2_list_e = []

    for t in range(1, N):
        x1 = x_1_list[t]
        x2 = x_2_list[t]
        
        point = np.array([x1, x2])
        
        angle = IK(point)
        
        # Inverse kinematics
        angle_1_list_e.append(np.round(angle[0].copy(), 5))
        angle_2_list_e.append(np.round(angle[1].copy(), 5))

    max_angle_1 = np.max(angle_1_list_e)
    max_angle_2 = np.max(angle_2_list_e)
    print("Max angle 1 (rad) :", max_angle_1)
    print("Max angle 2 (rad):", max_angle_2)
    if max_angle_1 < ANGLE_1_RANGE[0] or max_angle_1 > ANGLE_1_RANGE[1]:
        print("!!!!!! angle 1 is out of range !!!!!")
        print("max angle 1 :::", max_angle_1)
        exit()

    if max_angle_2 < ANGLE_2_RANGE[0] or max_angle_2 > ANGLE_2_RANGE[1]:
        print("!!!!!! angle 1 is out of range !!!!!")
        print("max angle 2 :::", max_angle_2)
        exit()
    
    way_points = np.vstack((angle_1_list_e, angle_2_list_e)).transpose()
    print('+' * 50)
    print("Check success with way_points :", way_points.shape[0])
    
    if plot_show:
        plot_stroke_path(period, traj, image_points, task_points, way_points)
     
    if save_path:
        np.savetxt('control/data/font_data/' + word_name + '/' + 'angle_list_' + str(stroke_name) + '.txt', way_points, fmt='%.05f')

    return way_points, image_points, task_points, period


def generate_word_path(
        traj_list,
        stiffness,
        damping,
        inter_list=None, 
        inverse_list=None, 
        center_shift=np.array([0.23, -WIDTH/2]), 
        velocity=0.04,
        filter_size=17,
        plot_show=False, 
        save_path=False,
        save_root='control/font_data',
        word_name='tian'): 
    """
        generate word path
    """
    # inter_list = np.ones(len(traj_list))
    word_angle_list = []
    word_image_points = []
    word_task_points = []
    period_list = []
    word_params_list = []
    for stroke_index in range(len(traj_list)):
        """ get one stroke """
        traj = traj_list[stroke_index]
        print('=' * 20)
        print('stroke index :', stroke_index)
        stroke_angle_list, stroke_image_points, stroke_task_points, period \
            = generate_stroke_path(
                traj,
                inter_type=inter_list[stroke_index],
                inverse=inverse_list[stroke_index],
                center_shift=center_shift,
                velocity=velocity,
                filter_size=filter_size,
                plot_show=False,
                save_path=save_path,
                word_name=word_name,
                stroke_name=stroke_index
            )
        
        stiffness_list = np.tile(stiffness, (stroke_angle_list.shape[0], 1))
        damping_list = np.tile(damping, (stroke_angle_list.shape[0], 1))
        params_list = generate_stroke_stiffness_path(stroke_angle_list, stiffness_list, damping_list,
                                       save_path=save_path, save_root=save_root, word_name=word_name, stroke_name=stroke_index)
        
        word_angle_list.append(stroke_angle_list)
        word_image_points.append(stroke_image_points)
        word_task_points.append(stroke_task_points.copy())
        period_list.append(period)
        word_params_list.append(params_list[:, :2])
        
        
    if plot_show:
        plot_word_path(period_list, traj_list, word_image_points, word_task_points, word_angle_list,
                       word_folder=save_root,
                       word_name=word_name)
        
        # plot_torque(word_params_list, period_list)


def path_planning(start_point, target_point, velocity=0.04):
    """
        path planning
    """
    dist = np.linalg.norm((start_point - target_point), ord=2)
    T = dist / velocity
    N = int(T / Ts)
    
    # print("start_point :", start_point[0])
    # print("end_point :", start_point[1])
    
    x_list = np.linspace(start_point[0], target_point[0], N)
    y_list = np.linspace(start_point[1], target_point[1], N)
    
    point = start_point
    angle_list = []
    for i in range(N):
        point[0] = x_list[i]
        point[1] = y_list[i]
        angle = IK(point)
        angle_list.append(angle)
    
    return np.array(angle_list), N


def plot_stroke_path(period, traj, image_points, task_points, angle_list, fig_name='Stroke Path'):
    """
        check the planned path
    """
    t_list = np.linspace(0.0, period, angle_list.shape[0])
    plt.rcParams['font.size'] = 8
    print("task points :", task_points)
    plt.figure(figsize=(15, 4))
    
    plt.title(fig_name)
    
    plt.subplot(1, 3, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    plt.plot(traj[:, 1], traj[:, 0], marker='o', linewidth=linewidth)
    plt.plot(image_points[:, 0], image_points[:, 1], linewidth=linewidth - 2)
    
    plt.xlim([0, 128])
    plt.ylim([0, 128])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    # plt.axis('equal')
    plt.tight_layout()
    
    plt.subplot(1, 3, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    plt.plot(task_points[:, 0], task_points[:, 1], linewidth=linewidth + 2, color='r')
    # plt.plot(x_inner, y_inner, linewidth=linewidth + 2, color='r')
    plt.scatter(task_points[0, 0], task_points[0, 1], s=100, c='b', marker='o')
    # plt.scatter(x_inner[0], y_inner[0], s=100, c='b', marker='o')
    # print("distance :::", np.sqrt((x_1_list[0] - x_inner[0])**2 + (x_2_list[0] - y_inner[0])**2))
    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0., 0.13 + WIDTH])
    plt.xlabel('$x_1$(m)')
    plt.ylabel('$x_2$(m)')
    
    plt.subplot(1, 3, 3)
    plt.plot(t_list, angle_list[:, 0], linewidth=linewidth, label='$q_1$')
    # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
    plt.plot(t_list, angle_list[:, 1], linewidth=linewidth, label='$q_2$')
    # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')
    
    plt.xlabel('Time (s)')
    plt.ylabel('One-loop Angle (rad)')
    plt.legend()
    
    plt.show()


def rotate_point(angle, x_list, y_list):
    x_rotated = x_list * math.cos(angle) + y_list * math.sin(angle)
    y_rotated = y_list * math.cos(angle) - x_list * math.sin(angle)
    
    return x_rotated, y_rotated


def plot_word_path(period_list, traj_list, image_points_list, task_points_list, word_angle_list,
                   word_folder='../control/data', word_name='Stroke Path'):
    """
        plot one word path
    """
    plt.figure(figsize=(15, 4))
    plt.title(word_name)
    
    plt.subplot(1, 3, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for i in range(len(traj_list)):
        # traj_list[i].transpose((0, 1)).copy()
        # plt.plot(traj_list[i][:, 1], traj_list[i][:, 0], marker='o', linewidth=linewidth)
        # plt.plot(image_points_list[i][:, 0], image_points_list[i][:, 1], linewidth=linewidth - 2)
        
        traj_x_rotated, traj_y_rotated = rotate_point(math.pi/2, traj_list[i][:, 1], traj_list[i][:, 0])
        # print("x_rotated :", traj_x_rotated, "y_rotated :", traj_y_rotated)
        plt.plot(traj_x_rotated, traj_y_rotated, marker='o', linewidth=linewidth)
        image_x_rotated, image_y_rotated = rotate_point(math.pi / 2, image_points_list[i][:, 0], image_points_list[i][:, 1])
        # plt.plot(traj_list[i][:, 0], traj_list[i][:, 1], marker='o', linewidth=linewidth)
        plt.plot(image_x_rotated, image_y_rotated, linewidth=linewidth - 2)
        
        # if i == range(len(traj_list) - 1):
        #     len(traj_list)
    
    plt.xlim([0, 128])
    plt.ylim([-128, 0])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    plt.subplot(1, 3, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for i in range(len(traj_list)):
        # plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2, color='r')
        # plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')
        
        rotated_task_points_x_list, rotated_task_points_y_list = \
            rotate_point(math.pi / 2, task_points_list[i][:, 0], task_points_list[i][:, 1])
        plt.plot(rotated_task_points_x_list, rotated_task_points_y_list, linewidth=linewidth + 2, color='r')
        if i == len(traj_list) -1:
            pass
        else:
            plt.scatter(rotated_task_points_x_list[0], rotated_task_points_y_list[0], s=100, c='b', marker='o')
        
        # rotated_task_points_x_list, rotated_task_points_y_list = \
        #     rotate_point(math.pi / 2, task_points_list[i][:, 0], task_points_list[i][:, 1])
        # plt.scatter(x_inner[0], y_inner[0], s=100, c='b', marker='o')
        # print("distance :::", np.sqrt((x_1_list[0] - x_inner[0])**2 + (x_2_list[0] - y_inner[0])**2))
    
    # plt.ylim([-WIDTH / 2, WIDTH / 2])
    # plt.xlim([0.13, 0.13 + WIDTH])
    plt.xlabel('$x_1$(m)')
    plt.ylabel('$x_2$(m)')
    
    plt.subplot(1, 3, 3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    total_period = sum(period_list)
    print("total_period :", total_period)
    for i in range(len(traj_list)):
        if i == 0:
            start_period = 0.0
            # t_list = np.linspace(0.0, period_list[i], word_angle_list[i].shape[0])
        else:
            start_period = sum(period_list[:i])
        
        t_list = np.linspace(start_period, period_list[i] + start_period, word_angle_list[i].shape[0])
        print("period :", start_period, period_list[i] + start_period)
        
        plt.plot(t_list, word_angle_list[i][:, 0], linewidth=linewidth, label='$q_1$')
        # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
        plt.plot(t_list, word_angle_list[i][:, 1], linewidth=linewidth, label='$q_2$')
        # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')
    
    plt.xlabel('Time (s)')
    plt.ylabel('One-loop Angle (rad)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(word_folder + '/' + word_name + '/' + word_name + '_traj.png')
    
    plt.show()
    
    fig = plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for i in range(len(traj_list)-1):
        # rotate_task_point = task_points_list[i].transpose((1, 0))
        if i == len(traj_list)-2:
            task_points_list_x_last = np.hstack((task_points_list[i][:, 0], task_points_list[i+1][:, 0]))
            task_points_list_y_last = np.hstack((task_points_list[i][:, 1], task_points_list[i + 1][:, 1]))
            plt.plot(task_points_list_x_last, task_points_list_y_last, linewidth=linewidth + 2)
            plt.scatter(task_points_list_x_last[0], task_points_list_y_last[0], s=100, c='b', marker='o')
            plt.text(task_points_list_x_last[0], task_points_list_y_last[0], str(i + 1), rotation=90)
        else:
            plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2)
            plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')
            plt.text(task_points_list[i][0, 0], task_points_list[i][0, 1], str(i + 1), rotation=90)
    
    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    plt.yticks([-WIDTH / 2, 0, WIDTH / 2])
    plt.xticks([0.13, 0.13 + WIDTH])
    # plt.xlabel('$x_1$(m)')
    # plt.ylabel('$x_2$(m)')
    plt.tight_layout()
    
    img_path = fig2data(fig)
    # img = img_path.transpose(Image.ROTATE_90)  # 将图片旋转90度
    # img_path.show()
    img_show = np.rot90(img_path, -1)
    cv2.imwrite(word_folder + '/' + word_name + '/' + word_name + '.png', img_show)
    # cv2.imshow(word_folder + '/' + word_name + '/' + word_name +'.png', img_show)
    # cv2.waitKey(0)
    # plt.imshow(img_show)
    # plt.show()


def real_stroke_path(task_points_list=None):
    fig = plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for i in range(len(task_points_list)):
        plt.plot(task_points_list[i][:, 0], task_points_list[i][:, 1], linewidth=linewidth + 2)
        # plt.scatter(task_points_list[i][0, 0], task_points_list[i][0, 1], s=100, c='b', marker='o')
        # plt.text(task_points_list[i][0, 0], task_points_list[i][0, 1], str(i + 1), rotation=90)
    
    plt.ylim([-WIDTH / 2, WIDTH / 2])
    plt.xlim([0.13, 0.13 + WIDTH])
    # plt.xlabel('$x_1$(m)')
    # plt.ylabel('$x_2$(m)')
    # plt.axis('off')
    plt.tight_layout()
    
    img_path = fig2data(fig)
    # img = img_path.transpose(Image.ROTATE_90)  # 将图片旋转90度
    # img_path.show()
    img_show = np.rot90(img_path, -1)
    # cv2.imwrite(word_folder + '/' + word_name + '/' + word_name +'.png', img_show)
    # plt.show()
    cv2.imshow("Real path :", img_show)
    cv2.waitKey(0)


def plot_torque(torque_list, period_list):
    """
        torque_list
    """
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    total_period = sum(period_list)
    print('*' * 50)
    print("total_period :", total_period)
    for i in range(len(torque_list)):
        if i == 0:
            start_period = 0.0
        else:
            start_period = sum(period_list[:i])
        
        t_list = np.linspace(start_period, period_list[i] + start_period, torque_list[i].shape[0])
        print("period :", start_period, period_list[i] + start_period)
        
        plt.plot(t_list, torque_list[i][:, 0], linewidth=linewidth, label='$q_1$')
        # plt.plot(t_list[1:], angle_vel_1_list_e, linewidth=linewidth, label='$d_{q1}$')
        plt.plot(t_list, torque_list[i][:, 1], linewidth=linewidth, label='$q_2$')
    # plt.plot(t_list[1:], angle_vel_2_list_e, linewidth=linewidth, label='$d_{q2}$')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Stiffness (Nm/rad)')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.tight_layout()
    
    plt.show()
    

def contact_two_stroke(angle_list_1, angle_list_2,
                       params_list_1, params_list_2, inverse=False):
    
    if inverse:
        final_angle_list = np.vstack((angle_list_1, angle_list_2))
        final_params_list = np.vstack((params_list_1, params_list_2))
    else:
        final_angle_list = np.vstack((angle_list_1, np.flipud(angle_list_2)))
        final_params_list = np.vstack((params_list_1, np.flipud(params_list_2)))

    print("Final angle shape :", final_angle_list.shape)
    print("Final params shape :", final_params_list.shape)
    return final_angle_list, final_params_list