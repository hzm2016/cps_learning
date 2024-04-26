import numpy as np
import os
import sys
sys.path.append("/home/yuxuan/Project/HPS_Perception/map_ws/src/HPS_Perception/hps_control/scripts")
from learning.mini_core.utils import *
from learning.mini_core.gait_recorder import GaitRecorder
from learning.mini_core.learning_cost import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


evaluation = 10


training_data_path = "/media/yuxuan/BackUp/HPS_Data/Debug_Ba/e{}/".format(int(evaluation))
saving_pair_path = training_data_path+"pair/"
os.makedirs(saving_pair_path, exist_ok=True)


state_files, policy_files = load_evaluation_file_list(training_data_path, evaluation)

num_imp = len(state_files)

ans = None
while ans != "y":
    ans = input("是否处理"+saving_pair_path+"的数据")



plt.ion()
plt.show(block=False)
plt.style.use("bmh")

fig = plt.figure(figsize=(10,5))
fig.canvas.draw()

grid = plt.GridSpec(3,1,wspace=0.5, hspace=0.5)
ax0 = plt.subplot(grid[0,0:])
ax1 = plt.subplot(grid[1,0:])
ax2 = plt.subplot(grid[2,0:])
axs = [ax0, ax1, ax2]

fig2 = plt.figure(figsize=(5,5))
fig2.canvas.draw()
ax_deq = fig2.add_subplot(111)

cost_of_v1s0 = []
file_of_v1s0 = []
cost_of_all = []
file_of_all = []
 

for n in range(num_imp):
    print("Load File", state_files[n])
    data_mat = np.load(training_data_path+state_files[n],allow_pickle=True)
    data_mat = data_mat[:,:-1].astype(np.float32)
    imp_mat = np.load(training_data_path+policy_files[n],allow_pickle=True)

    data_all = mat_to_dict(data_mat)

    plot_state_data(data_all, axs)
    
    for i in range(4):
        print("Phase{} Imp Paras(Kp, Kb, qe):".format(i), imp_mat[i,:])
    
    gait_recorder = GaitRecorder()

    idx = data_all['idx']

    gaits_info = []
    
    idx_deq = []


    for i in idx:
        try:
            q_rt = data_all['q_ht'][i]
            q_rs = data_all['q_hs'][i]
            q_rf = data_all['q_hf'][i]
            q_lt = data_all['q_pt'][i]
            q_ls = data_all['q_ps'][i]
            q_lf = data_all['q_pf'][i]
            q_tr = data_all['q_tr'][i]
            qd_rf = data_all['qd_hf'][i]
            qd_lf = data_all['qd_pf'][i]
            fr= data_all['fh'][i]
            fl = data_all['fp'][i]
            t = data_all['t'][i]

            if not gait_recorder.start_record:
                if gait_recorder.num_buffer_flush < 30:
                    gait_recorder.update_and_smooth_qd_and_qdd_for_gait_detection(q_rt, q_rs, q_rf, q_lt, q_ls, q_lf, 
                                                                                qd_rf, qd_lf, fr, fl)
                    gait_recorder.num_buffer_flush += 1
                else:
                    gait_recorder.update_and_smooth_qd_and_qdd_for_gait_detection(q_rt, q_rs, q_rf, q_lt, q_ls, q_lf, 
                                                                                qd_rf, qd_lf, fr, fl)
                    gait_recorder.detect_L_HS()
                if gait_recorder.start_record:
                    ax0.scatter(i, -data_all['q_pt'][i], c='m')
                    ax0.text(i-20, 0.6, "Begin Record", color='m',
                            fontsize=8, fontdict={"weight":'bold'})
                    idx_deq.append(i) 

            elif gait_recorder.start_record:
                gait_recorder.q_raw_enqueue([q_rt, q_rs, q_rf, q_lt, q_ls, q_lf, q_tr])
                gait_recorder.update_and_smooth_qd_and_qdd_for_gait_detection(q_rt, q_rs, q_rf, q_lt, q_ls, q_lf, 
                                                                                        qd_rf, qd_lf, fr, fl)
                if gait_recorder.in_stance_L:
                    gait_recorder.detect_L_FW()
                    if not gait_recorder.in_stance_L:
                        ax0.scatter(i, -data_all['q_pt'][i], c='g')
                else:
                    gait_recorder.detect_L_HS()
                    if gait_recorder.in_stance_L:
                        ax0.scatter(i, -data_all['q_pt'][i], c='r')

                if gait_recorder.in_stance_R:
                    gait_recorder.detect_R_FW()
                    if not gait_recorder.in_stance_R:
                        ax1.scatter(i, -data_all['q_ht'][i], c='g')
                        
                else:
                    gait_recorder.detect_R_HS()
                    if gait_recorder.in_stance_R:
                        ax1.scatter(i, -data_all['q_ht'][i], c='r')
                        
                if gait_recorder.num_R_HS == 2:
                    q_dequeue, t_dequeue, idx_HS = gait_recorder.gait_dequeue()
                    idx_l_deq = np.arange(idx_HS[0],idx_HS[1])+idx_deq[-1]
                    idx_r_deq = np.arange(idx_HS[2],idx_HS[3])+idx_deq[-1]
                    
                    q_lt_dequeue = -data_all['q_pt'][idx_l_deq]
                    q_ls_dequeue = data_all['q_ps'][idx_l_deq]-data_all['q_pt'][idx_l_deq]
                    q_lf_dequeue = data_all['q_pf'][idx_l_deq]+data_all['q_ps'][idx_l_deq]
                    t_l_dequeue = data_all['t'][idx_l_deq]
                    t_l_dequeue -= t_l_dequeue[0]
                    q_rt_dequeue = -data_all['q_ht'][idx_r_deq]
                    q_rs_dequeue = data_all['q_hs'][idx_r_deq]-data_all['q_ht'][idx_r_deq]
                    q_rf_dequeue = data_all['q_hf'][idx_r_deq]+data_all['q_hs'][idx_r_deq]
                    t_r_dequeue = data_all['t'][idx_r_deq]
                    t_r_dequeue -= t_r_dequeue[0]
                    phase_dequeue = data_all['phase'][idx_l_deq]

                    condition_dequeue = np.array([data_all['v'],data_all['s']])
                    # ax_deq.cla()
                    # ax_deq.plot(t_l_dequeue, gaussian_filter1d(q_ls_dequeue,2))
                    # ax_deq.plot(t_l_dequeue, phase_dequeue)
                    # ax_deq.plot(t_r_dequeue, gaussian_filter1d(q_rs_dequeue,2))

                    show_kinematic_cost(q_ls_dequeue, q_rs_dequeue, ax_deq, imp_mat)
                    
                    idx_deq.append(idx_HS[1]+idx_deq[-1])

                    save_data = [
                        q_lt_dequeue, q_ls_dequeue, q_lf_dequeue,t_l_dequeue,
                        q_rt_dequeue, q_rs_dequeue, q_rf_dequeue, t_r_dequeue, imp_mat,
                        condition_dequeue
                    ]

                    gaits_info.append(save_data)

        except KeyboardInterrupt as e:
            break
    if len(gaits_info)>4:
        a = input("是否生成键值对")
        if a == "y":
            state, cost = generate_pairs(gaits_info)
            print(state)
            np.save(saving_pair_path+"state_{}.npy".format(n),state)
            np.save(saving_pair_path+"reward_{}.npy".format(n),cost)
            # if gaits_info[0][-1][0,0] == 1 and gaits_info[0][-1][1,0] == 0:
            #     cost_of_v1s0.append(cost)
            #     file_of_v1s0.append(state_files[n])
            # cost_of_all.append(cost)
            # file_of_all.append(state_files[n])
    ax_deq.cla()


# cost_of_v1s0 = np.array(cost_of_v1s0)
# idx_min_cost_of_v1s0 = np.argsort(cost_of_v1s0)
# cost_best_in_v1s0 = cost_of_v1s0[idx_min_cost_of_v1s0[:5]]
# print(cost_best_in_v1s0)
# print("Best V1S0")
# for i in idx_min_cost_of_v1s0:
#     print(file_of_v1s0[i])

# cost_of_all = np.array(cost_of_all)
# idx_min_cost_of_all = np.argsort(cost_of_all)
# cost_best_in_all = cost_of_all[idx_min_cost_of_all[:5]]
# print(cost_best_in_all)
# print("Best of All")
# for i in idx_min_cost_of_all:
#     print(file_of_all[i])


    