import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

k0 = [[70., 50., 15., 12.], 
      [50., 50., 25., 20.]]

b0 = [[2., 2., 1., 1.], 
      [6., 6., 1., 1.]]

q_e0 = [[2., 12., 60., 1.],
        [5., 12., -11., 1.]]



lim_kp = [
    [[50.,100.],[35.,70.],[13.,35.],[11.5,20.]], #kp lim for knee
    [[30.,70.],[40.,80.],[5.,40.],[10.,30.]] #kp lim for ankle
]

lim_kb = [
    [[1,10],[1,10],[1,4],[1,4]], #kb lim for knee
    [[1,10],[1,10],[1,5],[1,5]] #kp lim for ankle
]

lim_qe = [
    [[0.,10.],[2.,25.],[55.,68.],[-2.,5.]], #qe lim for knee
    [[0.,5.], [8.,15.],[-15.,-10.],[-3.,5.]] #qe lim for ankle
]

lim_cost = [0., 10.]

lim_v = [0.8, 1.4]

lim_s = [-8.,8.]


def load_evaluation_file_list(data_path, evaluation=0):
    file_list = os.listdir(data_path)
    state_file = []
    policy_file = []
    for f in file_list:
        if f == "pair":
            continue
        context = f.split("_")
        e = int(context[5])
        kind = context[-1]
        if e == evaluation and kind == "state.npy":
            state_file.append(f)
        elif e == evaluation and kind == "policy.npy":
            policy_file.append(f)
    state_file.sort(key=lambda x: int(x.split('_')[7]))
    policy_file.sort(key=lambda x: int(x.split('_')[7]))
    return state_file, policy_file

def load_state_and_cost_list(data_path):
    file_list = os.listdir(data_path)
    state_file = []
    reward_file = []
    for f in file_list:
        if f[:5] == "state":
            state_file.append(f)
        elif f[:6] == "reward":
            reward_file.append(f)
    state_file.sort(key=lambda x: int(x.split('_')[1][0]))
    reward_file.sort(key=lambda x: int(x.split('_')[1][0]))
    state_info = []
    reward_info = []
    for f in state_file:
        state_info.append(np.load(data_path+f))

    state_info = np.array(state_info)
    for f in reward_file:
        reward_info.append(np.load(data_path+f))
    reward_info = np.array(reward_info)
    return state_info, reward_info

def fifo_vec(data_vec, data):
    data_vec[0:-1] = data_vec[1:]
    data_vec[-1] = data
    return data_vec

def kernel(size,std):
    r = np.arange(-size,size+1)
    kernel = np.exp(-r**2/(2*std**2))
    return kernel/sum(kernel)

def online_gaussian_filter(data,kernel):
    return np.sum(data*kernel)

def normalize(value, lim):
    upper = lim[1]
    lower = lim[0]
    value = np.clip(value, a_min=lower, a_max=upper)
    value = (value-lower)/(upper-lower)
    return value

def scaling(value, lim):
    upper = float(lim[1])
    lower = float(lim[0])
    value = np.clip(value, a_min=0, a_max=1)
    value = value*(upper-lower)+lower
    return value

def mat_to_dict(data):
    data_all = {}
    data_all['q_ht'] = data[:,0]
    data_all['q_hs'] = data[:,1]
    data_all['q_hf'] = data[:,2]
    data_all['q_pt'] = data[:,3]
    data_all['q_ps'] = data[:,4]
    data_all['q_pf'] = data[:,5]
    data_all['q_tr'] = data[:,6]
    data_all['qd_hf'] = data[:,7]
    data_all['qd_pf'] = data[:,8]
    data_all['q_knee'] = data[:, 11]
    data_all['q_ankle'] = data[:, 12]
    data_all['phase'] = data[:,13]
    data_all['v'] = data[:,14]
    data_all['s'] = data[:,15]
    data_all['fh'] = data[:,16]
    data_all['fp'] = data[:,17]
    data_all['fh_x'] = data[:,18]
    data_all['fp_x'] = data[:,19]
    # data_all['t'] = data[:,20]
    data_all['idx'] = np.arange(np.shape(data)[0])
    data_all['t'] = np.arange(np.shape(data)[0])*0.01
    return data_all

def plot_state_data(dict, axs):
    q_ht = -gaussian_filter1d(dict['q_ht'],2)
    q_hk = gaussian_filter1d(dict['q_hs']-dict['q_ht'],2)
    q_hf = gaussian_filter1d(dict['q_hs']+dict['q_hf'],2)

    q_pt = -gaussian_filter1d(dict['q_pt'],2)
    q_pk = gaussian_filter1d(dict['q_ps']-dict['q_pt'],2)
    q_pf = gaussian_filter1d(dict['q_ps']+dict['q_pf'],2)

    fh = dict['fh']
    fp = dict['fp']
    fh_x = dict['fh_x']
    fp_x = dict['fp_x']
    idx = dict['idx']

    axs[0].cla()
    axs[0].plot(idx, q_pt)
    axs[0].plot(idx, q_pk, alpha=0.2)
    axs[0].set_xlabel("Index", fontsize=10,
                   fontdict={"weight":'bold'})
    axs[0].set_ylabel("Pros Leg Angle(rad)",fontsize=10, 
                   fontdict={"weight":'bold'})
    axs[0].set_ylim([-0.35,1])

    axs[1].cla()
    axs[1].plot(idx, q_ht)
    axs[1].plot(idx, q_hk, alpha=0.2)
    axs[1].set_xlabel("Index", fontsize=10,
                   fontdict={"weight":'bold'})
    axs[1].set_ylabel("Healthy Leg Angle(rad)",fontsize=10, 
                   fontdict={"weight":'bold'})
    axs[1].set_ylim([-0.35,1])

    axs[2].cla()
    axs[2].plot(idx, fp)
    axs[2].plot(idx, fp_x)
    axs[2].plot(idx, fh)
    axs[2].plot(idx, fh_x)
    axs[2].set_xlabel("Index", fontsize=10,
                   fontdict={"weight":'bold'})
    axs[2].set_ylabel("Foot Plate(N)",fontsize=10, 
                   fontdict={"weight":'bold'})

def check_generated_imp(imp):
    if np.shape(imp)[1] != 24:
        return False
    idx_knee = 0
    idx_ankle = 1
    idx_kp = 0
    idx_kb = 1
    idx_qe = 2
    imp_ok = True

    def get_imp(idx_frame, idx_pde, idx_joint, phase):
        return imp[idx_frame,idx_pde*8+idx_joint*4+phase]
    
    def print_error(idx, idx_pde, idx_joint, phase):
        if idx_joint == 0:
            joint = "Knee"
        elif idx_joint == 1:
            joint = "Ankle"
        if idx_pde == 0:
            pde = "Kp"
        elif idx_pde == 1:
            pde = "Kb"
        elif idx_pde == 2:
            pde = "qe"
        print("\033[31mError in {} phase: {}, joint: {}, {}\033[0m".format(idx, phase, joint, pde))        

    for i in range(np.shape(imp)[0]):
        for phase in range(4):
            for joint in range(2):
                kp_range = lim_kp[joint][phase]
                if not (kp_range[0]<=get_imp(i, idx_kp, joint, phase)<=kp_range[1]):
                    print_error(i, idx_kp, joint, phase)
                    imp_ok = False
                kb_range = lim_kb[joint][phase]
                if not (kb_range[0]<=get_imp(i, idx_kb, joint, phase)<=kb_range[1]):
                    print_error(i, idx_kp, joint, phase)
                    imp_ok = False
                qe_range = lim_qe[joint][phase]
                if not (qe_range[0]<=get_imp(i, idx_qe, joint, phase)<=qe_range[1]):
                    print_error(i, idx_qe, joint, phase)
                    imp_ok = False
    return imp_ok
    
        