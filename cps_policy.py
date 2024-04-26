import sys 
import os  

# from .. import  
import glob
import argparse  

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)  

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
from utils import *  

np.set_printoptions(precision=4)     


class Dataset(object):  
    def __init__(self, context_dim, para_dim, max_size=int(1e6)):
        #
        self._context_buf = np.zeros(shape=(max_size, context_dim), dtype=np.float32)  
        self._para_buf = np.zeros(shape=(max_size, para_dim), dtype=np.float32)   
        self._reward_buf = np.zeros(shape=(max_size, 1), dtype=np.float32) 
        #
        self._max_size = max_size
        self._ptr = 0   
        self._size = 0   

    def add(self, context, para, reward):   
        self._context_buf[self._ptr] = context   
        self._para_buf[self._ptr] = para  
        self._reward_buf[self._ptr] = reward      
        #
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def sample(self, batch_size):  
        index = np.random.randint(0, self._size, size=batch_size)
        #
        return [self._context_buf[index], self._para_buf[index] ]   
    
    def sample_pairs(self, batch_size):  
        index = np.random.randint(0, self._size, size=batch_size)
        
        state_info = np.hstack((self._context_buf[index], self._para_buf[index])) 
        reward_info = self._reward_buf[index]   
        return state_info, reward_info       

    def save(self, save_dir, n_iter):  
        save_dir = save_dir + "/dataset"  
        if not os.path.exists(save_dir):  
            os.mkdir(save_dir)   
        data_path = save_dir + "/data_{0}".format(n_iter)
        np.savez(data_path, context=self._context_buf, para=self._para_buf)   

    def load(self, data_dir):   
        self._context_buf, self._para_buf = None, None
        for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
            size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
            np_file = np.load(data_file)
            #
            context_array = np_file['context'][:size, :]
            para_array = np_file['para'][:size, :]
            if i == 0:
                self._context_buf = context_array
                self._para_buf = para_array    
            else:
                self._context_buf = np.append(self._context_buf, context_array, axis=0)
                self._para_buf = np.append(self._para_buf, para_array, axis=0)
        #
        self._size  = self._context_buf.shape[0]
        self._ptr = self._size - 1


# class Actor(Model):

#     def __init__(self, obs_dim, act_dim, \
#         learning_rate=3e-4, hidden_units=[32, 32],  \
#         activation='relu', trainable=True, actor_name="actor"):
#         super(Actor, self).__init__(actor_name)
#         #
#         obs = Input(shape=(obs_dim, ))
#         #
#         x = Dense(hidden_units[0], activation=activation, trainable=trainable)(obs)
#         x = Dense(hidden_units[1], activation=activation, trainable=trainable)(x)
#         #
#         act = Dense(act_dim, trainable=trainable)(x)
#         #
#         self._act_net = Model(inputs=obs, outputs=act)
#         self._optimizer = Adam(lr=learning_rate)

#     def call(self, inputs):
#         return self._act_net(inputs)

#     def save_weights(self, save_dir, iter):
#         save_dir = save_dir + "/act_net"
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         weights_path = save_dir + "/weights_{0}.h5".format(iter)
#         self._act_net.save_weights(weights_path)

#     def load_weights(self, file_path):
#         self._act_net.load_weights(file_path)

#     @tf.function
#     def train_batch(self, obs_batch, act_batch):
#         with tf.GradientTape(persistent=True) as tape:
#             act_pred = self.call(obs_batch)

#             # mean squared error
#             mse_loss = 0.5 * tf.math.reduce_mean( (act_batch - act_pred)**2 )
#         #   
#         act_grad = tape.gradient(mse_loss, self._act_net.trainable_variables)
#         self._optimizer.apply_gradients( zip(act_grad, self._act_net.trainable_variables))

#         del tape

#         return mse_loss
    

class CPS(object):     
    def __init__(self, cps_para=None):      
        self.context_dim = cps_para['context_dim']      
        self.para_dim = cps_para['para_dim']    
        self.policy = cps_para['policy_type']     
        self.reps = None     
        self.cps_dataset = Dataset(self.context_dim, self.para_dim)    

        self.policy_path = cps_para['policy_path'] + '/subject_' + str(cps_para['subject_num']) + '/exp_num_' + str(cps_para['exp_num'])        
        self.impedance_path = cps_para['impedance_path'] + '/subject_' + str(cps_para['subject_num']) + '/exp_num_' + str(cps_para['exp_num'])        
        
        # Gaussian process regressor with an RBF kernel    
        # kernel = RBF(length_scale=2.0)    
        self.kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))     
        self.gp_model = GaussianProcessRegressor(kernel=self.kernel)       
        
        # set up all policy
        self.reps = GPREPS(
            context_dim=self.context_dim,       
            para_dim=self.para_dim,       
            para_lower_bound=0.0,    
            para_upper_bound=1.0,     
            eps=cps_para['epi']       
        )   

        self.iter_index = 0    
        self.reward = 0.0   
        
        self.beta = cps_para['beta']    
        self.num_sim = cps_para['num_sim']       
        self.num_real = cps_para['num_real']       
        self.num_space = cps_para['num_space']      

        self.task_var = None    
        self.para_range = None   
        self.context_range = cps_para['context_range']       

        #### real context ##### 
        # speed_range=np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),   
        # slop_range=np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])  
        # self.context_v = np.array([0.8, 1.0, 1.1, 1.2, 1.3, 1.4])    
        # self.context_s = np.array([-8, -4, -2, 0, 2, 4, 8])    
        self.speed_range, self.slop_range = np.array(self.context_range['speed']), np.array(self.context_range['slope'])  
        print(self.speed_range, self.slop_range)   
        self.speed_values, self.slop_values = self.generate_all()     
        

    def push_data(self, context_list=None, para_list=None, reward_list=None):     
        # reward_list = self.get_reward(context_list=context_list, para_list=-para_list)  
    
        self.cps_dataset.add(context_list, para_list, reward_list)   


    def get_reward(self, context_list=None, para_list=None):     
        assert context_list.shape[0] == para_list.shape[0]  
        assert context_list.shape[1] == self.context_dim  
        assert para_list.shape[1] == self.para_dim  

        reward_list = np.zeros((context_list.shape[0], 1))  
        return reward_list    
    

    def get_context(self, context_num=1):      
        """ 
            only use for training phase  
        """
        context_list = np.zeros((context_num, self.context_dim))  

        ####### random sample context from context space #########
        context_list = self.sample_real_context(context_num)     
        return context_list    


    def get_para(self, context_list=None):    
        assert context_list.shape[1] == self.context_dim  
        para_list = np.zeros((context_list.shape[0], self.para_dim))     

        if self.policy == 'bo':     
            # Generate the Upper Confidence Bound (UCB) using the Gaussian process model
            ucb = upper_confidence_bound(self.para_space, self.gp_model, self.beta)       

            # Select the next point based on UCB    
            para_list = self.para_space[np.argmax(ucb)][None, :]                  
 
        if self.policy == 'reps':   
            para_list = self.reps.get_para(context_list)    
        
        if self.policy == 'nn':   
            pass   

        return para_list    
    

    def load_policy(self, index=None):    
        policy_para = np.load(self.policy_path + "/reps_policy_para_" + str(index) + ".npz")    
        reps_obs_a, reps_obs_A, reps_obs_sigma = policy_para['arr_0'], policy_para['arr_1'], policy_para['arr_2'] 
        self.reps.set_para(reps_obs_a, reps_obs_A, reps_obs_sigma)   


    def update_policy(self, training_num=None):    
        state_list, reward_list = self.cps_dataset.sample_pairs(training_num)      
        self.gp_model.fit(state_list, reward_list)                 
        
        self.iter_index += 1  

        ################ for traing with artificial samples ######    
        context_info_sim = self.sample_sim_context(give_seed=self.iter_index)     
        para_info_sim = self.get_para(context_list=context_info_sim)          
        state_info_sim = np.hstack((context_info_sim, para_info_sim)).reshape(-1, self.context_dim + self.para_dim)   
 
        reward_info_sim, _ = self.gp_model.predict(state_info_sim, return_std=True)   
        reward_info_sim = reward_info_sim.reshape(-1, 1)    

        ########## update policy with artifical samples ##########  
        self.reps.update_policy(context_info_sim, para_info_sim, reward_info_sim)     

        save_policy_index_path = self.policy_path + '/eva_' + str(self.iter_index)     
        if not os.path.exists(save_policy_index_path):           
            os.makedirs(save_policy_index_path)      

        ################# save policy parameter ###################   
        np.savez(save_policy_index_path + "/reps_policy_para_" + str(self.iter_index) + ".npz", self.reps.a, self.reps.A, self.reps.sigma)     

        ################# generate new context  ###################
        context_list = self.sample_real_context(give_seed=self.iter_index, num_real=self.num_real)       

        ################ generate new impeance ####################  
        self.save_impedance(context_list=context_list)      


    def update_gp(self, training_num=None):    
        state_list, reward_list = self.cps_dataset.sample_pairs(training_num)      
        self.gp_model.fit(state_list, reward_list)    

        return self.gp_model.score(state_list, reward_list)      
        
    
    def generate_all(self, ):    
        speed_space_value = np.linspace(np.min(self.speed_range), np.max(self.speed_range), self.num_space)   
        slop_space_value = np.linspace(np.min(self.slop_range), np.max(self.slop_range), self.num_space) 

        speed_values, slop_values = np.meshgrid(speed_space_value, slop_space_value)          
        speed_values, slop_values = speed_values.flatten(), slop_values.flatten()     
        return speed_values, slop_values     


    def sample_sim_context(self, give_seed=None):      
        if give_seed is not None: 
            np.random.seed(give_seed)    
        
        context_sim = np.random.random(size=(self.num_sim, self.context_dim))    
        sim_index = np.random.randint(0, self.num_space * self.num_space, size=(self.num_sim,))  
        context_sim[:, 0] = self.speed_values[sim_index]   
        context_sim[:, 1] = self.slop_values[sim_index]   
        return context_sim   
    

    def sample_real_context(self, give_seed=None, num_real=None):            
        if give_seed is not None: 
            np.random.seed(give_seed)    
        
        ################### random #####################
        context_real = np.zeros((num_real, self.context_dim))     
        V, S = np.meshgrid(self.speed_range, self.slop_range)
        V, S = V.flatten(), S.flatten()  

        idx_chosen = np.random.randint(0, V.shape[0], size=(num_real,))   
        # print(idx_chosen)  
        context_real[:, 0] = V[idx_chosen]  
        context_real[:, 1] = S[idx_chosen]   

        ################## real ########################  
        real_index_slope = [0, 2, 4, 6, 8, -2, -4, -6, -8]    
        context_real[:, 1] = real_index_slope[self.iter_index]     
        return context_real     
    

    def cal_reward(self, context, para):  
        """ get reward  """  
        reward = 0.0   

        return reward    


    def save_impedance(self, context_list=None):      
        # TODO 生成新的impedance
        context_num = context_list.shape[0]   
        idx_knee = 0
        idx_ankle = 1  
        imp_total = np.zeros((context_num, 24))         

        para_list = self.get_para(context_list=context_list)     

        new_kp1 = para_list[:, 0]
        new_kp1 = scaling(new_kp1, lim_kp[idx_knee][1])
        new_kp2 = para_list[:, 1]   
        new_kp2 = scaling(new_kp2, lim_kp[idx_knee][2])   
        
        imp_total[:,0:8] = np.array(k0[idx_knee]+k0[idx_ankle])
        imp_total[:,8:16] = np.array(b0[idx_knee]+b0[idx_ankle])
        imp_total[:,16:] = np.array(q_e0[idx_knee]+q_e0[idx_ankle])
        imp_total[:, 1] = new_kp1[:]
        imp_total[:, 2] = new_kp2[:]    

        save_impedance_path = self.impedance_path + '/eva_' + str(self.iter_index)     
        if not os.path.exists(save_impedance_path):           
            os.makedirs(save_impedance_path)         
        
        if check_generated_imp(imp_total):  
            print("Save New Parameters in evaluation {}".format(self.iter_index))  
            np.save(self.impedance_path + "/eval_{}.npy".format(self.iter_index), imp_total)       



if __name__ == '__main__':     
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
    policy_path = current_dir + '/policy_para'    


    cps_para = {
        'context_dim': 2, 
        'para_dim': 2, 
        'policy_type': 'reps', 
        'epi' : 0.1, 
        'beta' : 1.0, 
        'num_sim' : 2000,  
        'num_real' : 10,  
        'num_space' : 200,  
        'policy_path': '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/scripts/cps_learning/policy_para',  
        'impedance_path': '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/scripts/cps_learning/impedance_para',   
        'subject_num': 0,  
        'exp_num': 0, 
        'context_range': {'speed': [0.8, 1.0, 1.1, 1.2, 1.3, 1.4], 'slope' : [-8, -4, -2, 0, 2, 4, 8]}   
    }
    context_range = {'speed': [0.8, 1.0, 1.1, 1.2, 1.3, 1.4], 'slope' : [-8, -4, -2, 0, 2, 4, 8]}     

    cps_policy = CPS(cps_para=cps_para)    

    # context_dim=args.context_dim, para_dim=args.para_dim, policy_type='reps', epi=0.1, 
    # policy_path=policy_path, num_sim=2000, num_space=200, context_range=context_range  
    
    # policy_path = '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/data/state_info_real420'  
    # for index in range(1, 10):   
    #     state_action_pair = np.load(policy_path + '/e' + str(index) + '/' + 'state_info_real.npy')    
    #     print(state_action_pair.shape)    

    
    policy_path = '/Users/zhimin/Desktop/2-code/HPS_Perception/hps_control/data/data420'   
    for index in range(1, 10):   
        path = policy_path + '/e' + str(index) + '/pair'  
        state_list = glob.glob(path + '/state_*.npy') 
        reward_list = glob.glob(path + '/reward_*.npy')    
        data_state = []  
        data_reward = []    
        for state_path, reward_path in zip(state_list, reward_list):  
            state_list = np.load(state_path)  
            reward_list = np.load(reward_path)  
            # print(state_list)  
            # print(reward_list)   
            data_state.append(state_list) 
            data_reward.append(reward_list)   

            cps_policy.push_data(context_list=state_list[:2], para_list=state_list[2:], reward_list=reward_list)    
        
        # data_state = np.array(data_state)  
        # data_reward = np.array(data_reward)  
        # # print(data_state.shape, data_reward.shape) 
        # cps_policy.push_data(context_list=data_state[:, :2], para_list=data_state[:, 2:4], reward_list=data_reward)   

        # state_action_pair = np.load(policy_path + '/e' + str(index) + '/pair/' + 'state_' + str(state_index) + '.npy')     
        # reward_pair = np.load(policy_path + '/e' + str(index) + '/pair/' + 'reward_' + str(state_index) + '.npy')   

    print("size of dataset :", cps_policy.cps_dataset._size)       

    score = cps_policy.update_gp(training_num=80)  
    print("score :", score)     

    cps_policy.update_policy(training_num=80)  
