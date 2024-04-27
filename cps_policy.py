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
    def __init__(self, context_dim, para_dim, max_size=int(1e3)):
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

    def sample_latest(self, batch_size):  
        return self._context_buf[-batch_size:, :], self._para_buf[-batch_size:, :], self._reward_buf[-batch_size:, :]  

    def save(self, save_dir):    
        if not os.path.exists(save_dir):            
            os.makedirs(save_dir)       
        save_dir = save_dir + '/state_reward_pair.npz' 
        np.savez(save_dir, context=self._context_buf[:self._size, :], para=self._para_buf[:self._size,:], reward=self._reward_buf[:self._size, :])        

    def load(self, data_dir):  
        self._context_buf, self._para_buf, self._reward_buf = None, None, None
        np_file = np.load(data_dir + '/state_reward_pair.npz')    
        
        self._context_buf = np_file['context']
        self._para_buf = np_file['para']   
        self._reward_buf = np_file['reward']    
        
        self._size  = self._context_buf.shape[0]
        self._ptr = self._size - 1

    # def load(self, data_dir):   
    #     self._context_buf, self._para_buf, self._reward_buf = None, None, None
    #     for i, data_file in enumerate(glob.glob(os.path.join(data_dir, "*.npz"))):
    #         size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
    #         np_file = np.load(data_file)
    #         #
    #         context_array = np_file['context'][:size, :]
    #         para_array = np_file['para'][:size, :]
    #         if i == 0:
    #             self._context_buf = context_array
    #             self._para_buf = para_array    
    #         else:
    #             self._context_buf = np.append(self._context_buf, context_array, axis=0)
    #             self._para_buf = np.append(self._para_buf, para_array, axis=0)
    #     #
    #     self._size  = self._context_buf.shape[0]
    #     self._ptr = self._size - 1


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
        self.data_pair_path = cps_para['data_pair_path'] + '/subject_' + str(cps_para['subject_num']) + '/exp_num_' + str(cps_para['exp_num'])     
        
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

        self.iter_index = cps_para['iter_index']     
        self.gait_index = 0  
        self.inter_reward = 0.0   
        
        self.beta = cps_para['beta']    
        self.num_sim = cps_para['num_sim']       
        self.num_real = cps_para['num_real']       
        self.num_space = cps_para['num_space']      

        self.task_var = None    
        self.para_range = None    
        self.context_range = cps_para['context_range']    

        self.batch_num = cps_para['batch_num']       
        self.batch_reward = np.zeros(self.batch_num)           

        #### real context ##### 
        # speed_range=np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),   
        # slop_range=np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])  
        # self.context_v = np.array([0.8, 1.0, 1.1, 1.2, 1.3, 1.4])    
        # self.context_s = np.array([-8, -4, -2, 0, 2, 4, 8])    
        self.speed_range, self.slop_range = np.array(self.context_range['speed']), np.array(self.context_range['slope'])  
        # print(self.speed_range, self.slop_range)   
        self.speed_values, self.slop_values = self.generate_all()   

        ### load previous data ### 
        self.load_data()  
        

    def push_pair(self, context_list=None, para_list=None, reward_list=None):     
        self.cps_dataset.add(context_list, para_list, reward_list)   


    def push_gait(self, state=None, reward=None):     
        self.batch_reward[self.gait_index%self.batch_num] = reward     
        self.gait_index += 1 
        if self.gait_index%self.batch_num == 0:  
            avg_reward = np.mean(self.batch_reward)  
            self.push_pair(context_list=state[:self.context_dim], para_list=state[self.context_dim:], reward_list=avg_reward)   


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

    def load_data(self,): 
        ################ load dataset of data pair ################
        data_pair_path = self.data_pair_path + '/eva_' + str(self.iter_index)   
        self.cps_dataset.load(data_pair_path)  
        print("successfully load data !!!")    
        print("datasize :", self.cps_dataset._size)           

    def update_policy(self, training_num=None):     
        state_list, reward_list = self.cps_dataset.sample_pairs(training_num)      
        self.gp_model.fit(state_list, reward_list)                   

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

        ################ save dataset of data pair ################
        data_pair_path = self.data_pair_path + '/eva_' + str(self.iter_index+1)         
        self.cps_dataset.save(data_pair_path)    


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
        context_real[:, 0] = V[idx_chosen]  
        context_real[:, 1] = S[idx_chosen]   

        ################## real ########################  
        real_index_slope = [0, 2, 4, 6, 8, -2, -4, -6, -8]    
        context_real[:, 1] = real_index_slope[self.iter_index]     
        return context_real     
    

    def cal_reward(self, batch_size=10):    
        batch_context, batch_para, batch_reward = self.sample_latest(batch_size)    
        avg_reward = np.mean(batch_reward)   
        std_reward = np.std(batch_reward)
        return avg_reward, std_reward      


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

        save_impedance_path = self.impedance_path + '/eva_' + str(self.iter_index+1)       
        if not os.path.exists(save_impedance_path):           
            os.makedirs(save_impedance_path)         
        
        if check_generated_imp(imp_total):  
            print("Save New Parameters in evaluation {}".format(self.iter_index))  
            np.save(self.impedance_path + "/impedance_{}.npy".format(self.iter_index), imp_total)      
            np.save(self.impedance_path + "/context_{}.npy".format(self.iter_index), context_list)        

        for send_index in range(context_list.shape[0]):    
            print("Please Enter to continue ...")
            input()   
            # publish one impedance # 

            print('Send ' + str(send_index) + ' impedance parameters !!!')   

        return imp_total  