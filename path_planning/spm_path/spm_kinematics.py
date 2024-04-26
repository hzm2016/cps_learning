import os, sys 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
import numpy as np 
import matplotlib.pyplot as plt  
from matplotlib import cm   
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.ticker import LinearLocator  
import copy as cp 
import argparse  

from path_planning.spm_path.cal_utils import *    

radian_to_angle = 180.0/np.pi    
angle_to_radian = np.pi/180.0     


class SPM_FIK(object):    
    def __init__(self, beta=60, alpha_1=45, alpha_2=45):      
        self.dof = 3  
        self.R = 70  
        self.R_1 = 70 * np.sqrt(3)   
        self.R_2 = 140   
        self.center_point = np.array([0.0, 0.0, -70.0])   
        self.alpha_1 = alpha_1 * angle_to_radian   
        self.alpha_2 = alpha_2 * angle_to_radian   
        self.beta = beta * angle_to_radian  
        self.alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))     
        
        # initial position   
        self.eta_list = np.array([0.0, 120.0, 240.0])  
        
        # configuration_space 
        self.theta_list = np.array([-54.73, -54.73, -54.73])   
        
        # task space  
        self.ee_euler = np.array([0.0, 0.0, 0.0])   
        
    def cal_w(  
        self, 
        theta_raw=np.array([-54.0, -54.0, -54.0])  
    ):   
        eta = np.array([0, 120, 240])    
        theta = (eta + theta_raw) * angle_to_radian      

        w_list = np.zeros((3, 3))     
        for i in range(3):      
            w_list[i, :] = np.array([np.sin(self.alpha_1) * np.cos(theta[i]), np.sin(self.alpha_1) * np.sin(theta[i]),  -1 * np.cos(self.alpha_1)])  
        return w_list       

    def cal_v_list(
        self, n, sigma, v_t_list
    ): 
        # # /// 
        v_t_list_r = np.zeros((3, 3))  
        for i in range(v_t_list.shape[0]):    
            v_t_1 = v_t_list[i, :] 
            v_t_list_r[i, :] = v_t_1 * np.cos(sigma) + np.cross(v_t_1, n) * np.sin(sigma) + n * (n.dot(v_t_1)) * (1 - np.cos(sigma))
        
        return v_t_list_r       
    
    def cal_FF(
        self, theta_raw=np.array([-54, -54, -54]),   
        x_list=np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5]) 
    ): 
        # //////////  
        # input joint angle deg   
        # eta = np.array([0, 120, 240]) # deg     
        # theta = (eta - theta_raw) * angle_to_radian     
        
        F = np.zeros(9)   
        w_list = np.zeros((3, 3))      
        w_list = cal_w(theta_raw)    
        # for i in range(3):      
        #     w_list[i, :] = np.array([np.sin(alpha_1) * np.cos(theta[i]), np.sin(alpha_1) * np.sin(theta[i]),  -1 * np.cos(alpha_1)])  
        
        v_t_1 = x_list[:3]   
        v_t_2 = x_list[3:6]   
        v_t_3 = x_list[6:]      
        w_t_1 = w_list[0, :]    
        w_t_2 = w_list[1, :]    
        w_t_3 = w_list[2, :]    

        F[0] = v_t_1[0]**2 + v_t_1[1]**2 + v_t_1[2]**2 - 1
        F[1] = v_t_2[0]**2 + v_t_2[1]**2 + v_t_2[2]**2 - 1 
        F[2] = v_t_3[0]**2 + v_t_3[1]**2 + v_t_3[2]**2 - 1  
        F[3] = w_t_1.transpose().dot(v_t_1) - np.cos(self.alpha_2)   
        F[4] = w_t_2.transpose().dot(v_t_2) - np.cos(self.alpha_2)   
        F[5] = w_t_3.transpose().dot(v_t_3) - np.cos(self.alpha_2)   
        F[6] = v_t_1.transpose().dot(v_t_2) - np.cos(self.alpha_3)   
        F[7] = v_t_2.transpose().dot(v_t_3) - np.cos(self.alpha_3)   
        F[8] = v_t_3.transpose().dot(v_t_1) - np.cos(self.alpha_3)   

        return F   
    
    def cal_J(
        self, theta_raw=np.array([-54, -54, -54]),  
        x_list = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5])  
    ):
        J = np.zeros((9,9))  
        
        F = np.zeros(9)   
        w_list = np.zeros((3,3))      
        w_list = cal_w(w_list)     
        
        v_t_1 = x_list[:3]     
        v_t_2 = x_list[3:6]    
        v_t_3 = x_list[6:]     

        w_t_1 = w_list[0, :]    
        w_t_2 = w_list[1, :]    
        w_t_3 = w_list[2, :]    

        J[0, 0] = 2 * v_t_1[0]
        J[0, 1] = 2 * v_t_1[1] 
        J[0, 2] = 2 * v_t_1[2]

        J[1, 3] = 2 * v_t_2[0]
        J[1, 4] = 2 * v_t_2[1] 
        J[1, 5] = 2 * v_t_2[2] 

        J[2, 6] = 2 * v_t_3[0] 
        J[2, 7] = 2 * v_t_3[1]  
        J[2, 8] = 2 * v_t_3[2] 

        J[3, 0] = w_t_1[0] 
        J[3, 1] = w_t_1[1]  
        J[3, 2] = w_t_1[2]  

        J[4, 3] = w_t_2[0]
        J[4, 4] = w_t_2[1] 
        J[4, 5] = w_t_2[2]

        J[5, 6] = w_t_3[0]
        J[5, 7] = w_t_3[1] 
        J[5, 8] = w_t_3[2]

        J[6, 0] = v_t_2[0]
        J[6, 1] = v_t_2[1] 
        J[6, 2] = v_t_2[2]
        J[6, 3] = v_t_1[0]
        J[6, 4] = v_t_1[1] 
        J[6, 5] = v_t_1[2]

        J[7, 3] = v_t_3[0]
        J[7, 4] = v_t_3[1] 
        J[7, 5] = v_t_3[2]
        J[7, 6] = v_t_2[0]
        J[7, 7] = v_t_2[1] 
        J[7, 8] = v_t_2[2]  

        J[8, 0] = v_t_3[0]   
        J[8, 1] = v_t_3[1]   
        J[8, 2] = v_t_3[2]    
        J[8, 6] = v_t_1[0]   
        J[8, 7] = v_t_1[1]   
        J[8, 8] = v_t_1[2]   

        return J   
    
    def cal_FK( 
        self, 
        theta_raw=np.array([-54, -54, -54]), 
        x_0 = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5])      
    ):  
        w_list = cal_w(theta_raw)    

        w_t_1 = w_list[0, :]    
        w_t_2 = w_list[1, :]    
        w_t_3 = w_list[2, :]    

        v_t_1 = np.zeros(3)     
        v_t_2 = np.zeros(3)     
        v_t_3 = np.zeros(3)     
        J = np.zeros((9,9))    
        F = np.zeros(9)    

        Num_iteration = 100 

        # x_1 :: [ 0.70712053  0.49999028 -0.49999028 -0.78655729  0.36241644 -0.49998195 0.07942406 -0.86235958 -0.50002777]
        # x_0 = np.array([1, 0, 0, 0, 1, 0, -1, -1, -1])   
        
        # x_0 = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5])   
        x_1 = x_0   
        for i in range(Num_iteration):   
            x_0 = x_1.copy()   
            v_t_1 = x_0[:3]   
            v_t_2 = x_0[3:6]  
            v_t_3 = x_0[6:]  

            J[0, 0] = 2 * v_t_1[0]
            J[0, 1] = 2 * v_t_1[1] 
            J[0, 2] = 2 * v_t_1[2]

            J[1, 3] = 2 * v_t_2[0]
            J[1, 4] = 2 * v_t_2[1] 
            J[1, 5] = 2 * v_t_2[2] 

            J[2, 6] = 2 * v_t_3[0] 
            J[2, 7] = 2 * v_t_3[1]  
            J[2, 8] = 2 * v_t_3[2] 

            J[3, 0] = w_t_1[0] 
            J[3, 1] = w_t_1[1]  
            J[3, 2] = w_t_1[2]  

            J[4, 3] = w_t_2[0]
            J[4, 4] = w_t_2[1] 
            J[4, 5] = w_t_2[2]

            J[5, 6] = w_t_3[0]
            J[5, 7] = w_t_3[1] 
            J[5, 8] = w_t_3[2]

            J[6, 0] = v_t_2[0]
            J[6, 1] = v_t_2[1] 
            J[6, 2] = v_t_2[2]
            J[6, 3] = v_t_1[0]
            J[6, 4] = v_t_1[1] 
            J[6, 5] = v_t_1[2]

            J[7, 3] = v_t_3[0]
            J[7, 4] = v_t_3[1] 
            J[7, 5] = v_t_3[2]
            J[7, 6] = v_t_2[0]
            J[7, 7] = v_t_2[1] 
            J[7, 8] = v_t_2[2]  

            J[8, 0] = v_t_3[0]
            J[8, 1] = v_t_3[1] 
            J[8, 2] = v_t_3[2]
            J[8, 6] = v_t_1[0] 
            J[8, 7] = v_t_1[1] 
            J[8, 8] = v_t_1[2] 

            F[0] = v_t_1[0]**2 + v_t_1[1]**2 + v_t_1[2]**2 - 1
            F[1] = v_t_2[0]**2 + v_t_2[1]**2 + v_t_2[2]**2 - 1 
            F[2] = v_t_3[0]**2 + v_t_3[1]**2 + v_t_3[2]**2 - 1  
            F[3] = w_t_1.transpose().dot(v_t_1) - np.cos(self.alpha_2) 
            F[4] = w_t_2.transpose().dot(v_t_2) - np.cos(self.alpha_2) 
            F[5] = w_t_3.transpose().dot(v_t_3) - np.cos(self.alpha_2) 
            F[6] = v_t_1.transpose().dot(v_t_2) - np.cos(self.alpha_3) 
            F[7] = v_t_2.transpose().dot(v_t_3) - np.cos(self.alpha_3) 
            F[8] = v_t_3.transpose().dot(v_t_1) - np.cos(self.alpha_3)  

            # print("J ::\n", J)
            # print("J inverse ::\n", np.linalg.pinv(J))  
            # print("J inverse ::\n", np.linalg.inv(J.transpose().dot(J)).dot(J.transpose())) 
            x_1 = x_0 - np.linalg.pinv(J).dot(F);  
            if np.linalg.norm(x_1 -  x_0) < 0.001:  
                break  
        
        v_t_1 = x_0[:3]  
        v_t_2 = x_0[3:6]   
        v_t_3 = x_0[6:]   
        
        print("index :", i)   
        # print("v_t_1 :", v_t_1)  
        # print("v_t_2 :", v_t_2)   
        # print("v_t_3 :", v_t_3)   
        # print("x_1 :", x_1)   
        n = (v_t_1 + v_t_2 + v_t_3)/np.linalg.norm(v_t_1 + v_t_2 + v_t_3)    
        
        return n   
    
    def cal_IK(
        self,   
        v_t_1=None,   
        v_t_2=None,   
        v_t_3=None    
    ):          
        v_list = np.zeros((3, 3))    
        v_list[0, :] = vec_normalized(v_t_1)   
        v_list[1, :] = vec_normalized(v_t_2)    
        v_list[2, :] = vec_normalized(v_t_3)      
        self.eta_list = np.array([0., 120, 240])   
        theta_list = np.zeros(3)  
        for i in range(3):   
            v_t_1 = v_list[i, :]  
            eta = self.eta_list[i] * angle_to_radian     
            A = -np.cos(eta) * np.sin(self.alpha_1) * v_t_1[0] - np.sin(eta) * np.sin(self.alpha_1) * v_t_1[1] - np.cos(self.alpha_1) * v_t_1[2] - np.cos(self.alpha_2)
            B = np.sin(eta) * np.sin(self.alpha_1) * v_t_1[0] - np.cos(eta) * np.sin(self.alpha_1) * v_t_1[1]
            C = np.cos(eta) * np.sin(self.alpha_1) * v_t_1[0] + np.sin(eta) * np.sin(self.alpha_1) * v_t_1[1] - np.cos(self.alpha_1) * v_t_1[2] - np.cos(self.alpha_2)

            x_1 = (-2 * B + np.sqrt(4 * pow(B, 2) - 4 * A * C))/2/A   
            x_2 = (-2 * B - np.sqrt(4 * pow(B, 2) - 4 * A * C))/2/A   
            theta_1 = np.arctan(x_1) * 2 * radian_to_angle   
            theta_2 = np.arctan(x_2) * 2 * radian_to_angle   
            
            # print("theta index :", i)  
            # print("theta_1 :", theta_1)     
            # print("theta_2 :", theta_2)     

            theta_list[i] = -1 * theta_2   
            # theta_list[i] = theta_2  
            # theta_list[i] = -1 * theta_1  
       
        # print("theta_list :", theta_list) 
        return theta_list   

    def cal_n(
        self, 
        v_list=None
    ):  
        v_t_1 = v_list[0, :]  
        v_t_2 = v_list[1, :]  
        v_t_3 = v_list[2, :]   
        n = (v_t_1 + v_t_2 + v_t_3)/np.linalg.norm(v_t_1 + v_t_2 + v_t_3)    
        return n 
        

class TwoSEAs(object):       
    def __init__(self, L_1=0.35, L_2=0.35):              
        self.dof = 2    
        self.joints = np.array([0.0, 0.0])    
        self.ee_position = np.array([0.0, 0.0])       
        self.L_1 = L_1     
        self.L_2 = L_2     
        self.J = np.array((self.dof, self.dof))     
        
    def cal_fk(self, theta=np.array([0.0,0.0])): 
        ee_position = np.zeros_like(theta)   
        ee_position[:, 0] = self.L_1 * np.cos(theta[:, 0]) + self.L_2 * np.cos(theta[:, 0] + theta[:, 1])        
        ee_position[:, 1] = self.L_1 * np.sin(theta[:, 0]) + self.L_2 * np.sin(theta[:, 0] + theta[:, 1])         
        return ee_position               
    
    def cal_ik(self, ee_position=np.array([0.0, 0.0])):   
        joints_list = []       
        for index in range(ee_position.shape[0]):          
            x1 = ee_position[index, 0]         
            x2 = ee_position[index, 1]          
            
            ### calculate angle 1    
            L = x1**2 + x2**2       
            gamma = np.arctan2(x2, x1)       
            
            if np.sqrt(L) > self.L_1 + self.L_2 or np.sqrt(L) < np.abs(self.L_1 - self.L_2):    
                exit()     
            else:  
                cos_belta = (self.L_1**2 + L - self.L_2**2)/(2 * self.L_1 * np.sqrt(L))  
                if cos_belta > 1:     
                    angle_1 = gamma     
                elif  cos_belta < -1:     
                    angle_1 = gamma - np.pi      
                else:    
                    angle_1 = gamma - np.arccos(cos_belta)     
                
                ### calculate angle 2     
                cos_alpha = (self.L_1**2 - L + self.L_2**2)/(2 * self.L_1 * self.L_2)   
                
                if cos_alpha > 1:     
                    angle_2 = np.pi      
                elif  cos_alpha < -1:       
                    angle_2 = 0       
                else:    
                    angle_2 = np.pi - np.arccos(cos_alpha)      
                
                joints = np.array([angle_1, angle_2])
                # + np.array([np.pi/2.0, -np.pi])  
            joints_list.append(joints)     
        return np.array(joints_list)     
    
    def inverse_kinematics(self, ee_position=None):  
        import math 
        joints_list = []       
        for index in range(ee_position.shape[0]):   
            x = ee_position[index, 0]     
            y = ee_position[index, 1]      
            d = (x ** 2 + y ** 2 - self.L_1 ** 2 - self.L_2 ** 2) / (2 * self.L_1 * self.L_2)     
            theta_2 = math.atan2(math.sqrt(1 - d ** 2), d)  
            theta_1 = math.atan2(y, x) - math.atan2(self.L_2 * math.sin(theta_2), self.L_1 + self.L_2 * math.cos(theta_2))    
            joints = np.array([theta_1, theta_2])     
            joints_list.append(joints)     
        return np.array(joints_list)     
    
    def cal_J(self, theta=np.array([-54, -54])):   
        self.J[0] = -self.L_1 * np.sin(theta[0]) - self.L_2 * np.sin(theta[0] + theta[1])
        self.J[1] = -self.L_2 * np.sin(theta[0] + theta[1])     
        self.J[2] = self.L_1 * np.cos(theta[0]) + self.L_2 * np.cos(theta[0] + theta[1])    
        self.J[3] = self.L_2 * np.cos(theta[0] + theta[1])              
        return self.J   
    
    def cal_circle(self, circle_center=np.array([0.5, 0.12]), radius=0.08, T=10, T_s=0.001):  
        t_list = np.linspace(0.0, T, int(T/T_s))  
        x_t = circle_center[0] + radius * np.cos(2 * np.pi * 1.0/T * t_list)
        y_t = circle_center[1] + radius * np.sin(2 * np.pi * 1.0/T * t_list)   
        
        path_list = np.hstack([x_t[:, None], y_t[:, None]])  
        print("path list :", path_list.shape)   
        # joint_list = self.cal_ik(ee_position=path_list)   
        joint_list = self.inverse_kinematics(ee_position=path_list)       
        return path_list, joint_list  

        
    # def cal_dy(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    #     # Calculate the required joint torques/forces for the given motion  
    #     # Inertia matrix
    #     M11 = (m1 * L1**2 + m2 * (L1**2 + L2**2 + 2 * L1 * L2 * np.cos(theta2)))  
    #     M12 = (m2 * (L2**2 + L1 * L2 * np.cos(theta2)))
    #     M21 = (m2 * (L2**2 + L1 * L2 * np.cos(theta2)))
    #     M22 = (m2 * L2**2)   
    #     M = np.array(
    #         [[M11, M12],
    #          [M21, M22]]
    #     )

    #     # Coriolis and centrifugal forces
    #     C1 = (-m2 * L1 * L2 * np.sin(theta2) * (2 * dtheta1 * dtheta2 + dtheta2**2))
    #     C2 = (m2 * L1 * L2 * np.sin(theta2) * dtheta1**2)
    #     C = np.array([C1, C2])

    #     # Gravity forces
    #     G1 = ((m1 * L1 + m2 * L1) * g * np.sin(theta1) + m2 * L2 * g * np.sin(theta1 + theta2))
    #     G2 = (m2 * L2 * g * np.sin(theta1 + theta2))
    #     G = np.array([G1, G2])

    #     # Joint torques/forces
    #     tau1 = M11 * ddtheta1 + M12 * ddtheta2 + C1 + G1
    #     tau2 = M21 * ddtheta1 + M22 * ddtheta2 + C2 + G2

    #     return tau1, tau2



def generate_sphere_reference(
    center_ori=[0, 0, 0],      
    spm_fik=None,       
	sample_num_1=[1000, 1000],         
	sample_num_2=50,       
	index=0,      
	theta_range=[7/8 * np.pi, 9/8*np.pi],     
	phi_range=[np.pi/2,np.pi/2],      
	psi=10,    
    save_fig=True     
):  
    center = np.zeros(3)    
    theta = np.linspace(theta_range[0], theta_range[1], sample_num_1[0])       
    phi = np.linspace(phi_range[0], phi_range[1], sample_num_1[1])     
    t, p = np.meshgrid(theta, phi)    

    # c1 = center[0] * np.ones((sample_num_1, 1))     
    # c2 = center[1] * np.ones((sample_num_1, 1))      
    # c3 = center[2] * np.ones((sample_num_1, 1))      

    # ////////////////////////////////////////
    r = spm_fik.R   
    x = r*np.sin(t)*np.cos(p)     
    y = r*np.sin(t)*np.sin(p)    
    z = r*np.cos(t)     

    x = x.flatten()     
    y = y.flatten()     
    z = z.flatten()     

    sample_point = []     
    theta_t_list = []     
    trajectory_theta_list = []  

    n_1_list = []    
    n_2_list = []    
    n_3_list = []    
    center_point = []     

    forward_n_list = []     
    true_n_list = []      

    sample_num = sample_num_1[0] * sample_num_1[1]      
    print("sample_num :", sample_num)      

    psi_list = psi * np.ones(sample_num)      

    q_list = np.zeros(3)   
    v_list = np.zeros((3, 3))     
    v_list_all = []    
     
    J = np.zeros((3, 3))     
    x_0 = np.zeros(9)      

    ee_torque = np.array([0, 0, 1])     

    input_torque_list = []    

    psi_t_list = []      
    for i in range(sample_num):        
        n = np.array([x[i], y[i], z[i]])        
        true_n_list.append(cp.deepcopy(vec_normalized(n)))        
        center_point.append(n)       
        
        psi_t = psi_list[i]      
        point_scatter, n_1, n_2, n_3 = get_3D_point(r=spm_fik.R_1, center=n, n=n, sample=np.array([0+psi_t, 120+psi_t, 240+psi_t]))       
        
        v_list[0, :] = cp.deepcopy(n_1)         
        v_list[1, :] = cp.deepcopy(n_2)        
        v_list[2, :] = cp.deepcopy(n_3)        
        v_list_all.append(cp.deepcopy(v_list))         
        
        q_list = spm_fik.cal_IK(v_t_1=v_list[0, :], v_t_2=v_list[1, :], v_t_3=v_list[2, :])    
        
        # print("q_list :", q_list, q_list_mirror)    
        trajectory_theta_list.append(cp.deepcopy(q_list))         
        
        # x_0[:3] = n_1/np.linalg.norm(n_1)       
        # x_0[3:6] = n_2/np.linalg.norm(n_2)       
        # x_0[6:] = n_3/np.linalg.norm(n_3)        

        # # faward kinematics   
        # theta_t, psi_t = cal_taskspace_angle(n/np.linalg.norm(n), x_0)   
        # theta_t_list.append(theta_t)   
        # psi_t_list.append(psi_t)   
        
        # # faward kinematics  
        # forward_n = spm_fik.cal_FK(theta_raw=q_list)      
        # forward_n_list.append(forward_n)      

        # # force simulate 
        # J = cal_Jacobian(   
        # 	theta_raw=q_list,     
        # 	v_list=v_list     
        # )   

        # input_torque_list.append(np.dot(J, ee_torque))    
        # np.array(input_torque_list)    
    
    center_point = np.array(center_point)       
    v_list_all = np.array(v_list_all)       
    
    return np.array(trajectory_theta_list), np.array(forward_n_list), np.array(true_n_list), np.array(theta_t_list), v_list_all, center_point  


if __name__ == "__main__":     
    parser = argparse.ArgumentParser()      
    
    # //// path /////////////  
    parser.add_argument('--mode', type=str, default="zero_force", help='choose mode first !!!!')   
    parser.add_argument('--index', type=int, default=0, help='choose mode first !!!!')    
    parser.add_argument('--game', type=int, default=0, help='choose mode first !!!!')    
    parser.add_argument('--flag', type=str, default="22", nargs='+', help='choose index first !!!!')     
    parser.add_argument('--save_fig', type=bool, default=True, help='choose index first !!!!')    
    parser.add_argument('--root_path', type=str, default='./data', help='choose index first !!!!')     
    parser.add_argument('--file_list', type=str, default='2023_05_09_09_28', nargs='+', help='choose index first !!!!')   
    parser.add_argument('--label_list', type=str, default='2023_05_09_09_28', nargs='+', help='choose index first !!!!')   
    parser.add_argument('--fig_path', type=str, default='./soft-tase', help='choose index first !!!!')     
    # //// basics ///////////  
    parser.add_argument('--num', type=int, default=3, help='choose index first !!!!')       
    parser.add_argument('--iter', type=int, default=1, help='choose index first !!!!')       
    parser.add_argument('--ee_force', type=int, default=0, help='select from {1, 2, 3}')         
    parser.add_argument('--policy', type=str, default='SAC', help='select from {1, 2, 3}')    

    parser.add_argument('--resample', type=int, default=250, help='select from {1, 2, 3}')        

    args = parser.parse_args()   
    
    ##################################################  
    # time_t, force_x_t, force_y_t, q_d_1, q_d_2, q_t_1, q_t_2 = load_all_data(file_list=['./data/online_tase/data_105.mat'], args=args)   
    # # time_t, force_x_t, force_y_t, q_d_1, q_d_2, q_t_1, q_t_2 = load_data(file_list=['./data/online_tase/data_112.mat'], args=args)  
    # # time_t, damping_1_t, damping_2_t, stiff_1_t, stiff_2_t, tau_1_t, tau_2_t, tau_o_1_t, tau_o_2_t = load_force_data(args=args)   
    
    # x_t, y_t = forward_kinematics(angle_t_1=-1 * q_t_1, angle_t_2=-1 * q_t_2)    
    # x_e, y_e = forward_kinematics(angle_t_1=-1 * q_d_1, angle_t_2=-1 * q_d_2)    
    
    # plot_figure_tase_11(args=args,   
    #                     x_e=x_e[:100], y_e=y_e[:100],    
    #                     x_t=x_t[:100], y_t=y_t[:100],    
    #                     obs_center=np.array([0.0,0.0]),    
    #                     via_points=None,  
    #                     start=np.array([0.6, 0.2]),    
    #                     end=np.array([0.5, 0.1])  
    #                     )   
    
    twoDofs = TwoSEAs()  
    curr_circle_center = np.array([0.50, 0.10])  
    path_list, joint_list = twoDofs.cal_circle(circle_center=curr_circle_center, radius=0.08, T=5, T_s=0.001)     
    
    save_data = np.hstack([joint_list, path_list])     
    print("save_data :", save_data.shape)     
    np.savetxt("data/wrist_demo/demo_data/trajectory_theta_list_new_circle.txt", save_data, fmt='%f', delimiter=',')     
    
    ik_path_list = twoDofs.cal_fk(joint_list)    
    
    plot_figure_tase_11(   
        args,  
        path_list[:, 0], path_list[:, 1],    
        ik_path_list[:, 0], ik_path_list[:, 1],    
        circle_center=curr_circle_center,     
        q_list=joint_list,    
        obs_center=None, via_points=None,      
        start=path_list[0, :],    
        end=path_list[1000, :]    
        )   
    
    plot_figure_tase_5(
        args=args, 
        time_t=None, 
        q_d_1=joint_list[:, 0],    
        q_d_2=joint_list[:, 1],    
        q_t_1=joint_list[:, 0],    
        q_t_2=joint_list[:, 1]     
    )   
    
    ##################################################
    ################### spm_path #####################  
    # spm_fik_left = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)   
    # spm_fik_right = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)   
             
    # # r = spm_fik.R      
    # # r_1 = spm_fik.R_1     
     
    # # sample_num_1=[1, 5000]  
    # sample_num_1=[1, 100]      
    # # sample_num_1=[2000, 1]       
    # # sample_num_1=[1, 5000]        
    # # sample_num_1=[4000, 1]        
    # # sample_num_1=[2000, 1]     
       
    # sample_num_2=50         
    
    # # theta_range=[7/8*np.pi,9/8*np.pi]   
    # # theta_range=[8/9*np.pi,10/9*np.pi]   
    # # theta_range=[np.pi,7/8*np.pi]    
    # # theta_range=[np.pi,9/8*np.pi]     
    # # theta_range=[0.0,0.0]  
    # # theta_range=[np.pi,10/9*np.pi]     
    # # theta_range=[np.pi,np.pi]  
    # # theta_range=[7/8 * np.pi, 9/8*np.pi]       
    # # theta_range=[7/8*np.pi, 9/8*np.pi]      
    # # theta_range=[8/9 * np.pi, 8/9*np.pi]     
    # # theta_range=[6/8 * np.pi, 10/8*np.pi]    
    # # theta_range=[7/8*np.pi,9/8*np.pi]  
    # theta_range=[10/9*np.pi,10/9*np.pi]  
    
    # # phi_range=[0,0]   
    # # phi_range=[np.pi*3/4,np.pi*3/4]   
    # # phi_range=[np.pi,np.pi]     
    # # phi_range=[3/2*np.pi,3/2*np.pi]   
    # # phi_range=[2*np.pi,2*np.pi]
    # # phi_range=[0,1/2*np.pi]   
    # # phi_range=[0,np.pi]   
    # phi_range=[0,2*np.pi]  
    # # phi_range=[np.pi/4.0,np.pi/4.0]      
    # # phi_range=[-np.pi/2,-np.pi/2]      
    # # phi_range=[-np.pi,-np.pi]      
    # # phi_range=[np.pi/2,np.pi/2]      
    # # phi_range=[np.pi*3.0/4.0,np.pi*3.0/4.0]       
    
    # psi=-54.73561     
    # psi=0.0     
    # plot_index=0      

    # trajectory_theta_list, forward_n_list, true_n_list, theta_t_list, v_list_all, center_point = generate_sphere_reference(
    #     [0, 0, 0],    
    #     spm_fik_left,    
    #     sample_num_1=sample_num_1,       
    #     sample_num_2=sample_num_2,     
    #     index=plot_index,     
    #     theta_range=theta_range,       
    #     phi_range=phi_range,     
    #     psi=psi    
    # )   
    # print("trajectory_theta_list :", trajectory_theta_list)  
    # # center_point_mirror, v_list_mirror_all, trajectory_theta_list_mirror = cal_mirror_path(spm_fik=spm_fik_right, v_list_all=v_list_all, sample_num_2=sample_num_2) 
    
    # plot_single_spm_path(
    #     spm_fik=spm_fik_left,    
    #     center_point=center_point,     
    #     v_list=v_list_all,     
    #     sample_num_2=sample_num_2,     
    #     plot_index=plot_index,     
    #     save_fig=True       
    # )  
    
    # plot_vector_comparison(
    #     trajectory_theta_list,   
    #     trajectory_theta_list  
    # )   
    
    # trajectory_theta_list = np.tile(np.array([-74.0, -74.0, -74.0]), (sample_num_1[0], 1))   
    # trajectory_theta_list = np.tile(np.array([-74.0, -74.0, -74.0]), (sample_num_1[0], 1)) 
    
    # if args.save:   
    #     # flag = "circ_5"     
    #     args.flag = "y_2"                  
    #     np.savetxt("../data/wrist_demo/demo_data/trajectory_theta_list_" + args.flag + ".txt", trajectory_theta_list, fmt='%f', delimiter=',')    
    
    # y_n = np.loadtxt("../data/wrist_demo/demo_data/trajectory_theta_list_" + flag + ".txt", delimiter=',')     
    # flag = "y_p"   
    # y_p = np.loadtxt("../data/wrist_demo/demo_data/trajectory_theta_list_" + flag + ".txt", delimiter=',')     
    
    # print("y_n :", y_n)    
    # y_n_inverse = y_n[::-1, :]   
    # print("y_n :", y_n_inverse)   
    # print("y_p :", y_p)  
    
    # y = np.concatenate((y_n_inverse, y_p), axis=0)   
    # print(y.shape)  
    # np.savetxt("../data/wrist_demo/demo_data/trajectory_theta_list_" + flag + ".txt", trajectory_theta_list, fmt='%f', delimiter=',')   
    
    # # /// plot path /// 
    # plot_spm_path(
    #     center_point=center_point,     
    #     v_list=v_list_all, 
    #     center_point_mirror=center_point_mirror, 
    #     v_list_mirror=v_list_mirror_all, 
    #     plot_index=4, 
    #     spm_fik=spm_fik_left     
    # )   
    
    # print("n_list :", n_1_list, n_2_list, n_3_list)   
    
    # for i in range(5):   
    #     v_t_1 = vec_normalized(n_1_list[i, :] * np.array([1, 1, 1]))    
    #     v_t_2 = vec_normalized(n_2_list[i, :] * np.array([1, 1, 1]))    
    #     v_t_3 = vec_normalized(n_3_list[i, :] * np.array([1, 1, 1]))      
    #     print("v_t_1 :", v_t_1, "v_t_2 :", v_t_2, "v_t_3 :", v_t_3)    
    #     print(spm_fik.cal_IK(v_t_1 =v_t_1, v_t_2=v_t_2, v_t_3=v_t_3))       
    
    # v_t_1 = vec_normalized(np.array([1.21243557e+02, -2.67878403e+01, -6.46715673e+01]))   
    # v_t_2 = vec_normalized(np.array([-60.62177826,  70.21951065, -104.85332767]))   
    # v_t_3 = vec_normalized(np.array([-60.62177826, -123.79519118, -24.48980688]))   

    # n_1_list = np.array([[1.21243557e+02, 8.57252759e-15, -7.00000000e+01],
    #                       [1.21243557e+02, -6.86119982e+00, -6.96629309e+01],
    #                       [1.21243557e+02, -1.36563225e+01, -6.86549696e+01],
    #                       [1.21243557e+02, -2.03199274e+01, -6.69858235e+01],
    #                       [1.21243557e+02, -2.67878403e+01, -6.46715673e+01]])
    # n_2_list = np.array([[-60.62177826, 105., -70. ],
    #                      [-60.62177826, 97.63319648, -79.9547306 ],
    #                      [-60.62177826, 89.3261319, -89.13945344],
    #                      [-60.62177826, 80.15880784, -97.46571461],
    #                      [-60.62177826, 70.21951065, -104.85332767]])  
    # n_3_list = np.array([[-60.62177826, -105., -70.], 
    #                      [-60.62177826, -111.35559612, -59.37113113],
    #                      [-60.62177826, -116.63877698, -48.17048582],
    #                      [-60.62177826, -120.79866266, -36.50593239],
    #                      [-60.62177826, -123.79519118, -24.48980688]])  
    
    # mirror_array = cal_mirror_motin(np.array([[1, 1.0], [2.0, 2.0]]))  
    # print(mirror_array)  

	# trajectory_theta_list = trajectory_theta_list - np.array([-54.73, -54.73, -54.73]) 
 

    # time_index = np.linspace(0.0, 10.0, 2000)  
    # ampl_angle = 20 
    # fre_angle = 0.1  
    # trajectory_theta_list = np.zeros((time_index.shape[0], 3)) 
    # for i in range(3): 
    #     trajectory_theta_list[:, i] = ampl_angle * np.sin(2 * np.pi * fre_angle * time_index) 
    #     # trajectory_theta_list = single_joint_ref
    
    # plot_vector_comparison(  
	# 	trajectory_theta_list,  
	# 	trajectory_theta_list  
	# )    
    
    # r = 70    
	# r_1 = r * np.sqrt(3)    
	# psi_t = 0    
	# index = 100
	# n = true_n_list[index]    
	# print("n: ", n)    
	# point_scatter, n_1, n_2, n_3 = get_3D_point(r=r_1, center=n, n=n, sample=np.array([0+psi_t, 120+psi_t, 240+psi_t]))     

	# psi = -90*np.pi/180   
	# n = n/np.linalg.norm(n)    
	# n_2_t = n_1 * np.cos(psi) + np.cross(n, n_1) * np.sin(psi) + n * (np.dot(n, n_1)) * (1 - np.cos(psi))    

	# n_1_t = n_1_list[index]   
	# print("n_1_t :", n_1_t, "n_1 :", n_1, "n_2_list :", n_2_t)    
	# print(cal_angle_two_vectors(n_2_t, n_1))   
	# print(cal_angle_two_vectors(n_2_t-true_n_list[index], n))   
	# print(cal_angle_two_vectors(n_1-true_n_list[index], n))     
	# print(cal_angle_two_vectors(np.array([0, 1, 0]), np.array([1, 0, 0])))      

	# fig = plt.figure(figsize=(8, 8))     
	# ax = fig.gca(projection='3d')     

	# n = true_n_list[index]
	# ax.quiver(n[0], n[1], n[2], n_1_list[index, 0]-n[0], n_1_list[index, 1]-n[1], n_1_list[index, 2]-n[2], arrow_length_ratio=0.1, color='g')   
	# ax.quiver(n[0], n[1], n[2], n_1[0]-n[0], n_1[1]-n[1], n_1[2]-n[2], arrow_length_ratio=0.1, color='m')   
	# ax.quiver(0, 0, 0, n[0], n[1], n[2], arrow_length_ratio=0.1, color='r')   
	# # ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], n_3_list[index, 0]-center_point[index, 0], n_3_list[index, 1]-center_point[index, 1], n_3_list[index, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='b')  
	# # for i in range(true_n_list.shape[0]): 
	# # 	n = true_n_list[i]
	# # 	ax.quiver(0, 0, 0, n[0], n[1], n[2], arrow_length_ratio=0.1, color='r')   
	
	# ax.set_xlim3d(-150, 150)     
	# ax.set_ylim3d(-150, 150)      
	# ax.set_zlim3d(-150, 0)     
	# plt.show() 
 
	# print("p :", t)   
	# print("psi_t_list :", psi_t_list)   

	# # print("p :", p)  
	# # print("theta_t_list :", theta_t_list)   
	# plot_vector_comparison(  
	# 	theta_t_list,  
	# 	np.array(t) 
	# ) 

	# plot_vector_comparison(  
	# 	psi_t_list,  
	# 	np.array(p)
	# ) 

	# plot_sphere_reference_eval(
	# 	sample_num_1=[200, 200],  
	# 	sample_num_2=50,   
	# 	index=0,    
	# 	theta_range=[np.pi, 10/8*np.pi],    
	# 	phi_range=[-np.pi/2,np.pi/2],    
	# 	psi=10   
	# )

	# x_0, y_0, z_0  = plot_sphere_reference_eval(
	# 	sample_num_1=[1, 200],   
	# 	sample_num_2=50,   
	# 	index=0,    
	# 	theta_range=[9/8*np.pi, 9/8*np.pi],    
	# 	# theta_range=[7/8*np.pi, 7/8*np.pi],   
	# 	phi_range=[0,2*np.pi],    
	# 	psi=10   
	# )   

	# x_1, y_1, z_1  = plot_sphere_reference_eval(
	# 	sample_num_1=[1, 200],   
	# 	sample_num_2=50,   
	# 	index=0,    
	# 	# theta_range=[9/8*np.pi, 9/8*np.pi],    
	# 	theta_range=[7/8*np.pi, 7/8*np.pi],   
	# 	phi_range=[0,2*np.pi],    
	# 	psi=10   
	# )   

	# plot_vector_comparison(
	# 	x_0, x_1
	# )

	# plot_vector_comparison(
	# 	y_0, y_1
	# )

	# plot_vector_comparison(
	# 	z_0, z_1
	# )