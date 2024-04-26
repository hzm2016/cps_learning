# from path_planning.plot_path import *
import numpy as np 
import matplotlib.pyplot as plt  
from matplotlib import cm   
from mpl_toolkits.mplot3d import Axes3D   
from matplotlib.ticker import LinearLocator    

import copy as cp   

from sklearn.metrics import mean_squared_error, r2_score    
import scipy.io as scio    

Action_Label_List = ['F', 'FV', 'FV+']      
Angle_Label_List = ['O', 'GVF']      
LEGEND_FONT_SIZE = 12   
STICKS_FONT_SIZE = 12     
TITLE_FONT_SIZE = 12   
LABEL_FONT_SIZE = 12      
TEXT_FONT_SIZE = 12  
LINE_WIDTH = 2.0        

FIGURE_1_YLABEL = ['Angle 1[rad]', 'Angle 2[rad]', 'Force[N]']   

FIGURE_5_YLABEL = ['Angle 1[rad]', 'Angle 2[rad]', 'Error[m]', 'Force[N]']   

FIGURE_7_YLABEL = ['Phase 1', 'Phase 2', 'Phase 3']   

FIGURE_8_YLABEL = ['Torque 1[Nm]', 'Torque 2[Nm]', 'Stiffness [Nm/rad]']    

alpha_list = [1.0, 0.85, 0.4]    

L_1 = 0.35   
L_2 = 0.35   


def cal_w(
	theta_raw=np.array([-54, -54, -54])  
):  
	angle_to_radian = np.pi/180    
	radian_to_angle = 180/np.pi   
	beta = 60 * angle_to_radian    
	alpha_1 = 45 * angle_to_radian     
	alpha_2 = 45 * angle_to_radian     
	alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))    

	eta = np.array([0, 120, 240])    
	theta = (eta + theta_raw) * angle_to_radian     

	w_list = np.zeros((3,3))     
	for i in range(3):      
		w_list[i, :] = np.array([np.sin(alpha_1) * np.cos(theta[i]), np.sin(alpha_1) * np.sin(theta[i]),  -1 * np.cos(alpha_1)])

	# print("w_list :", w_list)    
	return w_list     


def cal_Jacobian(
	theta_raw=np.array([-54, -54, -54]),   
	v_list=None   
):   
	Jacobian = np.zeros((3,3))     

	A = np.zeros((3,3))      
	B = np.zeros((3,3))      
	u = np.array([0, 0, -1])      

	w_list = cal_w(theta_raw=theta_raw)      
	
	for i in range(3):   
		A[i, :] = np.cross(w_list[i, :], v_list[i, :])
		B[i, i] = (np.cross(u,w_list[i,:])).transpose().dot(v_list[i, :])   
		# np.dot(A[i, :], u) 
	
	Jacobian = -1 * np.linalg.inv(A).dot(B) 
	
	return Jacobian   


def cal_forward_kinematics(  
	theta_raw=np.array([-54, -54, -54])   
):   
	angle_to_radian = np.pi/180    
	radian_to_angle = 180/np.pi    
	beta = 60 * angle_to_radian    
	alpha_1 = 45 * angle_to_radian     
	alpha_2 = 45 * angle_to_radian     
	alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))    

	eta = np.array([0, 120, 240])    
	theta = (eta - theta_raw) * angle_to_radian     

	# print("alpha_2 :", np.cos(alpha_2))  
	# print("alpha_3 :", np.cos(alpha_3))  
	# print("w_t_1 :%f, %f, %f", np.sin(alpha_1) * np.sin(theta[0]), -1 * np.sin(alpha_1) * np.cos(theta[0]),  -1 * np.cos(alpha_1)) 
	# print("w_t_2 :%f, %f, %f", np.sin(alpha_1) * np.sin(theta[1]), -1 * np.sin(alpha_1) * np.cos(theta[1]),  -1 * np.cos(alpha_1)) 
	# print("w_t_3 :%f, %f, %f", np.sin(alpha_1) * np.sin(theta[2]), -1 * np.sin(alpha_1) * np.cos(theta[2]),  -1 * np.cos(alpha_1))  

	w_list = np.zeros((3,3))     
	for i in range(3):      
		w_list[i, :] = np.array([np.sin(alpha_1) * np.cos(theta[i]), np.sin(alpha_1) * np.sin(theta[i]),  -1 * np.cos(alpha_1)])  
	
	# w_t_1 = np.array([0, -0.7071, -0.7071])  
	# w_t_2 = np.array([0.6124, 0.3536, -0.7071])   
	# w_t_3 = np.array([-0.6124, 0.3536, -0.7071])   
	w_t_1 = w_list[0, :]    
	w_t_2 = w_list[1, :]    
	w_t_3 = w_list[2, :]    
	# print("w_t_1 :", w_t_1)
	# print("w_t_2 :", w_t_2) 
	# print("w_t_3 :", w_t_3)

	v_t_1 = np.zeros(3)     
	v_t_2 = np.zeros(3)     
	v_t_3 = np.zeros(3)     
	J = np.zeros((9,9))    
	F = np.zeros(9)    

	Num_iteration = 100 

	# def cal_v(x): 
	# 	w_1 = np.array([0, -0.7071, -0.7071]) 
	# 	w_2 = np.array([0.6124, 0.3536, -0.7071]) 
	# 	w_3 = np.array([-0.6124, 0.3536, -0.7071])  
	# 	print("results :", w_1.dot(x[:3]), w_1[0]**2 + w_1[1]**2 + w_1[2]**2)  
	# 	return np.array([
	# 		x[0]**2 + x[1]**2 + x[2]**2 - 1,   
	# 		x[3]**2 + x[4]**2 + x[5]**2 - 1,   
	# 		x[6]**2 + x[7]**2 + x[8]**2 - 1,    
	# 		w_1.dot(x[:3]) - np.sqrt(2)/2,   
	# 		w_2.dot(x[3:6]) - np.sqrt(2)/2,   
	# 		w_3.dot(x[6:]) - np.sqrt(2)/2,   
	# 		x[:3].dot(x[3:6]) + 1/8,  
	# 		x[3:6].dot(x[6:]) + 1/8,  
	# 		x[6:].dot(x[:3]) + 1/8   
	# 	]) 

	# sol1_fsolve = fsolve(cal_v, np.array([1, 0, 0, 0, 1, 0, -1, -1, -1]))  

	# print(sol1_fsolve)   

	# x_1 :: [ 0.70712053  0.49999028 -0.49999028 -0.78655729  0.36241644 -0.49998195 0.07942406 -0.86235958 -0.50002777]
	# x_0 = np.array([1, 0, 0, 0, 1, 0, -1, -1, -1])   
	x_0 = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5]);   
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
		F[3] = w_t_1.transpose().dot(v_t_1) - np.cos(alpha_2) 
		F[4] = w_t_2.transpose().dot(v_t_2) - np.cos(alpha_2) 
		F[5] = w_t_3.transpose().dot(v_t_3) - np.cos(alpha_2) 
		F[6] = v_t_1.transpose().dot(v_t_2) - np.cos(alpha_3) 
		F[7] = v_t_2.transpose().dot(v_t_3) - np.cos(alpha_3) 
		F[8] = v_t_3.transpose().dot(v_t_1) - np.cos(alpha_3)  

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
	print("v_t_1 :", v_t_1)  
	print("v_t_2 :", v_t_2)  
	print("v_t_3 :", v_t_3)  
	print("x_1 :", x_1)  
	n = (v_t_1 + v_t_2 + v_t_3)/np.linalg.norm(v_t_1 + v_t_2 + v_t_3)    
	
	return n   


def cal_FF(
	theta_raw=np.array([-54, -54, -54])   
): 
	# 
	angle_to_radian = np.pi/180    
	radian_to_angle = 180/np.pi    
	beta = 60 * angle_to_radian    
	alpha_1 = 45 * angle_to_radian     
	alpha_2 = 45 * angle_to_radian     
	alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))    

	eta = np.array([0, 120, 240])    
	theta = (eta - theta_raw) * angle_to_radian      	
	x_list = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5])
	F = np.zeros(9)   
	w_list = np.zeros((3,3))     
	for i in range(3):      
		w_list[i, :] = np.array([np.sin(alpha_1) * np.cos(theta[i]), np.sin(alpha_1) * np.sin(theta[i]),  -1 * np.cos(alpha_1)])  
	
	v_t_1 = x_list[:3]   
	v_t_2 = x_list[3:6]   
	v_t_3 = x_list[6:]    
	# w_t_1 = np.array([0, -0.7071, -0.7071])  
	# w_t_2 = np.array([0.6124, 0.3536, -0.7071])   
	# w_t_3 = np.array([-0.6124, 0.3536, -0.7071])   
	w_t_1 = w_list[0, :]    
	w_t_2 = w_list[1, :]    
	w_t_3 = w_list[2, :]    

	F[0] = v_t_1[0]**2 + v_t_1[1]**2 + v_t_1[2]**2 - 1
	F[1] = v_t_2[0]**2 + v_t_2[1]**2 + v_t_2[2]**2 - 1 
	F[2] = v_t_3[0]**2 + v_t_3[1]**2 + v_t_3[2]**2 - 1  
	F[3] = w_t_1.transpose().dot(v_t_1) - np.cos(alpha_2)   
	F[4] = w_t_2.transpose().dot(v_t_2) - np.cos(alpha_2)   
	F[5] = w_t_3.transpose().dot(v_t_3) - np.cos(alpha_2)   
	F[6] = v_t_1.transpose().dot(v_t_2) - np.cos(alpha_3)   
	F[7] = v_t_2.transpose().dot(v_t_3) - np.cos(alpha_3)   
	F[8] = v_t_3.transpose().dot(v_t_1) - np.cos(alpha_3)   

	return F   


def cal_J(
	theta_raw=np.array([-54, -54, -54])  
):
	J = np.zeros((9,9))  

	angle_to_radian = np.pi/180    
	radian_to_angle = 180/np.pi    
	beta = 60 * angle_to_radian    
	alpha_1 = 45 * angle_to_radian     
	alpha_2 = 45 * angle_to_radian     
	alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))    

	eta = np.array([0, 120, 240])    
	theta = (eta - theta_raw) * angle_to_radian     	
	x_list = np.array([0.5,-0.70710678,-0.5,0.36237243,0.78656609,-0.5,-0.86237244,-0.07945932,-0.5])
	F = np.zeros(9)  
	w_list = np.zeros((3,3))     
	for i in range(3):      
		w_list[i, :] = np.array([np.sin(alpha_1) * np.cos(theta[i]), np.sin(alpha_1) * np.sin(theta[i]),  -1 * np.cos(alpha_1)])  
	
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

	print("J :", J)

	return J   


def cal_taskspace_angle(n, x_0):  
	# print("n :", n)  
	if n[0] < 0:   
		psi = np.arccos(n[2]/np.linalg.norm(n))   
	else:   
		psi = 2*np.pi - np.arccos(n[2]/np.linalg.norm(n))   

	phi = np.arctan(n[1]/n[0])     

	# print("psi :", psi, "phi :", phi)   
	return psi, phi  


def cal_angle_two_vectors(v_1, v_2): 
	La1 = np.sqrt(v_1.dot(v_1))
	La2 = np.sqrt(v_2.dot(v_2)) 
	cos_ang = v_1.dot(v_2) / (La1*La2)

	angle = np.arccos(cos_ang)  

	# angle = np.arccos(v_1.dot(v_2)/np.linalg.norm(v_1)/np.linalg.norm(v_2)) 

	return angle/np.pi*180  


def generate_3D_circle(
	sample_num=100, 
	r=1.8,  
	center=[0, 0, 0], 
	n=[1, 0, 0]  
):
	sample = np.linspace(-np.pi, np.pi, sample_num)  
	
	a = np.cross(n, [1, 0, 0])  # np.cross(), 向量叉积 
	if np.all(a == 0):   
		a = np.cross(n, [0, 1, 0])   
	
	b = np.cross(n, a)   
	
	# 归一化a，b（圆面两个互垂直向量）  
	a = a / np.linalg.norm(a)    
	b = b / np.linalg.norm(b)    
	
	# 利用空间圆的参数方程生成圆   
	c1 = center[0] * np.ones((sample_num, 1))    
	c2 = center[1] * np.ones((sample_num, 1))    
	c3 = center[2] * np.ones((sample_num, 1))    
	
	[c_x, c_y, c_z] = c1 + r * a[0] * np.cos(sample) + r * b[0] * np.sin(sample), \
					  c2 + r * a[1] * np.cos(sample) + r * b[1] * np.sin(sample), \
					  c3 + r * a[2] * np.cos(sample) + r * b[2] * np.sin(sample)    

	return c_x, c_y, c_z  


def get_3D_point(
	r=1.8,  
	center=[0, 0, 0],   
	n=[1, 0, 0],  
	sample=None   
):  
	sample = - sample * np.pi/180   
	sample_num = sample.shape[0]   
	a = np.cross(n, [0, 1, 0])   
	if np.all(a == 0):   
		a = np.cross(n, [1, 0, 0])    
	b = np.cross(n, a)   
	a = a/np.linalg.norm(a)    
	b = b/np.linalg.norm(b)    

	c1 = center[0] * np.ones((1, sample_num))     
	c2 = center[1] * np.ones((1, sample_num))     
	c3 = center[2] * np.ones((1, sample_num))      

	c_x, c_y, c_z = c1 + r * a[0] * np.cos(sample) + r * b[0] * np.sin(sample), \
					c2 + r * a[1] * np.cos(sample) + r * b[1] * np.sin(sample), \
					c3 + r * a[2] * np.cos(sample) + r * b[2] * np.sin(sample)
	
	# print(c_x, c_y, c_z)  
	# n_1 = np.array([c_x, c_y, c_z][0, :])   
	# print("n_1 :", n_1)  
	# print("sample :", r * a[0] * np.cos(sample) + r * b[0] * np.sin(sample))   
	# print(np.array([c_x, c_y, c_z]).shape)  
	
	n_1 = np.array([c_x[0][0], c_y[0][0], c_z[0][0]])    
	n_2 = np.array([c_x[0][1], c_y[0][1], c_z[0][1]])    
	n_3 = np.array([c_x[0][2], c_y[0][2], c_z[0][2]])    
	
	# n_1 = n_1/np.linalg.norm(n_1)     
	# n_2 = n_2/np.linalg.norm(n_2)     
	# n_3 = n_3/np.linalg.norm(n_3)     

	return [c_x, c_y, c_z], n_1, n_2, n_3   


def cal_inverse_kinematics(
	v_t_1,       
	v_t_2,        
	v_t_3       
):  
	angle_to_radian = np.pi/180    
	radian_to_angle = 180/np.pi    
	alpha_1 = 45 * angle_to_radian     
	alpha_2 = 45 * angle_to_radian     
	beta = 2*np.pi/3  
	alpha_3 = 2 * np.arcsin(np.sin(beta) * np.cos(np.pi/6))   
	# print("alpha_3 :", alpha_3 * 180/np.pi) 
	# print("cos(alpha_2) :", np.arccos(v_t_1.dot(v_t_2)) * 180/np.pi, 
	# np.arccos(v_t_1.dot(v_t_3)) * 180/np.pi, np.arccos(v_t_2.dot(v_t_3)) * 180/np.pi)   

	theta = np.zeros(3)    
	phi = np.zeros(3)    
	
	# if (np.sin(alpha_1) * v_t_1[0] > 0): 
	# 	phi[0] = np.arctan2((-np.sin(alpha_1) * v_t_1[1]), (np.sin(alpha_1) * v_t_1[0]))
	# 	theta[0] = np.arcsin((np.cos(alpha_2) + np.cos(alpha_1) * v_t_1[2]) / np.sqrt(
	# 	pow(np.sin(alpha_1) * v_t_1[0], 2) + pow(-np.sin(alpha_1) * v_t_1[1], 2))) - phi[0]
	# 	if ((theta[0] < 0) and (theta[0] > -np.pi)):  
	# 		theta[0] += 2*np.pi  
	
	# if (np.sin(alpha_1) * v_t_2[0] > 0):
	# 	phi[1] = np.arctan2((-np.sin(alpha_1) * v_t_2[1]), (np.sin(alpha_1) * v_t_2[0]))
	# 	theta[1] = np.arcsin((np.cos(alpha_2) + np.cos(alpha_1) * v_t_2[2])/np.sqrt(
	# 		pow(np.sin(alpha_1) * v_t_2[0], 2) + pow(-np.sin(alpha_1) * v_t_2[1], 2))) - phi[1]
	# 	if ((theta[1] < 0) and (theta[1] > -np.pi)):
	# 		theta[1] += 2*np.pi
	
	# if (np.sin(alpha_1) * v_t_3[0] < 0):
	# 	phi[2] = np.arctan2((-np.sin(alpha_1) * v_t_3[1]), (np.sin(alpha_1) * v_t_3[0]))
	# 	theta[2] = np.arcsin((np.cos(alpha_2) + np.cos(alpha_1) * v_t_3[2]) / np.sqrt(
	# 		pow(np.sin(alpha_1) * v_t_3[0], 2) + pow(-np.sin(alpha_1) * v_t_3[1], 2))) - phi[2]
	# 	if ((theta[2] < 0) and (theta[2] > -np.pi)):
	# 		theta[2] += 2 * np.pi  

	# if (np.sin(alpha_1) * v_t_1[0] > 0):   
	# 	phi[0] = np.arctan2(np.sin(alpha_1) * v_t_1[0], np.sin(alpha_1) * v_t_1[1])
	# 	theta[0] = np.arcsin((np.cos(alpha_2) + np.cos(alpha_1) * v_t_1[2]) / np.sqrt(
	# 	pow(np.sin(alpha_1) * v_t_1[0], 2) + pow(-np.sin(alpha_1) * v_t_1[1], 2))) - phi[0]
	# 	print("phi :", phi[0])  
	# 	# if ((theta[0] < 0) and (theta[0] > -np.pi)):  
	# 	# 	theta[0] += 2*np.pi   
	
	# if (np.sin(alpha_1) * v_t_2[1] < 0):  
	# 	phi[1] = np.arctan2((np.sin(alpha_1) * v_t_2[1]), (np.sin(alpha_1) * v_t_2[0]))
	# 	theta[1] = np.arccos((np.cos(alpha_2) + np.cos(alpha_1) * v_t_2[2])/np.sqrt(
	# 		pow(np.sin(alpha_1) * v_t_2[0], 2) + pow(np.sin(alpha_1) * v_t_2[1], 2))) + phi[1]
	# 	# if ((theta[1] < 0) and (theta[1] > -np.pi)):
	# 	# 	theta[1] += 2*np.pi  
	
	# if (np.sin(alpha_1) * v_t_3[1] > 0): 
	# 	phi[2] = np.arctan2((np.sin(alpha_1) * v_t_3[1]), (np.sin(alpha_1) * v_t_3[0]), )
	# 	theta[2] = np.arccos((np.cos(alpha_2) + np.cos(alpha_1) * v_t_3[2]) / np.sqrt(
	# 		pow(np.sin(alpha_1) * v_t_3[0], 2) + pow(np.sin(alpha_1) * v_t_3[1], 2))) + phi[2]
	# 	# if ((theta[2] < 0) and (theta[2] > -np.pi)):
	# 	# 	theta[2] += 2 * np.pi  

	# print("Theta :", theta*radian_to_angle)
	v_list = np.zeros((3, 3))    
	v_list[0, :] = v_t_1   
	v_list[1, :] = v_t_2    
	v_list[2, :] = v_t_3     
	eta_list = np.array([0., 120, 240])   
	theta_list = np.zeros(3)  
	for i in range(3):   
		v_t_1 = v_list[i, :]  
		eta = eta_list[i] * angle_to_radian     
		A = -np.cos(eta) * np.sin(alpha_1) * v_t_1[0] - np.sin(eta) * np.sin(alpha_1) * v_t_1[1] - np.cos(alpha_1) * v_t_1[2] - np.cos(alpha_2)
		B = np.sin(eta) * np.sin(alpha_1) * v_t_1[0] - np.cos(eta) * np.sin(alpha_1) * v_t_1[1]
		C = np.cos(eta) * np.sin(alpha_1) * v_t_1[0] + np.sin(eta) * np.sin(alpha_1) * v_t_1[1] - np.cos(alpha_1) * v_t_1[2] - np.cos(alpha_2)

		x_1 = (-2 * B + np.sqrt(4 * pow(B, 2) - 4 * A * C))/2/A
		x_2 = (-2 * B - np.sqrt(4 * pow(B, 2) - 4 * A * C))/2/A  
		theta_1 = np.arctan(x_1) * 2 * radian_to_angle  
		theta_2 = np.arctan(x_2) * 2 * radian_to_angle  
		print("theta_1 :", theta_1)   
		print("theta_2 :", theta_2)     

		# theta_list[i] = theta_2  

		theta_list[i] = -1 * theta_1  


	# print("theta_list :", theta_list) 
	return theta_list  


def cal_v_list(
	n, 
	sigma, 
	v_t_list
): 
	v_t_list_r = np.zeros((3, 3)) 
	for i in range(v_t_list.shape[0]): 
		v_t_1 = v_t_list[i, :] 
		v_t_list_r[i, :] = v_t_1 * np.cos(sigma) + np.cross(v_t_1, n) * np.sin(sigma) + n * (n.dot(v_t_1)) * (1 - np.cos(sigma))
	
	return v_t_list_r     


def vec_normalized(
	v_t=None
):  
    nor_v_t = cp.copy(v_t)  
    return nor_v_t/np.linalg.norm(nor_v_t)    


def cal_mirror_vec(
	v_t=None,  
):  
	mirror_v_t = cp.deepcopy(v_t)     
	mirror_v_t[1] = -1 * mirror_v_t[1]   
	return cp.deepcopy(mirror_v_t)       
    

def cal_mirror_motin(
    reference_motion_list  
):  
    # mirror_motion = np.zeros_like(reference_motion_list)  
    # mirror_joint_angle = np.zeros(3)   
    # mirrored by plane x-z   
    mirror_motion = reference_motion_list  
    for i in range(3): 
    	mirror_motion[i, 1] = -1 * reference_motion_list[i, 1]     
    return  mirror_motion     


def cal_mirror_path(  
    spm_fik=None,   
    v_list_all=None,
	sample_num_2=None           
):  
    v_list = np.zeros((3, 3))  
    v_list_mirror = np.zeros((3, 3))    
    q_list_mirror = np.zeros(3)        
    n = np.zeros(3)    
    n_mirror = np.zeros(3)      
    v_list_mirror_all = []    
    center_point_mirror = []     
    sample_point_mirror = []      
    trajectory_theta_list_mirror = []          
    
    for i in range(v_list_all.shape[0]):    
        v_list = v_list_all[i, :, :] 
        v_list_mirror[0, :] = cal_mirror_vec(v_list[0, :])       
        v_list_mirror[1, :] = cal_mirror_vec(v_list[2, :])     
        v_list_mirror[2, :] = cal_mirror_vec(v_list[1, :])          
        v_list_mirror_all.append(cp.deepcopy(v_list_mirror))                   
        
        q_list_mirror = spm_fik.cal_IK(v_t_1=v_list_mirror[0, :], v_t_2=v_list_mirror[1, :], v_t_3=v_list_mirror[2, :])     
        
        n_mirror = spm_fik.cal_n(v_list=v_list_mirror)    
        n_mirror = n_mirror * spm_fik.R    
        center_point_mirror.append(cp.deepcopy(n_mirror))      
        sample_point_mirror.append(generate_3D_circle(sample_num=sample_num_2, r=spm_fik.R_1, center=n_mirror, n=n_mirror))     
            
        trajectory_theta_list_mirror.append(cp.deepcopy(q_list_mirror))   
    
    v_list_mirror_all = np.array(v_list_mirror_all)  
    center_point_mirror = np.array(center_point_mirror)  
    trajectory_theta_list_mirror = np.array(trajectory_theta_list_mirror)     
    return center_point_mirror, v_list_mirror_all, trajectory_theta_list_mirror   


###############################################################
def plot_main( 
	forward_kinematics_name='./data/wrist/wrist_forward_kinematics.txt', 
	trajectory_theta_name='./data/wrist/wrist_encoder_circle.txt'
):   
	fig = plt.figure()    
	ax = Axes3D(fig)    
	
	angle_list_e = np.loadtxt(trajectory_theta_name, delimiter=',', skiprows=1)
	theta_list = angle_list_e[:, :3]      
	print("theta_list :", theta_list.shape)     
	
	n_list = []   
	for i in range(theta_list.shape[0]):   
		n = cal_forward_kinematics(theta_raw=-1* theta_list[i, :]) * 70    
		# n_list.append(n) 
		# n = theta_list[i, :] 
		n_list.append(n) 
		print("n :", n)   
		if (i%100==0): 
			print("n :", n)  
			ax.quiver(0, 0, 0, n[0], n[1], n[2], arrow_length_ratio=0.1)  
	
	r_list = np.array(n_list)  
	ax.plot3D(r_list[:, 0], r_list[:, 1], r_list[:, 2], color='r', linewidth=3)  
	ax.set_xlim(-70, 70)   
	ax.set_ylim(-70, 70)   
	ax.set_zlim(-150, 0)   

	plt.show()   
 
 
def plot_vector_comparison(
	trajectory_torque_t_list=None,    
	trajectory_torque_e_list=None    
):  
	fig = plt.figure(figsize=(8, 4))   

	trajectory_torque_e_list = (trajectory_torque_e_list - np.array([-54.73, -54.73, -54.73])) * np.array([-1, 1, -1]) + np.array([-54.73, -54.73, -54.73])
	num = 3   
	for i in range(num):   
		plt.plot(trajectory_torque_t_list[:, i], label=r"$\theta_t^{}$".format(str(i)), linewidth=3)    
		# plt.plot(trajectory_torque_e_list[:, i], label=r"$\theta_e^{}$".format(str(i)), linewidth=1, linestyle='--')      
		# plt.plot(trajectory_torque_t_list[:, i], label=r"$\theta_{}^t$".format(str(i)), linewidth=2)     
		# plt.plot(trajectory_theta_t_list[:, 0], trajectory_theta_t_list[:, 4+i], label=r"$q_{}^t$".format(str(i)), linewidth=2)    
		# plt.plot(trajectory_theta_t_list[:, 0], trajectory_theta_t_list[:, 1+i], label=r"$\theta_{}^t$".format(str(i)), linewidth=2)   
		# plt.plot(trajectory_theta_t_list[:, 0], trajectory_theta_e_list[:, i], label=r"$\theta_{}^e$".format(str(i)), linewidth=2)     

	plt.ylabel(r"$angle(deg)$")             
	plt.xlabel(r'$time(s)$')         

	plt.legend(loc="upper right")     
	# plt.legend(bbox_to_anchor=(1.05,1.0), prop = {'size':12})     
	# plt.legend(bbox_to_anchor=(0.2, 0.9), ncol=3)     
	plt.tight_layout()     
	plt.savefig('../data/reference_torque_comparison.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)        
	plt.show()    
 
 
def plot_sphere_reference_eval(
	sample_num_1=[1000,1000],    
	sample_num_2=50,    
	index=0,    
	theta_range=[7/8 * np.pi, 9/8*np.pi],    
	phi_range=[np.pi/2,np.pi/2],    
	psi=10    
):   
	fig = plt.figure(figsize=(8, 8))     
	ax = fig.gca(projection='3d')      
	
	# generate trajectory   
	r = 70    
	r_1 = r * np.sqrt(3)    
	center = np.zeros(3) 

	# theta = [8/9*np.pi]   
	# theta = np.linspace(np.pi, 7/8*np.pi, sample_num_1)      
	# theta = np.linspace(7/8*np.pi, np.pi, sample_num_1)      
	# theta = np.linspace(np.pi, 9/8*np.pi, sample_num_1)   
	theta = np.linspace(theta_range[0], theta_range[1], sample_num_1[0])    
	# theta = np.linspace(7/8 * np.pi, 9/8*np.pi, sample_num_1)    
	# phi = np.linspace(-np.pi, np.pi, sample_num_1)    
	phi = np.linspace(phi_range[0], phi_range[1], sample_num_1[1])     

	t, p = np.meshgrid(theta, phi)    
	
	# c1 = center[0] * np.ones((sample_num_1, 1))     
	# c2 = center[1] * np.ones((sample_num_1, 1))      
	# c3 = center[2] * np.ones((sample_num_1, 1))      

	x = r*np.sin(t)*np.cos(p)    
	y = r*np.sin(t)*np.sin(p)     
	z = r*np.cos(t)    
	print(x.shape,y.shape,z.shape)    

	# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.00, antialiased=False)   
	# ax.scatter(center[0], center[1], center[2], s=50, marker='^', color='black')      

	# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.00, antialiased=False)   
	
	x = x.flatten()    
	y = y.flatten()    
	z = z.flatten()    
	ax.scatter(center[0], center[1], center[2], s=50, marker='^', color='black')   
	ax.scatter(x, y, z, s=50, color='blue')   
	
	# ax.scatter(x, y, z, s=10)      

	plt.title(r'$\phi_t = {}$'.format(phi[0])) 
	ax.set_xlabel('$X$', fontsize=25)    
	ax.set_ylabel('$Y$', fontsize=25)    
	ax.set_zlabel('$Z$', fontsize=25)   

	ax.set_xlim3d(-100, 100)    
	ax.set_ylim3d(-100, 100)    
	ax.set_zlim3d(-100, 0)    
	
	# ax.grid(False)    
	# ax.set_xticks([])    
	# ax.set_yticks([])    
	# ax.set_zticks([])    
	# plt.axis('off')    
	
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
	
	plt.show()     

	return x, y, z   


def plot_single_spm_path(
    spm_fik=None,  
    center_point=None,     
    v_list=None,   
    sample_num_2=50,    
    plot_index=0,   
    save_fig=False    
):  
    fig = plt.figure(figsize=(8, 8))            
    ax = fig.gca(projection='3d')  
    print("center_point :", center_point.shape)    
    ax.scatter(0., 0.0, 0.0, s=50, color='black')    
    
    for i in range(center_point.shape[0]):    
        n = center_point[i, :]   
        ax.quiver(0, 0, 0, n[0], n[1], n[2], arrow_length_ratio=0.1, color='r')        
        
    index = plot_index     
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 0, 0]-center_point[index, 0], v_list[index, 0, 1]-center_point[index, 1], v_list[index, 0, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='g')   
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 1, 0]-center_point[index, 0], v_list[index, 1, 1]-center_point[index, 1], v_list[index, 1, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='m')   
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 2, 0]-center_point[index, 0], v_list[index, 2, 1]-center_point[index, 1], v_list[index, 2, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='b')   
    
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 0, 0], v_list[index, 0, 1], v_list[index, 0, 2], arrow_length_ratio=0.1, color='g')     
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 1, 0], v_list[index, 1, 1], v_list[index, 1, 2], arrow_length_ratio=0.1, color='m')     
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 2, 0], v_list[index, 2, 1], v_list[index, 2, 2], arrow_length_ratio=0.1, color='b')        
    
    # plot moving platform   
    sample_point = generate_3D_circle(sample_num=sample_num_2, r=spm_fik.R_1, center=center_point[plot_index, :], n=center_point[plot_index, :])     
    ax.scatter(sample_point[0], sample_point[1], sample_point[2], s=10, color='y')    
    
    ax.set_xlim3d(-150, 150)     
    ax.set_ylim3d(-150, 150)        
    ax.set_zlim3d(-150, 0)     

    # plt.title(r'$\phi_t = {}$'.format(phi[0]))      
    ax.set_xlabel('$X$[mm]', fontsize=20)    
    ax.set_ylabel('$Y$[mm]', fontsize=20)    
    ax.set_zlabel('$Z$[mm]', fontsize=20)    

    ax.view_init(40, 152)   
    ax.scatter(0., 0.0, 0.0, s=50, color='black')   
    
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    if save_fig: 
        plt.savefig('../data/reference_sphere_path.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)     

    plt.show()   
    
    
def plot_mirror_spm_path(
    center_point=None,     
    v_list=None, 
    center_point_mirror=None, 
    v_list_mirror=None, 
    sample_num_2=50, 
    plot_index=0, 
    spm_fik=None,  
    save_fig=False 
):   
    fig = plt.figure(figsize=(8, 8))          
    ax = fig.gca(projection='3d')     

    ax.scatter(0.0, 0.0, 0.0, s=50)      
    
    print("center_point :", center_point.shape, center_point_mirror.shape)  
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.00, antialiased=False)     
    # ax.scatter(center[0], center[1], center[2], s=50, marker='^', color='black')      
    
    for i in range(center_point.shape[0]):   
        n = center_point[i, :] 
        ax.quiver(0, 0, 0, n[0], n[1], n[2], arrow_length_ratio=0.1, color='r')      
        
        n_mirror = center_point_mirror[i, :] 
        print("n_mirror :", n_mirror)  
        ax.quiver(0, 0, 0, n_mirror[0], n_mirror[1], n_mirror[2], arrow_length_ratio=0.1, color='g') 
        
    # # print("center_points :", center_point, center_point_mirror)   
    # ax.scatter(center_point_mirror[:, 0], center_point_mirror[:, 1], center_point_mirror[:, 2], s=10)       
    
    # # for i in range(center_point.shape[0]):     
    # #     w_list = cal_w(theta_raw=np.array([-54.73, -54.73, -54.73]))     
    # #     w_list = w_list * 100    
    # #     ax.quiver(0.0, 0.0, 0.0, w_list[0, 0], w_list[0, 1], w_list[0, 2], arrow_length_ratio=0.1, color='g')    
    # #     ax.quiver(0.0, 0.0, 0.0, w_list[1, 0], w_list[1, 1], w_list[1, 2], arrow_length_ratio=0.1, color='m')    
    # #     ax.quiver(0.0, 0.0, 0.0, w_list[2, 0], w_list[2, 1], w_list[2, 2], arrow_length_ratio=0.1, color='b')    

    # #     # plot pose of end-effector 
    # #     # index = index     
    # #     ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], n_1_list[index, 0]-center_point[index, 0], n_1_list[index, 1]-center_point[index, 1], n_1_list[index, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='g')   
    # #     ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], n_2_list[index, 0]-center_point[index, 0], n_2_list[index, 1]-center_point[index, 1], n_2_list[index, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='m')   
    # #     ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], n_3_list[index, 0]-center_point[index, 0], n_3_list[index, 1]-center_point[index, 1], n_3_list[index, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='b')   

    index = plot_index     
    
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 0, 0]-center_point[index, 0], v_list[index, 0, 1]-center_point[index, 1], v_list[index, 0, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='g')   
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 1, 0]-center_point[index, 0], v_list[index, 1, 1]-center_point[index, 1], v_list[index, 1, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='m')   
    ax.quiver(center_point[index, 0], center_point[index, 1], center_point[index, 2], v_list[index, 2, 0]-center_point[index, 0], v_list[index, 2, 1]-center_point[index, 1], v_list[index, 2, 2]-center_point[index, 2], arrow_length_ratio=0.1, color='b')   
    
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 0, 0], v_list[index, 0, 1], v_list[index, 0, 2], arrow_length_ratio=0.1, color='g')     
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 1, 0], v_list[index, 1, 1], v_list[index, 1, 2], arrow_length_ratio=0.1, color='m')     
    ax.quiver(0.0, 0.0, 0.0, v_list[index, 2, 0], v_list[index, 2, 1], v_list[index, 2, 2], arrow_length_ratio=0.1, color='b')        
        
    ax.quiver(0.0, 0.0, 0.0, v_list_mirror_all[index, 0, 0], v_list_mirror_all[index, 0, 1], v_list_mirror_all[index, 0, 2], arrow_length_ratio=0.1, color='g')     
    ax.quiver(0.0, 0.0, 0.0, v_list_mirror_all[index, 1, 0], v_list_mirror_all[index, 1, 1], v_list_mirror_all[index, 1, 2], arrow_length_ratio=0.1, color='m')     
    ax.quiver(0.0, 0.0, 0.0, v_list_mirror_all[index, 2, 0], v_list_mirror_all[index, 2, 1], v_list_mirror_all[index, 2, 2], arrow_length_ratio=0.1, color='b')     
    
    ax.quiver(center_point_mirror[index, 0], center_point_mirror[index, 1], center_point_mirror[index, 2], v_list_mirror_all[index, 0, 0]-center_point_mirror[index, 0], v_list_mirror_all[index, 0, 1]-center_point_mirror[index, 1], v_list_mirror_all[index, 0, 2]-center_point_mirror[index, 2], arrow_length_ratio=0.1, color='g', ls='--')   
    ax.quiver(center_point_mirror[index, 0], center_point_mirror[index, 1], center_point_mirror[index, 2], v_list_mirror_all[index, 1, 0]-center_point_mirror[index, 0], v_list_mirror_all[index, 1, 1]-center_point_mirror[index, 1], v_list_mirror_all[index, 1, 2]-center_point_mirror[index, 2], arrow_length_ratio=0.1, color='m', ls='--')   
    ax.quiver(center_point_mirror[index, 0], center_point_mirror[index, 1], center_point_mirror[index, 2], v_list_mirror_all[index, 1, 0]-center_point_mirror[index, 0], v_list_mirror_all[index, 2, 1]-center_point_mirror[index, 1], v_list_mirror_all[index, 2, 2]-center_point_mirror[index, 2], arrow_length_ratio=0.1, color='b', ls='--')   
    

    # plot moving platform   
    sample_point = generate_3D_circle(sample_num=sample_num_2, r=spm_fik.R_1, center=center_point[plot_index, :], n=center_point[plot_index, :])     
    ax.scatter(sample_point[0], sample_point[1], sample_point[2], s=10, color='y')    

    mirror_sample_point = generate_3D_circle(sample_num=sample_num_2, r=spm_fik.R_1, center=center_point_mirror[plot_index, :], n=center_point_mirror[plot_index, :])      
    ax.scatter(mirror_sample_point[0], mirror_sample_point[1], mirror_sample_point[2], s=10, color='c')    
    
    ax.set_xlim3d(-150, 150)    
    ax.set_ylim3d(-150, 150)     
    ax.set_zlim3d(-150, 0)    

    # plt.title(r'$\phi_t = {}$'.format(phi[0]))    
    ax.set_xlabel('$X$[mm]', fontsize=20)    
    ax.set_ylabel('$Y$[mm]', fontsize=20)    
    ax.set_zlabel('$Z$[mm]', fontsize=20)    

    # ax.grid(False)    
    # ax.set_xticks([])    
    # ax.set_yticks([])    
    # ax.set_zticks([])    
    # plt.axis('off')    
    
    ax.view_init(40, 45) 

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    if save_fig: 
        plt.savefig('../data/reference_sphere_path.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)     

    plt.show()   

    # plot_vector_comparison(np.array(trajectory_theta_list), np.array(trajectory_theta_list_mirror))   
    # plot_vector_comparison(np.array(forward_n_list), np.array(true_n_list))      


###############################################################
def load_all_data(file_list=None, args=None):    
    # # for file_name in file_list:   
    # start_index_list = [0, 150, 350]
    # end_index_list = [100, 220, 500]
    start_index_list = [0, 600, 1600]     
    end_index_list = [400, 880, 2000]     

    # data = scio.loadmat(file_list[0])      
    # time_t_all = data['time_112'][:, ::args.resample].T     
    # force_x_t_all = data['data_Fx_112'][:, ::args.resample].T   
    # force_y_t_all = data['data_Fy_112'][:, ::args.resample].T  
    # q_d_1_all = data['data_qd1_112'][:, ::args.resample].T
    # q_d_2_all = data['data_qd2_112'][:, ::args.resample].T
    # q_t_1_all = data['data_qt1_112'][:, ::args.resample].T  
    # q_t_2_all = data['data_qt2_112'][:, ::args.resample].T    

    data = scio.loadmat(file_list[0])      
    time_t_all = data['time_105'][:, ::args.resample].T     
    force_x_t_all = data['data_Fx_105'][:, ::args.resample].T   
    force_y_t_all = data['data_Fy_105'][:, ::args.resample].T  
    q_d_1_all = data['data_qd1_105'][:, ::args.resample].T
    q_d_2_all = data['data_qd2_105'][:, ::args.resample].T
    q_t_1_all = data['data_qt1_105'][:, ::args.resample].T  
    q_t_2_all = data['data_qt2_105'][:, ::args.resample].T    

    # # start_index_list *= start_index_list
    # # end_index_list *= end_index_list
    # time_t = time_t_all[start_index_list[0]:end_index_list[0], :]  
    # force_x_t = force_x_t_all[start_index_list[0]:end_index_list[0], :]  
    # force_y_t = force_y_t_all[start_index_list[0]:end_index_list[0], :]  
    # q_d_1 = q_d_1_all[start_index_list[0]:end_index_list[0], :]  
    # q_d_2 = q_d_2_all[start_index_list[0]:end_index_list[0], :]  
    # q_t_1 = q_t_1_all[start_index_list[0]:end_index_list[0], :]  
    # q_t_2 = q_t_2_all[start_index_list[0]:end_index_list[0], :]  

    time_t = time_t_all  
    force_x_t = force_x_t_all
    force_y_t = force_y_t_all  
    q_d_1 = q_d_1_all
    q_d_2 = q_d_2_all
    q_t_1 = q_t_1_all
    q_t_2 = q_t_2_all 

    return time_t, force_x_t, force_y_t, q_d_1, q_d_2, q_t_1, q_t_2   


def load_data(file_list=None, args=None):    
    import h5py   
    # # for file_name in file_list:   
    # start_index_list = [0, 150, 350]
    # end_index_list = [100, 220, 500]
    start_index_list = [0, 600, 1600]   
    end_index_list = [400, 880, 2000]    
    data = scio.loadmat(file_list[0])     
    time_t_all = data['time_112'][:, ::args.resample].T  
    force_x_t_all = data['data_Fx_112'][:, ::args.resample].T
    force_y_t_all = data['data_Fy_112'][:, ::args.resample].T  
    q_d_1_all = data['data_qd1_112'][:, ::args.resample].T
    q_d_2_all = data['data_qd2_112'][:, ::args.resample].T
    q_t_1_all = data['data_qt1_112'][:, ::args.resample].T  
    q_t_2_all = data['data_qt2_112'][:, ::args.resample].T  

    # start_index_list *= start_index_list
    # end_index_list *= end_index_list
    time_t = time_t_all[start_index_list[0]:end_index_list[0], :]  
    force_x_t = force_x_t_all[start_index_list[0]:end_index_list[0], :]  
    force_y_t = force_y_t_all[start_index_list[0]:end_index_list[0], :]  
    q_d_1 = q_d_1_all[start_index_list[0]:end_index_list[0], :]  
    q_d_2 = q_d_2_all[start_index_list[0]:end_index_list[0], :]  
    q_t_1 = q_t_1_all[start_index_list[0]:end_index_list[0], :]  
    q_t_2 = q_t_2_all[start_index_list[0]:end_index_list[0], :]  

    for i in range(1, 3):   
        # time_t_i = time_t_all[start_index_list[i]:end_index_list[i], :] - (time_t_all[start_index_list[i], :] - time_t_all[end_index_list[i-1], :])  
        force_x_t_i = force_x_t_all[start_index_list[i]:end_index_list[i], :]   
        force_y_t_i = force_y_t_all[start_index_list[i]:end_index_list[i], :]   
        q_d_1_i = q_d_1_all[start_index_list[i]:end_index_list[i], :]   
        q_d_2_i = q_d_2_all[start_index_list[i]:end_index_list[i], :]   
        q_t_1_i = q_t_1_all[start_index_list[i]:end_index_list[i], :]   
        q_t_2_i = q_t_2_all[start_index_list[i]:end_index_list[i], :]   

        # time_t = np.vstack([time_t, time_t_i])   
        force_x_t = np.vstack([force_x_t, force_x_t_i])     
        force_y_t = np.vstack([force_y_t, force_y_t_i])    
        q_d_1 = np.vstack([q_d_1, q_d_1_i])      
        q_d_2 = np.vstack([q_d_2, q_d_2_i])    
        q_t_1 = np.vstack([q_t_1, q_t_1_i])     
        q_t_2 = np.vstack([q_t_2, q_t_2_i])      

    # print(time_t_i.shape, force_x_t_i.shape, force_y_t_i.shape)   
    # print(data['time_112'].shape)   
    # print("time_t :", time_t)  
    return time_t, force_x_t, force_y_t, q_d_1, q_d_2, q_t_1, q_t_2   


def load_force_data(args=None):      
    import h5py   
    # # for file_name in file_list:   
    # start_index_list = [0, 150, 350]
    # end_index_list = [100, 220, 500]
    start_index_list = [0, 600, 1600]   
    end_index_list = [400, 880, 2000]    
    data = scio.loadmat('./online_tase/data/force_112.mat')   

    time_t_all = data['time_112'][:, ::args.resample].T  

    damping_1_t_all = data['force_C_1_112'][:, ::args.resample].T
    damping_2_t_all = data['force_C_2_112'][:, ::args.resample].T  
    stiff_1_t_all = data['force_K_1_112'][:, ::args.resample].T
    stiff_2_t_all = data['force_K_2_112'][:, ::args.resample].T  
    tau_1_t_all = data['force_tau_1_112'][:, ::args.resample].T    
    tau_2_t_all = data['force_tau_2_112'][:, ::args.resample].T    
    tau_o_1_t_all = data['force_tau_o_1_112'][:, ::args.resample].T    
    tau_o_2_t_all = data['force_tau_o_2_112'][:, ::args.resample].T     

    # start_index_list *= start_index_list
    # end_index_list *= end_index_list
    time_t = time_t_all[start_index_list[0]:end_index_list[0], :]  
    damping_1_t = damping_1_t_all[start_index_list[0]:end_index_list[0], :]  
    damping_2_t = damping_2_t_all[start_index_list[0]:end_index_list[0], :]  
    stiff_1_t = stiff_1_t_all[start_index_list[0]:end_index_list[0], :]    
    stiff_2_t = stiff_2_t_all[start_index_list[0]:end_index_list[0], :]    
    tau_1_t = tau_1_t_all[start_index_list[0]:end_index_list[0], :]  
    tau_2_t = tau_2_t_all[start_index_list[0]:end_index_list[0], :]  
    tau_o_1_t = tau_o_1_t_all[start_index_list[0]:end_index_list[0], :]  
    tau_o_2_t = tau_o_2_t_all[start_index_list[0]:end_index_list[0], :]   

    for i in range(1, 3):   
        damping_1_t_i = damping_1_t_all[start_index_list[i]:end_index_list[i], :]  
        damping_2_t_i = damping_2_t_all[start_index_list[i]:end_index_list[i], :]  
        stiff_1_t_i = stiff_1_t_all[start_index_list[i]:end_index_list[i], :]    
        stiff_2_t_i = stiff_2_t_all[start_index_list[i]:end_index_list[i], :]    
        tau_1_t_i = tau_1_t_all[start_index_list[i]:end_index_list[i], :]  
        tau_2_t_i = tau_2_t_all[start_index_list[i]:end_index_list[i], :]  
        tau_o_1_t_i = tau_o_1_t_all[start_index_list[i]:end_index_list[i], :]  
        tau_o_2_t_i = tau_o_2_t_all[start_index_list[i]:end_index_list[i], :]    

        # time_t = np.vstack([time_t, time_t_i])   
        damping_1_t = np.vstack([damping_1_t, damping_1_t_i])     
        damping_2_t = np.vstack([damping_2_t, damping_2_t_i])    
        stiff_1_t = np.vstack([stiff_1_t, stiff_1_t_i])     
        stiff_2_t = np.vstack([stiff_2_t, stiff_2_t_i])    
        tau_1_t = np.vstack([tau_1_t, tau_1_t_i])     
        tau_2_t = np.vstack([tau_2_t, tau_2_t_i])     
        tau_o_1_t = np.vstack([tau_o_1_t, tau_o_1_t_i])     
        tau_o_2_t = np.vstack([tau_o_2_t, tau_o_2_t_i])     

    print("time_t :", time_t)  
    return time_t, damping_1_t, damping_2_t, stiff_1_t, stiff_2_t, tau_1_t, tau_2_t, tau_o_1_t, tau_o_2_t   
    # return time_t_i, force_x_t_i, force_y_t_i, q_d_1_i, q_d_2_i, q_t_1_i, q_t_2_i    


def forward_kinematics(angle_t_1=None, angle_t_2=None): 
    ###### input joint angle ########   
    x_t = L_1 * np.cos(angle_t_1) + L_2 * np.cos(angle_t_1 + angle_t_2)  
    y_t = L_1 * np.sin(angle_t_1) + L_2 * np.sin(angle_t_1 + angle_t_2)     

    return x_t, y_t     


def plot_figure_tase_11(args=None,   
                        x_e=None, y_e=None,   
                        x_t=None, y_t=None,     
                        circle_center=None,    
                        q_list=None, 
                        obs_center=None,   
                        obs_r=0.1,   
                        via_points=None,  
                        start=None,   
                        end=None   
                        ):   
	plt.figure(figsize=(4, 4), dpi=100)     

	if obs_center is not None:    
		draw_circle_obs = plt.Circle(obs_center, obs_r, fill=True, color='c', alpha=0.6)        
		# plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')     
		plt.gcf().gca().add_artist(draw_circle_obs)        

	# reference
	if x_e is not None: 
		plt.plot(x_e, y_e, color="black", linewidth=3, label=r'$\widetilde{x_d}$')  
		plt.scatter(circle_center[0], circle_center[1], color=[0.64, 0., 0.65], marker='X', s=80)   

	# actual path
	if x_t is not None:
		plt.plot(x_t, y_t, color="green", linewidth=2.0, alpha=0.8, label=r'$x_t$')
		plt.scatter(circle_center[0], circle_center[1], color=[0.64, 0., 0.65], marker='X', s=80)            

	if via_points is not None:    
		plt.scatter(via_points[0, 0], via_points[0, 1], color=[0.64, 0., 0.65], marker='X', s=80, label='via-points')   
		for i in range(1, via_points.shape[0]):       
			plt.scatter(via_points[i, 0], via_points[i, 1], color=[0.64, 0., 0.65], marker='X', s=80)    

	# L_1 = 0.35    
	# for i in range(q_list.shape[0]):   
	# 	x_m = L_1 * np.cos(q_list[:, 0])    
	# 	y_m = L_1 * np.sin(q_list[:, 1])    

	# 	if i%500 == 0:    
	# 	# if i == 0: 
	# 		print("i :", i)  
	# 		plt.arrow(0.0, 0.0, x_m[i], y_m[i], fc='b', ec='b', 
	# 			lw=2, length_includes_head=True, head_width=0.01, head_length=0.01, color='b')     
			
	# 		plt.arrow(x_m[i], y_m[i], x_t[i]-x_m[i], y_t[i]-y_m[i], fc='k', ec='k', 
	# 			lw=2, length_includes_head=True, head_width=0.01, head_length=0.01, color='k')    

	plt.ylabel(r"$X_2$[m]", fontsize=LABEL_FONT_SIZE)           
	plt.xlabel(r"$X_1$[m]", fontsize=LABEL_FONT_SIZE)          
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=LEGEND_FONT_SIZE)          
	plt.ylim(-0.4, 0.4)           
	plt.xlim(0.0, 0.80)        
	# ax.set_xlim(-30.0, 30.0)         
	# ax.set_ylim(-30.0, 30.0)      
	# plt.axis('equal') 
	# plt.locator_params(nbins=3)     
	plt.tick_params(labelsize=STICKS_FONT_SIZE)     
	plt.tight_layout()   

	# if args.save_fig:   
	#     print("save kmp !!!")    
	#     plt.savefig(args.fig_path + '.pdf', bbox_inches='tight', pad_inches=0.0)        
	plt.savefig('expected_path.png', bbox_inches='tight', pad_inches=0.0)   
	plt.show()   
 

def plot_figure_tase_5(args=None, time_t=None, q_d_1=None, q_d_2=None, q_t_1=None, q_t_2=None):      
    num_fig = 4  
    fig= plt.figure(figsize=(11, num_fig*2), dpi=150)    
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)   
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.90, wspace=0.4, hspace=0.4)    

    scale = 4    
    # start_index_list = [0, 400/scale, 680/scale]     
    # end_index_list = [401/scale, 681/scale, 1100/scale]     
    # angle_t_1 = -1 * q_t_1     
    # angle_t_2 = -1 * q_t_2     
    # angle_e_1 = -1 * q_d_1     
    # angle_e_2 = -1 * q_d_2     

    length = q_d_1.shape[0] 
    time_t = np.linspace(0, length/4, length)   

    for index in range(num_fig):  
        ax = plt.subplot(num_fig, 1, index + 1)      
        if index == num_fig-1:       
            plt.xlabel("Times[s]", fontsize=LABEL_FONT_SIZE)     

        plt.ylabel(FIGURE_5_YLABEL[index], fontsize=LABEL_FONT_SIZE)     

        if index == 0: 
            plt.plot(time_t, q_d_1, linewidth=LINE_WIDTH+2, label=r'$q_d^1$', color='#30A2FF')     
            plt.plot(time_t, q_t_1, linewidth=LINE_WIDTH-1, label=r'$q_t^1$', color='k')   
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=LEGEND_FONT_SIZE)  

        if index == 1: 
            plt.plot(time_t, q_d_2, linewidth=LINE_WIDTH+2, label=r'$q_d^2$', color='#30A2FF')     
            plt.plot(time_t, q_t_2, linewidth=LINE_WIDTH-1, label=r'$q_t^2$', color='k')   
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2,fontsize=LEGEND_FONT_SIZE)     

        # plt.grid()    
        # plt.legend(loc='best', fontsize=LEGEND_FONT_SIZE)     
        plt.xticks(fontsize=STICKS_FONT_SIZE)  
        plt.yticks(fontsize=STICKS_FONT_SIZE)  
        # plt.ylim(YLIM[index][0], YLIM[index][1])  
        # plt.xlim(XLIM[0], XLIM[1])  

    # # Create the legend outside of the subplots  
    # legend = fig.legend(labels=Label_list, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(Force_list))    
    # if args.save_fig:  
    #     plt.savefig(args.fig_path + '.pdf', bbox_inches='tight', pad_inches=0.0)     
    #     print("save success !!!")    
    
    plt.savefig('joint_position.png', bbox_inches='tight', pad_inches=0.0)     
    print("save success !!!")   
    plt.show()   

# plt.text(via_points[1, 0] - 5.0, via_points[1, 1] + 0.5, r'$t_s$', size=TEXT_FONT_SIZE)     
# plt.text(via_points[4, 1] - 5.0, via_points[4, 1] + 0.5, r'$t_e$', size=TEXT_FONT_SIZE)      
# # plt.text(via_points[4, 1] + 0.6, via_points[4, 1] + 0.3, r'$t_e$', size=font_size)   
# # plt.text(center_point_left[0]-1, center_point_left[1]-4, 'Impaired')     

# # # ori value    
# # plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color='black', marker='X', s=80, label='start-points')   
# # plt.scatter(mu_gmr[198, 0], mu_gmr[198, 1], color='green', marker='X', s=80, label='end-points')  

# plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], fc='b', ec='b', 
#           lw=2, length_includes_head=True, head_width=0.01, head_length=0.01, color='b')    

# plt.text(end[0] + 1.5, end[1] + 1.2, r'$F_h$', size=font_size)    
# plt.text(start[0] + 0.01, start[1] + 0.01, r'$F_h$', size=TEXT_FONT_SIZE)


def plot_figure_tase_12(fig_path=None):     
    x_list = ['Force', 'Vision', 'SAC+Force', 'Ours']   
    mean_assembly_steps_list = [229.87878787878788,317.5,271.2,183.56875]   
    std_assembly_steps_list = [14.166593744748313,35.4700577952729,50.82098483894227,15.733174296291896] 

    # mean_force_list = [1002.0661027102541,614.0365806955814,918.5083789121447,435.4161330305983] 
    # std_force_list = [192.05918543758767,111.3434457888319,215.48080130834373,97.62239492606486] 

    # ======================================================================================  
    # R = 0
    # for i in range(len(force)):
    #     r_force = -(np.linalg.norm(abs(force[i][:3])) + np.linalg.norm(abs(force[i][3:6])) * 100)
    #     R += r_force
    # print("Reward: ", R)

    # sns.set(style = 'white')  #darkgrid，whitegrid，dark, white, ticks
    # sns.set(font_scale=1.4) 
    plt.figure(figsize=(5.5, 4))   

    plt.bar(x_list, mean_assembly_steps_list, yerr=np.array(std_assembly_steps_list), color=['royalblue','royalblue','royalblue','royalblue'], alpha=0.8)  
    # plt.bar(x_list, mean_force_list, yerr=np.array(std_force_list), color=['darkorange','darkorange','darkorange','darkorange'], alpha=0.5)  

    # plt.errorbar(x=x_list, y=mean_list,yerr=np.array(std_list))   
    # x_axis_labels = ['a[0]','a[1]','a[2]','a[3]'] # labels for x-axis
    # y_axis_labels = [0, 30, 60, 90, 120, 150]
    # sns.heatmap(action[0::2,:],linewidths=0.,annot=False,xticklabels=x_axis_labels,yticklabels=False)  
    plt.ylabel("Complete Steps")  
    # plt.ylabel("Assembly Force")  
    # plt.title("Episode "+str(N)) 
    plt.tight_layout()  
    # sns.despine()  
    if args.save_fig:    
        plt.savefig(fig_path + '.pdf', bbox_inches='tight', pad_inches=0.0)     
        print("save success !!!")  
    # plt.savefig('./Prediction-vision-force/figures/histogram_contact_force.pdf', bbox_inches='tight', pad_inches=0)  
    # plt.savefig('./Prediction-vision-force/figures/histogram_assembly_steps.pdf', bbox_inches='tight', pad_inches=0)  
    plt.show()   
