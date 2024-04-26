import os
import numpy as np
import rospy
import cPickle as pickle




path = os.path.expanduser("~/.rosbags/demos_TEST8/promp_backup")
fn = "promp_lib_033.pkl"

with open(os.path.join(path, fn), 'r') as f:
    promp_lib = pickle.load(f)

poses = promp_lib.get_candidate_poses()
for promp_key in promp_lib._promps.keys():
    promp = promp_lib._promps[promp_key]
                    
    for pose in poses:
        x = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        # TODO for now just adding a small offset to the y axis of reflex pad
        x[1] += 0.05
                
        
        # TODO scaling data as it seems to resolve some numerical issues, this scaling
        # factor has also been applied to the input data for learning ProMPs, and it's
        # scaled back when a ProMP is generated
        x *= 10.0
        
        # Compute ProMP probability
        # TODO is dividing by mu what you really want? doesn't always give valid probability
        mu_traj    = promp.get_mu_traj(1.0)
        sigma_traj = promp.get_sigma_traj(1.0)
        # sigma_inv  = np.linalg.pinv(sigma_traj)
        p_point    = promp_lib.prob(x, mu_traj, sigma_traj)

        # print "POINT PROB", p_point
