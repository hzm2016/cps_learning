import os
import numpy as np
import rospy
import cPickle as pickle
import matplotlib.pyplot as plt


# path = "/media/adam/data_haro/demos_TEST9/backup"

path = os.path.expanduser("~/.rosbags/demos/backup"); idx = 15


fn = "promp_lib_%03d.pkl" % idx

with open(os.path.join(path, fn), 'r') as f:
    lib = pickle.load(f)

with open(os.path.join(path, "gmm_%03d.pkl" % idx), 'r') as f:
    gmm = pickle.load(f)



    
# chosen        = None
# max_entropy   = 0.0
# max_pos       = 0.0 # For debugging output
# max_neg       = 0.0 # For debugging output
# poses         = lib.get_candidate_poses()

# # TODO using uniform mixture weights, should probably do a similarity measure
# promp_weight = 1.0 / len(lib._promps.keys())

# for pose in poses:
#     # TODO this pose right now is the object pose in the world frame but we need to
#     # do ProMP inference off the end-effector pose. Can do something more complicated
#     # to infer this, but for now just considering a canonical approach point some
#     # constant offset in one dimensions of the object frame
#     # TODO Need to implement this ^^^
#     x = np.array([pose.position.x, pose.position.y, pose.position.z])
    
#     # TODO scaling data as it seems to resolve some numerical issues, this scaling
#     # factor has also been applied to the input data for learning ProMPs, and it's
#     # scaled back when a ProMP is generated
#     x *= 10.0
    
#     # Compute ProMP probability
#     promp_prob = 0.0
#     for promp_key in lib._promps.keys():
#         promp = lib._promps[promp_key]
#         mu    = promp.get_mu_traj(1.0)
#         sigma = promp.get_sigma_traj(1.0)
#         promp_prob += promp_weight * lib.prob(x, mu, sigma, max_dim=2)
        
#     # Compute GMM probability
#     gmm_prob = gmm.prob_in_gmm(x)                


    
#     pos_prob = promp_prob / (promp_prob + gmm_prob)
#     neg_prob =   gmm_prob / (promp_prob + gmm_prob)
            
#     # Compute entropy value and see if it's a winner
#     entropy = -( (pos_prob * np.log(pos_prob)) +
#                  (neg_prob * np.log(neg_prob)) )

#     if entropy > max_entropy:
#         max_entropy = entropy
#         max_pos     = pos_prob
#         max_neg     = neg_prob
#         chosen      = pose
            
#     print "\nPos PROB", pos_prob
#     print "Neg PROB", neg_prob
#     print "    ENTROPY", entropy
                
# print "\nCHOSEN:", chosen
# print "    Entropy:", max_entropy
# print "   Pos prob:", max_pos
# print "   Neg prob:", max_neg


    
plt.figure()
for pose in lib._candidate_poses.keys():
    plt.plot(lib._candidate_poses[pose]['neg_prob'])
plt.show()
