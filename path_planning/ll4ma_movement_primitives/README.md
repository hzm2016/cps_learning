# LL4MA Movement Primitives

This package implements various movement primitive policy representations, including Dynamic Movement Primitives (DMPs), Cartesian Dynamic Movement Primitives (CDMPs), and Probabilistic Movement Primitives (ProMPs). These implementations were utilized in the following publications:

* [Learning Task Constraints from Demonstration for Hybrid Force/Position Control](https://arxiv.org/abs/1811.03026)
* Active Learning of Probabilistic Movement Primitives (Forthcoming)

**Disclaimer:** Some of this code is not fully developed and was implemented more for the purpose of doing research quickly than for providing robust software for external users. As such, use at your own risk. You will likely benefit more from reading the implementations and taking the pieces you need than using this package directly.

---
## Installation

This package has some internal dependencies that are also ROS packages and can be cloned to a catkin workspace as usual:

  * [`ll4ma_msgs`](https://bitbucket.org/robot-learning/ll4ma_msgs) - ROS msg and srv files for logging, visualization, and execution.
  * [`ll4ma_trajectory_util`](https://bitbucket.org/robot-learning/ll4ma_trajectory_util) - Utilities for visualization and trajectory generation.
  
 Additionally, if you want to actually execute ProMP task space policies from the action server, you will need the [`ll4ma_opt_wrapper`](https://bitbucket.org/robot-learning/ll4ma_opt_utils/src/master/ll4ma_opt_wrapper/) package, which provides its own installation instructions and scripts.

---
## Dynamic Movement Primitives

Dynamic Movement Primitives (DMPs) represent a robot motion as a learned forcing function applied to a critically damped dynamical system. DMPs are useful because they theoretically guarantee the robot motion will converge to a desired goal configuration (because of the critically damped system), are simple to learn (typically by linear regression over squared exponential basis functions), and can be adapted to new goal configurations without re-learning (in virtue of the goal being an open parameter). The system is typically decoupled from explicit time by using a phase variable (also referred to as a "canonical system"), which has advantages such as allowing motion to start and stop gracefully when a perturbation is encountered. 

The implementation here is primarily based on the formulation of [Pastor et al. (2009)](http://ieeexplore.ieee.org/abstract/document/5152385/) while taking some guidance from [Ijspeert et al. (2013)](https://infoscience.epfl.ch/record/185437/files/neco_a_00393.pdf) for parameter settings. We have further extended these methods in our paper [Learning Task Constraints from Demonstration for Hybrid Force/Position Control](https://arxiv.org/abs/1811.03026) to account for contact-awareness and task constraint learning for hybrid force/position control.

See [`test/ll4ma_movement_primitives/dmps/test_dmp.py`](https://bitbucket.org/robot-learning/ll4ma_movement_primitives/src/master/test/ll4ma_movement_primitives/dmps/test_dmp.py) for an example of basic usage.

## Cartesian Dynamic Movement Primitives

Cartesian Dynamic Movement Primitives (CDMPs) follow the same principles as DMPs, but extend it to Cartesian space for orientation. Orientation is handled differently with DMPs, since the dimensions of a singularity-free orientation representation are not independent. To avoid adhoc normalization at every timestep, one can formulate the DMPs to output valid rotations by creating a DMP that considers all coupled dimensions of orientation simultaneously. This is formalized nicely in [Ude et al. (2014)](http://ieeexplore.ieee.org/abstract/document/6907291/). We utilized CDMPs in our paper [Learning Task Constraints from Demonstration for Hybrid Force/Position Control](https://arxiv.org/abs/1811.03026) to learn a dynamic Cartesian constraint frame for hybrid force/position control. 

**TODO:** This code is a bit unstable right now. DMPs were refactored and CDMPs fell behind. Need to get these back up and running.

## Probabilistic Movement Primitives

Probabilistic Movement Primitives represent a probability distribution of trajectories. This provides a more expressive representation for learning from multiple demonstrations, as it can represent not only the mean trajectory (as DMPs do), but capture properties of the variance as well. Generalization is still achievable at the start and goal positions like DMPs, but can also accommodate passing through desired intermediate waypoints through probabilistic conditioning. Please see [Paraschos et al. (2018)](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf) for a comprehensive description.

We utilized this implementation in our paper Active Learning for Probabilistic Movement Primitives (forthcoming), where we leverage the probabilistic information encoded in the ProMPs to guide task instance selection. The learner seeks out task instances that is uncertain it is able to achieve, thereby increasing its ability over a given space more quickly than if it just receives random demonstrations.

* See [`test/ll4ma_movement_primitives/promps/test_promp.py`](https://bitbucket.org/robot-learning/ll4ma_movement_primitives/src/master/test/ll4ma_movement_primitives/promps/test_promp.py) for an example of basic usage.
* See [`launch/execute_task_promp.launch`](https://bitbucket.org/robot-learning/ll4ma_movement_primitives/src/master/launch/execute_task_promp.launch) for a launch file that integrates the ProMP learning in our active learning setting, including visualization of the policies in rviz and execution on the real robot.