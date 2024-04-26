import textwrap
from datetime import datetime


class ActiveLearnerMetadata(object):
    """
    Metadata for capturing details of the active learner used in an experiment.
    """
    
    def __init__(self, **kwargs):
        """
        description          : High-level description or notes on experiment, environment, etc.
        experiment_type      : Experiment key (e.g. "experiment_1")
        active_learning_type : Active learning key (e.g. "max_entropy")
        object_type          : Type of object used in the experiment
        datetime             : Date and time the file was saved
        task                 : Task performed in the experiment
        robot                : Robot used in the experiment
        end_effector         : End-effector used in the experiment
        strategy             : Active learning strategy used
        num_demos            : Number of demonstrations given by user
        num_promp_samples    : Number of samples ProMPs were learned from (can differ from num_demos
                               if robot is allowed to attempt new instances through conditioning)
        num_promps           : Number of ProMPs in the library
        num_obj_gmm_components : Number of components in GMM for object
        num_ee_gmm_components : Number of components in GMM for end-effector
        num_tot_candidates   : Number of candidate task instances (finite set for sampling from)
        num_pos_instances    : Number of task instances with a positive label
        num_neg_instances    : Number of task instances with a negative label
        promp_names          : List of names of ProMPs in library
        promp_num_demos      : List of numbers of demos in each ProMP in library        
        promp_data_names     : List of filenames associated with learned demos for each ProMP in library
        """
        self.description = kwargs.get("description",
                                      "No description provided.")
        self.experiment_type = kwargs.get("experiment_type", "")
        self.active_learning_type = kwargs.get("active_learning_type", "")
        self.object_type = kwargs.get("object_type", "")
        self.datetime = kwargs.get("datetime", datetime.now())
        self.task = kwargs.get("task", "Grasp object on table")
        self.robot = kwargs.get("robot", "KUKA LBR4+")
        self.end_effector = kwargs.get("end_effector", "ReFlex")
        self.num_demos = kwargs.get("num_demos", 0)
        self.num_promp_samples = kwargs.get("num_promp_samples", 0)
        self.num_promps = kwargs.get("num_promps", 0)
        self.num_obj_gmm_components = kwargs.get("num_obj_gmm_components", 0)
        self.num_ee_gmm_components = kwargs.get("num_ee_gmm_components", 0)
        self.num_tot_candidates = kwargs.get("num_tot_instances", 0)
        self.num_pos_instances = kwargs.get("num_pos_instances", 0)
        self.num_neg_instances = kwargs.get("num_neg_instances", 0)
        self.promp_names = kwargs.get("promp_names", [])
        self.promp_num_demos = kwargs.get("promp_num_demos", {})
        self.promp_data_names = kwargs.get("promp_data_names", {})

    def __str__(self):
        output = "\n\n{s:{c}^{n}}\n\n".format(
            s=" Active Learner ", c='=', n=90)
        lines = textwrap.wrap(self.description, 60)
        output += "{:4}{s:<24}{v}\n".format("", s="Description:", v=lines[0])
        if len(lines) > 1:
            # output += "{}\n".format(lines[0])
            for line in lines[1:]:
                output += "{:4}{s:<24}{desc}\n".format("", s="", desc=line)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Experiment:", v=self.experiment_type)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Learning Method:", v=self.active_learning_type)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Object:", v=self.object_type)
        output += "{:4}{s:<24}{v}\n".format("", s="Datetime:", v=self.datetime)
        output += "{:4}{s:<24}{v}\n".format("", s="Task:", v=self.task)
        output += "{:4}{s:<24}{v}\n".format("", s="Robot:", v=self.robot)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="End-Effector:", v=self.end_effector)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num Demos:", v=self.num_demos)
        for name in self.promp_names:
            output += "{:8}{s:<20}{v}\n".format(
                "", s=name, v=self.promp_num_demos[name])
        output += "{:4}{s:<24}\n".format("", s="Trajectory Names:")
        for name in self.promp_names:
            output += "{:8}{s:<18}\n".format("", s=name)
            for filename in self.promp_data_names[name]:
                output += "{:12}{s}\n".format("", s=filename)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num ProMP Samples:", v=self.num_promp_samples)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num Obj GMM Components:", v=self.num_obj_gmm_components)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num EE GMM Components:", v=self.num_ee_gmm_components)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num Total Candidates:", v=self.num_tot_candidates)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num Pos Instances:", v=self.num_pos_instances)
        output += "{:4}{s:<24}{v}\n".format(
            "", s="Num Neg Instances:", v=self.num_neg_instances)
        output += "\n {}\n\n".format('=' * 90)
        return output

    def update_datetime(self):
        self.datetime = datetime.now()


if __name__ == '__main__':
    import os
    import cPickle as pickle
    from ll4ma_movement_primitives.promps import ActiveLearner
    al = ActiveLearner()
    al._metadata.experiment_type = "experiment_1"
    al._metadata.active_learning_type = "max_entropy"
    al._metadata.object_type = "sugar"
    al._metadata.num_demos = 4
    al._metadata.promp_names = ["promp_0", "promp_1"]
    al._metadata.promp_num_demos = {"promp_0": 3, "promp_1": 1}
    al._metadata.promp_data_names = {
        "promp_0": ["trajectory_001", "trajectory_002", "trajectory_003"],
        "promp_1": ["trajectory_004"]
    }
    print al._metadata
    path = os.path.expanduser("~/.rosbags/demos/backup")
    filename = "test.pkl"
    with open(os.path.join(path, filename), 'w+') as f:
        pickle.dump(al, f)
