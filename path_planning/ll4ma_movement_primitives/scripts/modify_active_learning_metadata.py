"""
Interactive CLI to change information associated with an ActiveLearner instance's metadata. 
Setting one of the CL flags enables the associated attribute in the metadata for modification.
Interactive CL prompts will retrieve user responses, save them to the ActiveLearner's metadata, 
and re-save the learner as a pickle.

Usage:

    $ python modify_active_learning_metadata.py -f FILENAME -h

The filename (without extension, should be 'pkl') is the only required argument, and you should 
ensure path is set to the directory containing this file. If no other flags are set, nothing will
be modified and it will just print out the metadata to console. 
"""
import os
import sys
import argparse
import textwrap
import cPickle as pickle
from argparse import RawTextHelpFormatter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-f', '--file', dest='filename', type=str, required=True,
                        help=("Filename of pickled active learner (without extension, "
                              "e.g. 'my_learner' if your filename is 'my_learner.pkl')"))
    parser.add_argument('-p', '--path', dest='path', type=str, default="~/.rosbags/demos/backup",
                        help="Absolute filepath of directory containing pickled active learner")
    parser.add_argument('--desc', dest='change_description', action='store_true',
                        help="Change description/notes on experiment")
    parser.add_argument('--task', dest='change_task', action='store_true',
                        help="Change task of experiment")
    parser.add_argument('--robot', dest='change_robot', action='store_true',
                        help="Change robot used in experiment")
    parser.add_argument('--end-effector', dest='change_end_effector', action='store_true',
                        help="Change end-effector used in experiment")
    parser.add_argument('--objects', dest='change_objects', action='store_true',
                        help="Change objects used in experiment")
    parser.add_argument('--strategy', dest='change_strategy', action='store_true',
                        help="Change active learning strategy used in experiment")
    args = parser.parse_args(sys.argv[1:])
    
    filename = os.path.join(os.path.expanduser(args.path), args.filename + ".pkl")
    with open(filename, 'r') as f:
        learner = pickle.load(f)

    os.system('clear')
    print "\nChanging metadata for active learner: {}".format(filename)
        
    something_changed = False
    if args.change_description:
        desc = raw_input("\nDescription: ")
        learner._metadata.description = desc
        something_changed = True
    if args.change_task:
        task = raw_input("\nTask: ")
        learner._metadata.task = task
        something_changed = True
    if args.change_robot:
        robot = raw_input("\nRobot: ")
        learner._metadata.robot = robot
        something_changed = True
    if args.change_end_effector:
        ee = raw_input("\nEnd-Effector: ")
        learner._metadata.end_effector = ee
        something_changed = True
    if args.change_objects:
        objects = raw_input("\nObjects: ")
        learner._metadata.objects = objects
        something_changed = True
    if args.change_strategy:
        strategy = raw_input("\nActive Learning Strategy: ")
        learner._metadata.strategy = strategy
        something_changed = True

    os.system('clear')
        
    if something_changed:
        with open(filename, 'w+') as f:
            pickle.dump(learner, f)
        print "\n\nSaved changes to active learner: {}".format(filename)
    else:
        print "\n\nNo changes were made to the active learner."

    print learner._metadata
