import numpy as np; import os; import pickle

# This is the run script for the galform_py model
# Your model name and other model parameters are set in input_parameters.py
# If this is your first time running the code, you will need to set appropriate paths
# To do this, go to the Paths object defined in input_parameters.py

########### Set options for this script #########################

# Run the build_merger_tree code
run_build_merger_tree = True

# Run the galaxy_formation code
run_galaxy_formation = True


########### Write used model parameters to output directory ################

# Read in default model parameters as a class instance

from input_parameters import *

# Write output directories if necessary
try:
    os.makedirs(Paths.output_path+Parameter.model_name)
except OSError:
    pass # Directory already exists

# Write parameters to output directory
parameter_file_path = Paths.output_path+Parameter.model_name+"/used_parameters.p"
pickle.dump(Parameter,open(parameter_file_path,"wb"))



############ Run build_merger_tree.py ########################
# This takes the raw merger tree data from an input file (hdf5 format), creates the merger trees data structure and writes it to disk

if run_build_merger_tree:
    print "running build_merger_tree.py"
    os.system("python build_merger_tree.py ")
    print "done"

############ Run galaxy_formation.py #########################
# This runs the baryonic physics, using the merger tree data structure as an input

if run_galaxy_formation:
    print "running galaxy_formation.py"
    os.system("python galaxy_formation.py")
    print "done"
