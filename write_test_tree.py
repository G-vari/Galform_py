# This routine makes a simple merger tree, following the Fakhouri et al. (2011) fit to the mean mass accretion histories from the Millenium simulation.
# The tree has no subhaloes and is intendend for test purposes.

import numpy as np; from utilities_cosmology import *; import os; import h5py

###### inputs ######################

# Set cosmological parameters (note, only masses and exansion factors are output, so it doesn't matter if these don't agree with main simulation, at least for testing purposes)
omm = 0.3; oml = 0.7; h = 0.7

# Set log_10 of the desired final halo mass
logmhalo_desired = 12

# Set the number of timesteps. These will be evenly spaced in ln(expansion factor) between z = zmin and z = zmax
nsnap = 60

zmin = 0.0
zmax = 10.0

lna_min = np.log(1./(1.+zmax))
lna_max = np.log(1./(1.+zmin))

dlna = (lna_max - lna_min) / float(nsnap-1)

lna = np.arange( lna_min , lna_max +dlna, dlna )
a = np.exp(lna)

# Fakhouri mean halo mass accretion history
def hfr_function(tlb, m):
    t = t_Universe(1.0, omm, h)[0] - tlb
    a = a_Universe(t, omm, h)
    z = 1./a -1.0
    
    hfr = -46.1 * (m/10**12)**1.1 * (1.+1.11*z) * (omm*(1+z)**3 + oml)**0.5 # Msun yr^-1

    return hfr

node_mass = np.zeros_like(a)
node_mass[0] = 10.**logmhalo_desired

# Integrate the Fakhouri MAH backwards to get halo mass for each snapshot
for n in range(nsnap-1):

    a_n = a[::-1][n] # Expansion factor at this snapshot (::-1 means we are going backwards)
    a_np1 = a[::-1][n+1]

    # Age of universe and lookback time at this expansion factor
    t, tlb = t_Universe(a_n, omm, h)

    # Age of the universe at the next (earlier) expansion factor
    t2, tlb2 = t_Universe(a_np1, omm, h)

    # Timestep
    dt = t - t2

    # 4th order Runge-Kutta integration
    k1 = hfr_function(tlb         , node_mass[n]               ) # Msun yr^1
    k2 = hfr_function(tlb+dt/2. , node_mass[n] + dt/2. * k1)
    k3 = hfr_function(tlb+dt/2. , node_mass[n] + dt/2. * k2)
    k4 = hfr_function(tlb+dt    , node_mass[n] + dt    * k3)

    node_mass[n+1] = node_mass[n] + 1./6. * dt *10**9 * (k1 + 2*k2 + 2*k3 + k4)

# Set halo masses that are below a threshold to 0.0
# See input_parameters.py to see where this comes from.
# If mform_min changes in input_parameters.py, in principle it should change here too, but in practice that shouldn't be necessary
mform_min = 1.282732254*10**10 # Msun
mhalo_min = 2.*mform_min

below_min = node_mass < mhalo_min
node_mass[below_min] = 0.0

# Invert the order so that we go from low a to high a (a=expansion factor)
node_mass = node_mass[::-1]

# Snapshot number
snapshot = np.arange(0,nsnap)

# Create Ids in the format required by build_merger_trees.py
node_index = np.copy(snapshot)
host_index = np.copy(snapshot)
descendantIndex = np.copy(node_index) +1
descendantIndex[-1] = -1

isInterpolated = np.zeros_like(snapshot)
isMainProgenitor = np.ones_like(snapshot)
final_host_index = node_index[-1]

############# Write merger tree data to disk ###################

output_path = "/gpfs/data/d72fqv/PythonSAM/input_data/"
filename = "merger_tree_test.hdf5"

# Delete existing hdf5 file if one exists before writing a new one
if os.path.isfile(output_path+filename):
    os.remove(output_path+filename)

ntrees_choose = 1 # Only one tree for this test case

File = h5py.File(output_path+filename)
group = File.create_group("expansion_factor")
dset = group.create_dataset("a",data=a)
group = File.create_group("ntrees")
dset = group.create_dataset("ntrees",data=ntrees_choose)

group_trees = File.create_group("trees")

group = group_trees.create_group("tree_0")

dset = group.create_dataset("node_mass",data=node_mass)
dset = group.create_dataset("snapshot",data=snapshot)
dset = group.create_dataset("node_index",data=node_index)
dset = group.create_dataset("host_index",data=host_index)
dset = group.create_dataset("descendantIndex",data=descendantIndex)
dset = group.create_dataset("isInterpolated",data=isInterpolated)
dset = group.create_dataset("isMainProgenitor",data=isMainProgenitor)
dset = group.create_dataset("final_host_index",data=final_host_index)
