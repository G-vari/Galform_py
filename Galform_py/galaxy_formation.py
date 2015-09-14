import numpy as np; import cPickle as pickle; from merger_tree_class import *; from input_parameters import *; from evolve_galaxies import *; import time; import sys

print "loading merger tree"

code_time1 = time.time()

trees = pickle.load(open(Input_Data_Path.halo_data_path+"merger_trees"+Parameter.model_name+".p","rb"))

code_time2 = time.time()
print "done, time taken = ", code_time2 - code_time1

# Loop over merger trees

for i_tree, tree in enumerate(trees.trees):

    print "tree", i_tree, " of ", len(trees.trees)

    # Level (snapshot) loop

    code_time3 = time.time()

    nsnap = len(tree.levels)

    for ilevel in range(nsnap-1):

        code_time_i1 = time.time()

        level = tree.levels[ilevel]
        dt = level.dt

        nodes_ts = tree.levels[ilevel].NodesOnLevel()
        nodes_ns = tree.levels[ilevel+1].NodesOnLevel()

        # Halo loop
        for halo in level.haloes:

            # Initialise the total y array for this step (includes halo y and all subhalo y)
            # y is an array containing the mass, metal mass and angular momentum in each reservoir
            y_total = Initialise_y_total(halo)

            # Calculate baryon accretion rates onto into this halo and it's subhaloes for this timestep
            halo.Calculate_Baryon_Accretion_Rates()
            for subhalo in halo.subhaloes:
                subhalo.Calculate_Baryon_Accretion_Rates()

            # Do galaxy formation on this halo and it's subhaloes for this level
            y_total_next = Evolve_System(halo, y_total)

            # Pass on baryonic mass and angular momentum to the next step
            Update_Halo(halo, y_total_next)

            # Save desired output properties for the descendant
            Save_Halo_Properties(halo)

        # Check baryonic mass/J in each halo == fb x mchalo
        for ihalo, halo in enumerate(tree.levels[ilevel+1].haloes):

            if hasattr(halo, 'y'):
                mbaryon = np.sum(halo.y[1:5])
                Jbaryon = np.sum(halo.y[10:14])
                for subhalo in halo.subhaloes:
                    if hasattr(subhalo, 'y'):
                        mbaryon += np.sum(subhalo.y[1:5])
                        Jbaryon += np.sum(subhalo.y[10:14])


                # I'm not completely sure but the tolerance here might need to be dependent on the number of nodes?
                if abs(mbaryon - Constant.fb * halo.mchalo)/mbaryon > 0.01:

                    print ""
                    print "If there is a warning message above, this means the code might have crashed because the integrator gave up"
                    print ""
                    print "Error in Mass Conservation"
                    print np.log10(mbaryon), np.log10(Constant.fb * halo.mchalo), np.log10(Constant.fb*halo.y[0])

                    print ""
                    mbaryon_progenitor = np.sum(halo.progenitors[0].y[1:5])
                    print np.log10(mbaryon_progenitor)
                    for subhalo in halo.subhaloes:
                        mbaryon_progenitor += np.sum(subhalo.progenitors[0].y[1:5])
                        print np.log10(np.sum(subhalo.progenitors[0].y[1:5]))

                    print "baryons in progenitors", np.log10(mbaryon_progenitor)

                    print ""
                    quit()
                if abs(Jbaryon != Constant.fb * halo.J_halo)/Jbaryon > 0.0001:
                    print "Error in Angular momentum conservation"
                    print Jbaryon, Constant.fb * halo.J_halo
                    quit()


        # Check circular velocity/ virial radii are self-consistent
        for node in nodes_ts:
            if node.vchalo > 0.0:
                rhalo_test = Constant.G * node.mchalo * Constant.Msun / np.square( node.vchalo * Constant.kms) / Constant.kpc
                if abs(rhalo_test - node.rchalo)/node.rchalo > 0.01:
                    print "Error, node = ",node, " rhalo and vhalo are not self-consistent"
                    print "rhalo from vhalo, mhalo is", rhalo_test
                    print "rhalo in code is", node.rchalo
                    quit()

        code_time_i2 = time.time()

        if len(level.NodesOnLevel()) > 0 and len(trees.trees)==1:
            print ilevel, " of ", nsnap, " tsim = ", level.t, " On this level, there are ", len(level.NodesOnLevel()), " nodes. This level took ", code_time_i2 - code_time_i1

    print "This merger tree took ", code_time_i2-code_time3

print "Finished, all the merger trees together took", code_time_i2-code_time2

print "writing merger trees to disk"
# For large merger trees, we need to increase the recursion limit quite a bit when writing the merger tree to disk.
sys.setrecursionlimit(50000)

pickle.dump(trees,open(Input_Data_Path.output_data_path+"merger_trees_with_galaxies_"+Parameter.model_name+".p","wb"))


print "done"
