import numpy as np; from merger_tree_class import *; import cPickle; import time; import sys; from utilities_cosmology import *; from input_parameters import *
from halo_properties_routines import *; from cooling_routines import *; from merging_routines import *; import h5py

code_t0 = time.time()

# Read in merger trees data

input_file_path = Paths.input_path + Parameter.input_file
trees_file = h5py.File(input_file_path,'r')

a = np.array(trees_file["expansion_factor/a"])
ntrees = np.array(trees_file["ntrees/ntrees"])

# Loop over merger trees

trees = MergerTrees()

for i_tree in range(ntrees):

    print i_tree, " of ", ntrees

    tree_group = "trees/tree_"+str(i_tree)+"/"
    
    node_mass = np.array(trees_file[tree_group+"node_mass"])
    node_snapshot = np.array(trees_file[tree_group+"snapshot"])
    node_index = np.array(trees_file[tree_group+"node_index"])
    host_index = np.array(trees_file[tree_group+"host_index"])
    descendant_index = np.array(trees_file[tree_group+"descendantIndex"])
    isInterpolated = np.array(trees_file[tree_group+"isInterpolated"])
    isMainProgenitor = np.array(trees_file[tree_group+"isMainProgenitor"])

    node_data = np.array([node_index, host_index, descendant_index, isMainProgenitor]).astype('int64')

    nsnap = node_snapshot.max() +1
    nnodes = len(node_snapshot)

    # Build the merger tree


    tree = MergerTree()
    trees.trees.append(tree)

    node = 0
    node2 = 0 # Seperate index for checking isInterpolated
    for snap in range(nsnap):
        nnodes_ts = len(node_mass[node_snapshot==snap])

        tree.Create_Level(a[snap])

        for node_ts in range(nnodes_ts):

            # Create a halo if node_index = halo_index
            if node_index[node] == host_index[node]:
                tree.levels[snap].Create_Halo(node_data[:,node], node_mass[node])

            # Else, make a subhalo attached to the above halo
            else:
                tree.levels[snap].haloes[-1].Create_Subhalo(node_data[:,node], node_mass[node])

            node += 1

        for halo in tree.levels[snap].haloes:

            # Do not double count interpolated haloes in halo mass unless this node is the only thing in the halo
            if isInterpolated[node2] != 1 or len(halo.subhaloes)==0:
                halo.mchalo += halo.node_mass

            node2 += 1

            for subhalo in halo.subhaloes:

                #Do not double count interpolated subhaloes in halo mass
                if isInterpolated[node2] != 1:
                    halo.mchalo += subhalo.node_mass

                subhalo.mchalo = subhalo.node_mass

                node2 += 1


    # Define timesteps

    for snap in range(nsnap-1):

        level = tree.levels[snap]
        next_level = tree.levels[snap+1]

        t1 = t_Universe(level.a, Parameter.omm, Parameter.h)[0]
        t2 = t_Universe(next_level.a, Parameter.omm, Parameter.h)[0]

        # Full step size in Gyr
        dt = t2 - t1

        level.t = t1
        level.dt = dt

        if snap == nsnap-2:
            next_level.t = t2


    ############################## Connect descendants/progenitors. ####################################################

    for snap in range(nsnap-1):

        nodes_ts = tree.levels[snap].NodesOnLevel()
        nodes_ns = tree.levels[snap+1].NodesOnLevel()

        # Loop over nodes (haloes and subhaloes) in the current snapshot
        for node_ts in nodes_ts:

            has_descendant = False

            # Loop over nodes in the next snapshot to look for descendants
            for node_ns in nodes_ns:

                if node_ts.descendant_index == node_ns.node_index:
                    node_ts.descendant = node_ns
                    node_ns.progenitors.append(node_ts)

                    has_descendant = True

                    # A node can only have 1 descendant
                    break


            # For subhaloes that don't have a defined descendant index, point them to the descendant of their host
            # This is the case where subhaloes vanish out in the field. Can't think of a better way of dealing with this presently.
            if not node_ts.IsHalo() and node_ts.descendant_index == -1:

                node_ts.descendant = node_ts.host_halo.descendant
                node_ts.descendant.progenitors.append(node_ts)
                node_ts.isMainProgenitor = 0
                has_descendant = True

            # For rare cases where the subhalo does not have a descendant index that matches a node index, just point them to the descendant of their host
            # This can occur if the subhalo switches between final hosts that were stored in different merger tree files
            if not node_ts.IsHalo() and not has_descendant:
                node_ts.descendant = node_ts.host_halo.descendant
                node_ts.descendant.progenitors.append(node_ts)
                has_descendant = True
                node_ts.isMainProgenitor = 0


            # Check that each node has a descendant at the next timestep. Print diagnostic information if it doesn't
            if not has_descendant:
                print "Fatal Error: a node at snapshot level", snap, " doesn't have a descendant at the next level"
                print "Lost halo was of type", node_ts.__class__.__name__
                print "Descendant index = ", np.array([node_ts.descendant_index]).astype('int64')
                exit()

    ######## Check main progenitor status has been assigned correctly ###################

    for snap in range(nsnap-1):

        nodes_ts = tree.levels[snap].NodesOnLevel()

        # Perform checks for problem cases
        for node_ts in nodes_ts:

            # Check if a node has more than one "Main" progenitor
            n_main_progenitor = 0
            for progenitor in node_ts.progenitors:
                if progenitor.isMainProgenitor == 1:
                    n_main_progenitor += 1

            if n_main_progenitor > 1:
                print "Warning, a node has more than one main progenitor"
                print "Setting main progenitor as the most massive"
                print node_ts
                print node_ts.progenitors
                for progenitor in node_ts.progenitors:
                    print np.array([progenitor.node_index]).astype('int64'), np.array([progenitor.host_index]).astype('int64'), np.array([progenitor.descendant_index]).astype('int64'), progenitor.isMainProgenitor, progenitor.mchalo, progenitor.node_mass
                quit()

            # Check that all nodes with at least one progenitor have a main progenitor
            if len(node_ts.progenitors)>0 and n_main_progenitor == 0:
                print ""
                print "Warning, a node has progenitors but no main progenitor. If this is not a rare event, there is a bug. Snap = ", snap
                print "Setting single progenitor as the most main progenitor"

                if len(node_ts.progenitors)>1:
                    print "Error: there is more than one progenitor halo"
                    quit()
                else:
                    progenitor.isMainProgenitor = 1
                print ''


    ############# Ensure monotonic halo masses. Create empty progenitor nodes for all nodes once they appear for the first time. ###################

    for snap in range(nsnap):

        nodes_ts = tree.levels[snap].NodesOnLevel()

        # Loop over nodes (haloes and subhaloes) in the current snapshot
        for node_ts in nodes_ts:

            # Assign an empty progenitor node (of the same type as this node) to nodes that have no progenitors. Make this empty node the main progenitor.
            # This is so the galaxy formation code can avoid having haloes suddenly appearing over a substep
            if len(node_ts.progenitors) == 0 and snap > 0:
                empty_node_data = [-1, -1, node_ts.node_index, 1]
                empty_node_mass = 0.0
                if node_ts.IsHalo():
                    tree.levels[snap-1].Create_Halo(empty_node_data, empty_node_mass)
                    node_ts.progenitors.append( tree.levels[snap-1].haloes[-1] )
                    tree.levels[snap-1].haloes[-1].descendant = node_ts

                else:
                    # Check if the current host of this subhalo has a progenitor
                    current_host = node_ts.host_halo
                    nprogenitors_current_host = len(current_host.progenitors)

                    # If the current host has no progenitor, attach empty subhalo progenitor to empty halo progenitor of the host
                    if nprogenitors_current_host == 0:
                        tree.levels[snap-1].haloes[-1].Create_Subhalo(empty_node_data, empty_node_mass)
                        tree.levels[snap-1].haloes[-1].subhaloes[-1].descendant = node_ts
                        node_ts.progenitors.append(tree.levels[snap-1].haloes[-1].subhaloes[-1])

                    # If the current host has progenitor(s), attach empty subhalo progenitor to the most massive existing progenitor halo
                    if nprogenitors_current_host >= 1:
                        max_progenitor_mass = 0.0
                        for progenitor in current_host.progenitors:
                            if progenitor.mchalo > max_progenitor_mass:
                                previous_host = progenitor
                                max_progenitor_mass = progenitor.mchalo

                        # Ensure we attach the subhalo progenitor to a progenitor halo (not subhalo)
                        if not previous_host.IsHalo():
                            nodes_ps = tree.levels[snap-1].NodesOnLevel() # List of nodes from the previous level
                            previous_host = previous_host.host_halo

                        previous_host.Create_Subhalo(empty_node_data, empty_node_mass)
                        previous_host.subhaloes[-1].descendant = node_ts
                        node_ts.progenitors.append(previous_host.subhaloes[-1])

            # Ensure host haloes do not lose mass. i.e. the halo mass should never be less than the sum of all the haloes from the previous snapshot that end up inside this halo
            if node_ts.IsHalo():
                combined_halo_progenitor_mass = 0.0
                for progenitor in node_ts.progenitors:               
                    if progenitor.IsHalo():
                        combined_halo_progenitor_mass += progenitor.mchalo

                for subhalo in node_ts.subhaloes:
                    for progenitor in subhalo.progenitors:
                        if progenitor.IsHalo():
                            combined_halo_progenitor_mass += progenitor.mchalo

                if node_ts.mchalo < combined_halo_progenitor_mass:
                    node_ts.mchalo = combined_halo_progenitor_mass

            # Store maximum past mass of each subhalo as mchalo
            if not node_ts.IsHalo():
                for progenitor in node_ts.progenitors:
                    if progenitor.isMainProgenitor == 1:
                        node_ts.mchalo = max(progenitor.mchalo ,node_ts.mchalo)


    # Append newly formed Type 2 satellites to the merger tree

    for snap in range(nsnap-1):

        level = tree.levels[snap]

        for halo in level.haloes:

            # In this case, we have a halo merging onto a subhalo.
            if halo.isMainProgenitor == 0 and halo.descendant.IsHalo() == False:

                # Make a type 2 satellite attached to the descendant subhaloes host halo at the next timestep
                type2_node_data = [-1, -1, -1, 1]
                halo.isMainProgenitor = 1 # This halo is now the main progenitor of the new type 2 satellite
                halo.descendant.host_halo.Create_Subhalo(type2_node_data, halo.mchalo) # Create the Type 2 satellite
                halo.descendant.progenitors.remove(halo) # Remove this halo as a progenitor from the progenitors of the subhalo
                halo.descendant = halo.descendant.host_halo.subhaloes[-1] # Set this haloes descendant as the new type 2 satellite
                halo.descendant.progenitors.append(halo) # Add this halo as a progenitor to the the new Type 2 satellite
                halo.descendant.isType2 = True
                halo.descendant.mchalo = halo.mchalo


            # In this case, we have a halo merging onto a halo
            elif halo.isMainProgenitor == 0 and halo.descendant.IsHalo():

                # Make a Type 2 satellite attached to the descendant halo at the next timestep
                type2_node_data = [-1, -1, -1, 1]
                halo.isMainProgenitor = 1 # This halo is now the main progenitor of the new type 2 satellite
                halo.descendant.Create_Subhalo(type2_node_data, halo.mchalo) # Create the Type 2 satellite
                halo.descendant.progenitors.remove(halo) # Remove this halo as a progenitor from the progenitors of the halo
                halo.descendant = halo.descendant.subhaloes[-1] # Set this haloes descendant as the new type 2 satellite
                halo.descendant.progenitors.append(halo) # Add this halo as a progenitor to the the new Type 2 satellite
                halo.descendant.isType2 = True
                halo.descendant.mchalo = halo.mchalo


            for subhalo in halo.subhaloes:

                # In this case we have a subhalo merging onto a subhalo
                if subhalo.isMainProgenitor == 0 and subhalo.descendant.IsHalo()==False:

                    # Our response to this is to make this subhalo a type 2 satellite of the descendant subhalo's host halo
                    type2_node_data = [-1, -1, -1, 1]
                    subhalo.isMainProgenitor = 1 # This subhalo is now the main progenitor of the new type 2 satellite
                    subhalo.descendant.host_halo.Create_Subhalo(type2_node_data, subhalo.mchalo) # Create the Type 2 satellite
                    subhalo.descendant.progenitors.remove(subhalo) # Remove this subhalo as a progenitor from the progenitors of the subhalo
                    subhalo.descendant = subhalo.descendant.host_halo.subhaloes[-1] # Set this subhalo's descendant as the new type 2 satellite
                    subhalo.descendant.progenitors.append(subhalo) # Add this subhalo as a progenitor to the the new Type 2 satellite
                    subhalo.descendant.isType2 = True
                    subhalo.descendant.mchalo = subhalo.mchalo

                # In this case we have a subhalo merging onto a halo
                elif subhalo.isMainProgenitor == 0 and subhalo.descendant.IsHalo():

                    # Make a Type 2 satellite attached to the descendant halo at the next timestep
                    type2_node_data = [-1, -1, -1, 1]
                    subhalo.isMainProgenitor = 1 # This subhalo is now the main progenitor of the new type 2 satellite
                    subhalo.descendant.Create_Subhalo(type2_node_data, subhalo.mchalo) # Create the Type 2 satellite
                    subhalo.descendant.progenitors.remove(subhalo) # Remove this subhalo as a progenitor from the progenitors of the halo
                    subhalo.descendant = subhalo.descendant.subhaloes[-1] # Set this subhalo's descendant as the new type 2 satellite
                    subhalo.descendant.progenitors.append(subhalo) # Add this subhalo as a progenitor to the the new Type 2 satellite
                    subhalo.descendant.isType2 = True
                    subhalo.descendant.mchalo = subhalo.mchalo





    # Add descendants of Type 2 satellites to the tree. Note, these never dissapear here but will be pruned from the tree in the galaxy formation calculation once they run out of baryons.

    for snap in range(nsnap-1):

        level = tree.levels[snap]
        next_level = tree.levels[snap+1]

        nodes = level.NodesOnLevel()

        type2_node_data = [-1, -1, -1, 1]

        for node in nodes:
            if node.isType2:

                # Check if the descendant of this satellites host is a halo. If it is attach, this type 2 satellite to the descendant at the next step
                if node.host_halo.descendant.IsHalo():
                    node.host_halo.descendant.Create_Subhalo(type2_node_data, node.mchalo) # Create the type 2 at the next step
                    node.descendant = node.host_halo.descendant.subhaloes[-1] # Set this Type 2's descendant as the type 2 on the next step
                    node.descendant.progenitors.append(node) # Add this Type 2 as a progenitor to the Type 2 on the next step
                    node.descendant.mchalo = node.mchalo
                    node.descendant.isType2 = True

                # If not, attach this type 2 to the host halo of the descendant subhalo at the next step
                else:
                    descendant_temp = node.host_halo.descendant.host_halo # Host halo at the next step
                    descendant_temp.Create_Subhalo(type2_node_data, node.mchalo) # Create the type 2 at the next step
                    node.descendant = descendant_temp.subhaloes[-1] # Set this Type 2's descendant as the type 2 on the next step
                    node.descendant.progenitors.append(node) # Add this Type 2 as a progenitor to the Type 2 on the next step
                    node.descendant.mchalo = node.mchalo
                    node.descendant.isType2 = True

        # Set isMainProgenitor to 0 for the final snapshot (so that Type 2 sats are consistent with everything else)
        if snap == nsnap-2:
            for node in next_level.NodesOnLevel():
                node.isMainProgenitor = 0


    # Add merger clocks to satellite galaxies

    for snap in range(nsnap-1):

        level = tree.levels[snap]
        next_level = tree.levels[snap+1]
        nodes = level.NodesOnLevel()

        for node in nodes:
            if not node.IsHalo():

                # Check to see if satellite has been assigned a merger clock yet
                if not hasattr(node,"t_since_infall"):
                    # Set t_since_infall = 0, the time since the infall of the satellite
                    node.t_since_infall = 0.0
                    # Set the time until this satellite will merge (assuming nothing happens to the host)
                    node.t_merge = Calculate_Merging_Timescale(node)

                # Otherwise, pass on merger clock info to descendants (if the descendant is a satellite)
                descendant = node.descendant

                if not descendant.IsHalo():

                    # If the host of this subhalo is the main progenitor of the host of the descendant, pass on the clock
                    if node.host_halo.isMainProgenitor == 1 and node.host_halo.descendant == descendant.host_halo:
                        descendant.t_since_infall = node.t_since_infall + level.dt
                        descendant.t_merge = node.t_merge

                    # Otherise (this subhalo has been permanently accreted onto a different host), In this case the merger clock will be reset at the next step

                    # Alternatively (exception case), if the host of this subhalo is the main progenitor of the host of the descendant of the descendant, pass on the clock
                    # This might seem bizarre, but it catches those cases where subfind swaps the central/satellite defn between two density peaks       
                    if node.host_halo.isMainProgenitor ==1 and snap < nsnap-2:
                        if not descendant.descendant.IsHalo():
                            if node.host_halo.descendant.descendant == descendant.descendant.host_halo:

                                descendant.t_since_infall = node.t_since_infall + level.dt
                                descendant.t_merge = node.t_merge

                                descendant.descendant.t_since_infall = node.t_since_infall + level.dt + next_level.dt
                                descendant.descendant.t_merge = node.t_merge






    ##### Tag the main progenitor of the final host halo. Now that each node has only one progenitor, we can do this iteratively

    level = tree.levels[-1]
    nodes = level.NodesOnLevel()
    node_MP = level.haloes[0]
    node_MP.isMainProgenitor_Final = True
    while len(node_MP.progenitors)>0:
        node_MP = node_MP.progenitors[0]
        node_MP.isMainProgenitor_Final = True



    ################## Search for halo formation events ##################################


    for snap in range(nsnap):

        level = tree.levels[snap]
        nodes_ts = tree.levels[snap].NodesOnLevel()

        for node in nodes_ts:

            # If the node is not the main progenitor, there will be no formation events over this step
            if node.isMainProgenitor == 0:
                node.t_formation_events = []
                node.m_formation_events = []
                node.a_formation_events = []

            # If the node is a main progenitor of it's descendant, search for formation events over this step
            else:
                node.t_formation_events = []
                node.m_formation_events = []
                node.a_formation_events = []

                # If this an empty node, set mform = mform_min
                if node.mchalo == 0.0:
                    node.mform = Parameter.mform_min
                    node.tform = level.t
                    node.aform = level.a

                # Calculate halo mass accretion rate in Msun Gyr^-1
                hfr = (node.descendant.mchalo - node.mchalo) / level.dt

                # Function that returns the time during this step that the halo has a specified mass, mhalo_form
                t_of_mhalo = lambda mhalo_form, hfr, mhalo_start, t_start: (mhalo_form-mhalo_start) / hfr + t_start

                # Search formation events within this step
                mform_temp = node.mform
                tform_temp = node.tform
                aform_temp = node.aform

                while Parameter.fform * mform_temp <= node.descendant.mchalo:

                    # There is a formation event here during the step. Store the formation mass
                    mform_temp = Parameter.fform * mform_temp
                    node.m_formation_events.append(mform_temp)

                    # Now calculate when it occured
                    tform_temp = t_of_mhalo(mform_temp, hfr, node.mchalo, level.t)
                    node.t_formation_events.append(tform_temp)

                    # Calcuate corresponding expansion factor
                    aform_temp = a_Universe(tform_temp, Parameter.omm, Parameter.h)
                    node.a_formation_events.append(aform_temp)

                # Set mform, tform and aform for the descendant of this node
                node.descendant.mform = mform_temp
                node.descendant.tform = tform_temp
                node.descendant.aform = aform_temp

    ################## Calculate halo angular momentum ##########################################

    # This will involve calculating vhalo, strc and rhalo for the start of each main step for each node
    # This is also where halo spins are assigned

    for snap in range(nsnap-1):

        level = tree.levels[snap]
        nodes_ts = tree.levels[snap].NodesOnLevel()

        a_next = tree.levels[snap+1].a

        for node in nodes_ts:

            # If this is an empty node, initialise the spin, rhalo, vhalo, strc and angular momentum of the halo
            if node.mchalo == 0:
                node.J_halo = 0.0
                node.spin = Halo_Spin()
                node.strc = 0.0
                node.rchalo = 0.0
                node.vchalo = 0.0

            # If this node is a main progenitor, calculate vhalo, rhalo, strc and the angular momentum of that descendant at the start of the next main step

            if node.isMainProgenitor == 1:

                spin_next = node.spin

                if Parameter.propagate_vhalo:
                    vchalo_next = Calculate_Halo_Virial_Velocity(node.descendant.mform, node.descendant.aform)
                else:
                    vchalo_next = Calculate_Halo_Virial_Velocity(node.descendant.mchalo, a_next)

                if Parameter.propagate_strc:
                    strc_next = NFW_Scale_Radius(node.descendant.mform, node.descendant.aform)
                else:
                    strc_next = NFW_Scale_Radius(node.descendant.mchalo, a_next)

                rchalo_next = Calculate_Halo_Virial_Radius(node.descendant.mchalo, vchalo_next)

                # Calculate the halo angular momentum at the next timestep
                v_rot_norm = Calculate_V_Rotation_Normalisation(strc_next)
                J_o_Vrot = Calculate_J_o_Vrot(strc_next)
                Vrot = v_rot_norm * vchalo_next * spin_next # kms^-1
                J_halo_next = J_o_Vrot * Vrot * rchalo_next * node.descendant.mchalo # kms^-1 kpc Msun

                node.descendant.J_halo = J_halo_next
                node.descendant.spin = spin_next
                node.descendant.rchalo = rchalo_next
                node.descendant.vchalo = vchalo_next
                node.descendant.strc = strc_next

                # Check that halo/subhalo angular momentum does not decrease with time.
                if node.descendant.J_halo < node.J_halo:
                    print "Warning: halo angular momentum decreased over time. Snap = ", snap, node
                    print np.log10(node.J_halo), np.log10(node.descendant.J_halo)
                    print np.log10(node.mchalo), np.log10(node.descendant.mchalo)
                    print node, node.descendant
                    print "This can happen in rare instances"
                    print "Conserving halo angular momentum"
                    node.descendant.J_halo = node.J_halo

                    '''print "vchalo, vchalo_next"
                    print node.vchalo, vchalo_next
                    print "rchalo, rchalo_next"
                    print node.rchalo, rchalo_next
                    print node.strc, strc_next
                    print np.log10(node.mform), np.log10(node.descendant.mform)
                    print node.aform, node.descendant.aform
                    print ""
                    print Calculate_Halo_Virial_Radius(node.mchalo, node.vchalo), Calculate_Halo_Virial_Radius(node.descendant.mchalo, vchalo_next)
                    quit()'''


    ################ Calculate halo accretion rates #############################################################

    for snap in range(nsnap-1):

        level = tree.levels[snap]
        nodes_ts = tree.levels[snap].NodesOnLevel()

        a_next = tree.levels[snap+1].a

        for node in nodes_ts:

            # There is only accretion if this node is a main progenitor
            if node.isMainProgenitor == 0:
                node.hfr = 0.0; node.Jdot_halo = 0.0

            if node.isMainProgenitor == 1:

                # Calculate halo accretion rates
                node.hfr = (node.descendant.mchalo - node.mchalo) / level.dt # Msun Gyr^-1
                node.Jdot_halo = (node.descendant.J_halo - node.J_halo) / level.dt

                if node.hfr < 0.0 or node.Jdot_halo < 0.0:
                    print "Error, negative halo M/J accretion"
                    quit()


################## Write output #############################################################

print "done"

code_t1 = time.time()
total = code_t1-code_t0
print "time taken = ", total


print "writing merger tree to disk"
# For large merger trees, we need to increase the recursion limit quite a bit when writing the merger tree to disk.
sys.setrecursionlimit(50000)

outpath = "/gpfs/data/d72fqv/PythonSAM/halo_data/"

cPickle.dump(trees,open(Paths.output_path+Parameter.model_name+"/merger_trees.p","wb"))


print "done"
