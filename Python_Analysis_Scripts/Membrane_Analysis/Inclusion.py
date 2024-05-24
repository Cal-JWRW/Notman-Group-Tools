import MDAnalysis as mda
import lipyphilic
import numpy as np
import matplotlib.pyplot as plt
import os
from lipyphilic.lib.z_positions import ZPositions
from lipyphilic.lib.z_angles import ZAngles
from lipyphilic.lib.plotting import JointDensity
from lipyphilic.lib.z_thickness import ZThickness
from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.lib.plotting import ProjectionPlot
from tqdm import tqdm
import sklearn.mixture as skmix
import hmmlearn.hmm

class Inclusion_Analysis:
    def __init__(self, tpr, xtc, lipids, inclusion_res, HA, TA, HGR, HGA):
        self.tpr = tpr
        self.xtc = xtc
        self.lipids = lipids
        self.inclusion_res = inclusion_res
        self.HA = HA
        self.TA = TA
        self.HGR = HGR
        self.HGA = HGA

        plt.rcParams["figure.figsize"] = (9,7)
        STANDARD_SIZE=9
        SMALL_SIZE=7

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=STANDARD_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=STANDARD_SIZE)    # legend fontsize   


    def Inclusion_Amt_Calc(self):

        #Load Universe
        u = mda.Universe(self.tpr, self.xtc)

        #Extract Trajectory Length and Timestep
        trj_len = len(u.trajectory)
        dt = u.trajectory.dt

        times = np.arange(0, trj_len, 10)*dt 

        Lipid_Sel_String = ''
        for i in self.lipids:
            Lipid_Sel_String += f'(resname {i} and not name H*) or '

        Lipid_Sel_String = Lipid_Sel_String[:-4]

        height = ZPositions(
        universe=u,
        lipid_sel=Lipid_Sel_String,
        height_sel=f"name {self.HGA} and resname {self.HGR}"
        )
        height.run(step=10, start=0, verbose=True)
        cer_positions = height.z_positions

            #Calculate Average Lipid Headgroup height and std 
        tot_t = [] #Array to hold the absolute of every value
        mean_top = [] #Holds the mean of upper leaflet z positions
        mean_bot = [] #Holds the mean of lower leaflet z positions
        for i in range(np.shape(cer_positions)[1]): # Loops over every time frame
            #Two arrays to temporarily store the upper and lower z positions
            top_t = []
            bot_t = []
            for j in cer_positions[:, i]: #Loops over each N positionn at time i
                #Seperate them into the relevent upper and lower arrays
                if j >= 0:
                    top_t.append(j)
                    tot_t.append(j)
                else:
                    bot_t.append(j)
                    tot_t.append(abs(j))
            #Add the average position of the top and bottom to the mean_top/bot arrays
            mean_top.append(np.mean(top_t))
            mean_bot.append(np.mean(bot_t))

        cutoff = np.mean(tot_t) #Calculate the mean of the absolute positions

        #Calculate the standard deviation of the absolute positions
        std = 0
        for i in tot_t:
            std += (i-cutoff)**2
        std = np.sqrt(std/len(tot_t))
        stde = std/np.sqrt(len(tot_t))

        cutoff += std #Adjusts the cutoff to account for outlier association       

         #Extract Oleate Headgroup Z positions
        height = ZPositions(
            universe=u,
            lipid_sel=f"(resname CER3 and not name H*) or (resname LIGNP and not name H*)", 
            height_sel=f"resname {self.inclusion_res} and name {self.HA}"
        )
        height.run(step=10, start=0, verbose=True)
        positions = height.z_positions

        #Specify the upper and lower leaflet bounds for inclusion
        lower_bound = -cutoff
        upper_bound = cutoff
        print(f'Lower Bound:{lower_bound}, Upper Bound:{upper_bound}')
        #Generate an empty array to hold inclusion and transfer times
        inclusion_times = []
        transition_times = []

        #Loops over each oleate
        for ap in positions:
            #Sets our next line identifier < 0 
            next_l = -1
            for p in range(1, len(ap)): #Loops over oleate ap Z positions
                check = True
                if (lower_bound < ap[p] < upper_bound) and (p>next_l): #If Z position is within our cutoff and frame p is > next_l
                    for p2 in range(p+1, len(ap)):#From frame p we loop until the end of the trajectory

                        #If we have passed onto the other side of the membrane during this then registrer this as a transition only once
                        if (ap[p-1] < lower_bound) and (ap[p2] > (0+upper_bound/2)) and check==True: #If we have passed onto the other side of the membrane record the time this has taken (flip-flop)
                            transition_times.append((p2-p)*10*dt)
                            check = False
                        elif (ap[p-1] > upper_bound) and (ap[p2] < (0-upper_bound/2)) and check==True:
                            transition_times.append((p2-p)*10*dt)
                            check = False

                        if (ap[p2] < lower_bound) or (ap[p2] > upper_bound): #If we have exited the membrane
                            inclusion_times.append((p2-p)*10*dt) #Append the total time spent inside the membrane from entering    

                            next_l = p2
                            break

                        if p2 == len(ap)-1:
                            inclusion_times.append((p2-p)*10*dt)
                            next_l = p2
                            break

        inclusion_amt = []
        for i in range(np.shape(positions)[1]):
            inclusion = 0
            for k in positions[:, i]:
                if lower_bound < k < upper_bound:
                    inclusion += 1
            inclusion_amt.append(inclusion)




        return(inclusion_times, transition_times, times, mean_top, mean_bot, positions, inclusion_amt)

    def Inclusion_Free_Energy(self, start_time, step):

        #Load Universe
        u = mda.Universe(self.tpr, self.xtc)

        temperature = 303.15
        angle_bins = np.linspace(0, 180, 181)
        height_bins = np.linspace(-50, 50, 101)

        Lipid_Sel_String = ''
        for i in self.lipids:
            Lipid_Sel_String += f'(resname {i} and not name H*) or '

        Lipid_Sel_String = Lipid_Sel_String[:-4]

        height = ZPositions(
            universe=u,
            lipid_sel=Lipid_Sel_String,
            height_sel=f'resname {self.inclusion_res} and name {self.HA}'
        )
        height.run(start=start_time, step=step, verbose=True)

        angles = ZAngles(
            universe=u,
            atom_A_sel=f'name {self.HA} and resname {self.inclusion_res}',
            atom_B_sel=f'name {self.TA} and resname {self.inclusion_res}'
        )
        angles.run(start=start_time, step=step, verbose=True)

        pmf = JointDensity(
            ob1=angles.z_angles,
            ob2=height.z_positions
        )

        pmf.calc_density_2D(
            bins=(angle_bins, height_bins),
            temperature=temperature
        )
        pmf.interpolate()

        dx = np.diff(pmf.ob1_mesh_bins[0][:2])[0]
        dy = np.diff(pmf.ob2_mesh_bins[:, 0][:2])[0]
        extent = [
                pmf.ob1_mesh_bins[0][0] - dx / 2,
                pmf.ob1_mesh_bins[0][-1] + dx / 2 + 1,
                pmf.ob2_mesh_bins[0][0] - dy / 2,
                pmf.ob2_mesh_bins[-1][0] + dy / 2,
            ]





        return(pmf.joint_mesh_values, extent)

    def Hidden_Lipid_State(self, start, step, tail_1, tail_2, memb):

        def add_intermediate_state(means, sds):
            """
            Add an intermediate gaussian
            """
            small_mean, large_mean =means
            small_sd, large_sd = sds

            intermediate_mean = (small_mean * large_sd + large_mean * small_sd) / (small_sd+large_sd)

            intermediate_sd = min(
                abs(small_mean-intermediate_mean),
                abs(large_mean - intermediate_mean)

            )/3.0

            new_means = np.array([small_mean,intermediate_mean, large_mean])
            new_sds = np.array([small_sd, intermediate_sd, large_sd])
            return new_means, new_sds

        def get_gmm_params(gmm):
            means = gmm.means_.flatten()
            weights = gmm.weights_.flatten()
            sds = np.sqrt(gmm.covariances_).flatten()
            sorter = np.argsort(means)
            return means[sorter], weights[sorter], sds[sorter]

        def get_emission_states(thickness, means, n_states=9):
            """
            Bin the raw values into discrete emission states
            """
            small_mean, large_mean = means[0], means[-1]
            emission_states = np.full_like(thickness, np.nan, dtype=np.int8)
            emission_state_edges = np.linspace(small_mean, large_mean, n_states-1)

            # emission states < smaller_mean have an emission state if 0
            # Thickness greater than the upper
            emission_states[thickness<small_mean] = 0
            emission_states[thickness > large_mean] = n_states-1

            for edge_index, edge in enumerate(emission_state_edges[:-1], start=1):

                state_mask = np.logical_and(
                    thickness >= edge,
                    thickness < emission_state_edges[edge_index]
                )
                emission_states[state_mask]= edge_index

            return(emission_states, emission_state_edges)

        def get_emission_probs(thickness, states, edges, means, sds):
            """Calculate emission probabilities.
    
            An emission probability is the probability that a given hidden state
            emits a given emitted state.

            There are `n_states` emission probabilities for each hidden state,
            normalised such that the sum of emission probabilities for a given
            hidden state equals 1.

            The probability of an emitted state is calculated by integrating the
            Gaussian of a hidden state between the limits of emission states defined above.
            """

            def _calc_emission_prob(mean, sd, bin_edges):
                """
                Calculate the given emission probabilities for a given gaussian and bins
                """

                x = np.linspace(
                    mean - 3 * sd,
                    mean + 3 * sd,
                    1000
                )
                x_hist, _ = np.histogram(x, bins=bin_edges, density=True)

                return x_hist / np.sum(x_hist)

            #Setup intergration edges
            emission_prob_edges = np.full(edges.size + 2, fill_value = np.NaN)
            emission_prob_edges[0] = np.min(thickness)
            emission_prob_edges[1:-1] = edges
            emission_prob_edges[-1] = np.max(thickness)

            emission_probs = []
            for index, mean in enumerate(means):

                emission_prob = _calc_emission_prob(
                    mean,
                    sds[index],
                    emission_prob_edges
                )
                emission_probs.append(emission_prob)
            return np.asarray(emission_probs)

        u = mda.Universe(self.tpr, self.xtc)

        membrane = u.select_atoms(memb)
        thickness_1 = ZThickness(
            universe=u,
            lipid_sel = tail_1
        )
        thickness_1.run(start=start, step=step, verbose=True)

        thickness_2 = ZThickness(
            universe=u,
            lipid_sel = tail_2
        )
        thickness_2.run(start=start, step=step, verbose=True)

        thickness = ZThickness.average(
            thickness_1,
            thickness_2
        )

        lipid_order = np.zeros_like(thickness.z_thickness, dtype=np.int8)

        n_hidden_states = 2

        for species in tqdm(np.unique(membrane.resnames)):

            spesies_mask = membrane.residues.resnames == species
            spesies_thickness = thickness.z_thickness[spesies_mask]

            #Decompose the thickness into two gaussians
            mixture = skmix.GaussianMixture(
                n_components=2,
                covariance_type='full',
                tol=1e-3
            ).fit(spesies_thickness.flatten().reshape(-1,1))

            means, weights, sds = get_gmm_params(gmm=mixture)

            if n_hidden_states == 3:
                means, sds =  add_intermediate_state(means, sds)

            #Calculate HMM input parameters
            emission_states, emission_state_edges = get_emission_states(
                thickness=spesies_thickness,
                means=means,
                n_states=9
            )

            emission_probs = get_emission_probs(
                spesies_thickness,
                states=emission_states,
                edges=emission_state_edges,
                means = means,
                sds=sds
            )

            #Construct the Model
            n_lipids, n_frames = spesies_thickness.shape
            lengths = np.full(n_lipids, fill_value=n_frames)
            model = hmmlearn.hmm.CategoricalHMM(
                n_components=n_hidden_states,
                init_params='',
                verbose=True,
                n_iter=10000
            )
            model.emissionprob_ = emission_probs

            #Train the model
            model = model.fit(
                np.concatenate(emission_states).reshape(-1,1),
                lengths=lengths
            )

            #Decode most likely sequence of hidden states

            _, spesies_order = model.decode(
                np.concatenate(emission_states).reshape(-1,1),
                lengths = lengths
            )

            spesies_order = spesies_order.reshape(emission_states.shape)

            #Ensure Ld = -1, Ld/o = 0, L0 = 1
            if n_hidden_states == 3:
                spesies_order -= 1
            else:
                spesies_order[spesies_order == 0] = -1

            lipid_order[spesies_mask] = spesies_order

        bin_edges = np.linspace(0,50.0, 252)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)[0] /2 
        labels = [r"$L_d$", r"$L_o$"]

        for species in np.unique(membrane.resnames):
            fig, ax = plt.subplots()
            species_mask = membrane.residues.resnames == species
            species_thickness = thickness.z_thickness[species_mask]
            species_order = lipid_order[species_mask]
            hmm_weights = np.array([
                np.sum(species_order == -1),
                np.sum(species_order == 1),
            ]) / species_order.size
            for index, state in enumerate([-1,1]):
                hist, bins = np.histogram(species_thickness[species_order == state].flatten(), bins=bin_edges, density=True)
                ax.plot(bin_centers, hist * hmm_weights[state], label=labels[index])

                print(f"{species} {labels[index][1:-1]} mean thickness: {np.mean(species_thickness[species_order == state]):.2f} Å")

            ax.set_title(species, loc="left", weight="bold")
            ax.legend()
            plt.show()

        labels = [r"$L_d$", r"$L_o$"]
        for species in np.unique(membrane.resnames):
            fig, ax = plt.subplots()
            # extract Lo and Ld states for this species
            species_order = lipid_order[membrane.residues.resnames == species]

            # Count the number of each state (Ld or Lo) at each frame
            count_disordered = np.apply_along_axis(lambda x: np.count_nonzero(x==-1), axis=0, arr=species_order)
            count_ordered = np.apply_along_axis(lambda x: np.count_nonzero(x==1), axis=0, arr=species_order)

            # Combine these values into a 2d array for plotting
            order_by_frame = np.vstack([count_disordered, count_ordered])

            ax.stackplot(
                np.arange(order_by_frame.shape[1])*step,
                order_by_frame/np.sum(order_by_frame, axis=0),
                labels=labels,
                baseline="zero",
                lw=0.0,
                alpha=1
            )
           

            leg = ax.legend(fancybox=True, framealpha=1, loc=3, fontsize=20)
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            ax.set_title(species, loc="left", weight="bold")

            plt.show()

        
        return(lipid_order)

    def Assign_Leaflets(self):
        u = mda.Universe(self.tpr, self.xtc)
        memsel = 'resname '
        for l in self.lipids:
            memsel += f'{l} '
        leaflets = AssignLeaflets(
            universe=u,
            lipid_sel=memsel
        )

        leaflets.run(verbose=True)

        return(leaflets)

    def Project_Disorder(self, leaflets, lipid_order):
        u = mda.Universe(self.tpr, self.xtc)
        frame_index = -1
        u.trajectory[frame_index]
        memsel = 'resname '
        for l in self.lipids:
            memsel += f'{l} '


        membrane = u.select_atoms(memsel).residues

        x_pos, y_pos, _ = membrane.center_of_mass(compound="residues").T
        upper_mask = (leaflets.leaflets[:, frame_index]==1)

        upper_x_pos = x_pos[upper_mask]
        upper_y_pos = y_pos[upper_mask]
        upper_lipid_order = lipid_order[upper_mask, frame_index]

        projection = ProjectionPlot(
            x_pos=upper_x_pos,
            y_pos=upper_y_pos,
            values=upper_lipid_order
        )
        bins = np.linspace(0, u.dimensions[0], int(np.ceil(u.dimensions[0])+1))
        projection.project_values(bins=bins)
        projection.interpolate(method="linear")
        projection.plot_projection()
        return()



