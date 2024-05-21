import MDAnalysis as mda
import lipyphilic
import numpy as np
import matplotlib.pyplot as plt
import os
from lipyphilic.lib.z_positions import ZPositions
from lipyphilic.lib.z_angles import ZAngles
from lipyphilic.lib.plotting import JointDensity

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




