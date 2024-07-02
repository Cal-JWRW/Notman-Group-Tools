import numpy as np
np.seterr(over='ignore')
np.seterr(invalid='ignore')

def Construct_MM(count_matrices, bias_matrices, T):
        """
        Constructs and unbiases a markov model based off the transition count matrices c[i,j]
        c[i, :] ie. row i represents all the transitions from intiial state i
        c[:, i] ie column i represents all transitions into state i
        numerator = sum observed transitions from state i to state j
        denominator = sum observed occupation of state i
        """
        num_simulations, num_bins, _ = np.shape(count_matrices)
        MM = np.zeros((num_bins, num_bins), dtype=np.float128)

        # Loop over each pair of bins i, j
        for i in range(num_bins):
                for j in range(num_bins):
                     
                        numerator = np.sum(count_matrices[:,i, j])
                        denominator = 0.0

                        for k in range(num_simulations):

                                n_i_k = np.sum(count_matrices[k,i, :])
                                u_j_k = bias_matrices[j, k]
                                u_i_k = bias_matrices[i, k]
                                denominator += n_i_k * np.exp(-(u_j_k-u_i_k)/2)


                        if denominator != 0:
                                MM[i, j] = numerator / denominator

        MM = np.nan_to_num(MM, nan=0)


        MM = MM / np.sum(MM, axis=1)[:, None]
        return(MM)

def Initial_MM(sumtr, trvec, biaspot, states, traj, temp):
    
        
        MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
        qsp = states[1] - states[0]
        for i in range(sumtr.shape[0]):
            for j in range(sumtr.shape[1]):
                if sumtr[i, j] > 0:
                    sump1 = 0.0
                    for k in range(trvec.shape[0]):
                        u = 0.5 * biaspot * np.square(traj[k,0] - states - qsp / 2) / (temp*1.9872041E-3)
                        if trvec[k, i] > 0:
                            sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / 2)
                    MM[i, j] = sumtr[i, j] / sump1
        MM = MM / np.sum(MM, axis=1)[:, None]
       
        return MM