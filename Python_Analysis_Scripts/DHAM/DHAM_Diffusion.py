import numpy as np

def Calculate_Diffusion(MM, states, lag):
    states = states * 1E-8
    lag = lag * 1E-13

    N = len(MM)

    # Initialize arrays to store drift and diffusion coefficients for each state
    drift_coefficients = np.zeros(N)
    diffusion_coefficients = np.zeros(N)

    # Calculate the transition rates
    R=MM
  
    #R = (transition_matrix - transition_matrix.T) / delta_t

    # Loop through each state (bin)
    for i in range(N):
        # Calculate the first Kramers-Moyal moment (drift) for state i
        drift_coefficients[i] = np.dot(R[:, i], states-states[i])
    
        # Calculate the second moment of (dX/dt)^2 for state i
        dXdt_sq = np.dot(R[:, i], (states-states[i])**2)
       
    
        # Calculate the diffusion coefficient for state i
        diffusion_coefficients[i] = (dXdt_sq - (drift_coefficients[i]**2))/(2*lag)

    return(diffusion_coefficients)