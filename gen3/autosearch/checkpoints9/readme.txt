demonstration of how 0.01 in 

weights_mask = np.ones((len(averaged_data_fluence),400))
if averaged_data_fluence[i][k] < 0.01*max_fluence_in_i:
    weights_mask[i][:k] = 100

in creating weights ruined predictions




