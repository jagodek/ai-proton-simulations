weighted loss but this time criterion changed to 0.001 to cover more data

weights_mask = np.ones((len(averaged_data_fluence),400))

if averaged_data_fluence[i][k] < 0.001*max_fluence_in_i:
    weights_mask[i][:k] = 100


Epoch    0 |  Train loss: 9.1459e-02 |  LR: 1.00e-03 Time: 1.23
Epoch   50 |  Train loss: 6.8612e-05 |  LR: 1.00e-03 Time: 15.80
Epoch  100 |  Train loss: 4.9166e-05 |  LR: 1.00e-03 Time: 30.79
Epoch  150 |  Train loss: 4.5226e-05 |  LR: 1.00e-03 Time: 45.16
Epoch  200 |  Train loss: 4.1028e-05 |  LR: 1.00e-03 Time: 59.20
Epoch  250 |  Train loss: 3.8236e-05 |  LR: 1.00e-03 Time: 73.87
Epoch  300 |  Train loss: 3.6573e-05 |  LR: 1.00e-03 Time: 88.22
Epoch  350 |  Train loss: 3.4187e-05 |  LR: 1.00e-03 Time: 102.20
Epoch  400 |  Train loss: 3.3275e-05 |  LR: 1.00e-03 Time: 116.55
Epoch  450 |  Train loss: 3.3591e-05 |  LR: 1.00e-03 Time: 131.18
Epoch  500 |  Train loss: 3.1776e-05 |  LR: 1.00e-03 Time: 145.22
Epoch  550 |  Train loss: 3.1037e-05 |  LR: 1.00e-03 Time: 158.04

--- Model Evaluation ---
Wholes Test MSELoss: 3.9624e+03
Halves Test MSELoss: 3.9555e+03