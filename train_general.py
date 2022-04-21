import numpy as np
from models.keras_nn import hazardNN
from models.baseline_nn import baselineNN
from models.simulator import SimulatedData
import time



N = 2500                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
N_sim = 100

generator = SimulatedData()

nn_config = {
    "hidden_layers_nodes": 64,
    "learning_rate": 0.001,
    "activation": 'relu', 
    "optimizer": 'adam',
    "batch_size": 100,
    "patience": 10
}



baseline_elapsed_list = []
new_elapsed_list = []

min_min_y = 0
max_max_y = 0
min_y_list = []

##### training procedure

for i in range(N_sim):

    seed = i+1
    print("################## Simulation",i+1,"#####################")


    train_data, train_y = generator.generate_data(N,seed)
    valid_data, valid_y = generator.generate_data(N,seed+N_sim)
    
    
    min_min_y = min(min_min_y,min(train_y))
    max_max_y = max(max_max_y,max(train_y))
    min_y_list.append(min(train_y))

    l2_min_loss = np.inf
    min_loss = np.inf

    for j in range(1): # different initial weights

        print("random initial weight: ",j+1)
        start = time.time()

        baseline_model = baselineNN(nn_config)
        x_mean, x_std, loss = baseline_model.train(
            x_train = train_data.groupby('id').first().iloc[:,1:-2],
            y_train = train_y,
            x_valid = valid_data.groupby('id').first().iloc[:,1:-2],
            y_valid = valid_y
        )
        
        if loss<l2_min_loss:
            baseline_model.model.save('saved_models/l2_model_{}.h5'.format(i))
            np.save('saved_models/l2_model_standard_{}.npy'.format(i),[x_mean,x_std])
            l2_min_loss = loss

        baseline_elapsed = (time.time() - start)
        baseline_elapsed_list.append(baseline_elapsed)

        start = time.time()

        model = hazardNN(nn_config)
        x_mean, x_std, loss = model.train(
            x_train = train_data.groupby('id').apply(lambda x: x.values[1:,2:]), # set aside the first time point
            fail_train = train_data.groupby('id').apply(lambda x: x.values[1:,1]),
            x_valid = valid_data.groupby('id').apply(lambda x: x.values[1:,2:]),
            fail_valid = valid_data.groupby('id').apply(lambda x: x.values[1:,1])
        )

        if loss<min_loss:
            model.model.save('saved_models/model_{}.h5'.format(i))
            np.save('saved_models/model_standard_{}.npy'.format(i),[x_mean,x_std])
            min_loss = loss

        new_elapsed = (time.time() - start)
        new_elapsed_list.append(new_elapsed)

    #model.model.save('saved_models/model_{}.h5'.format(i)) #final initial weight
    #np.save('saved_models/model_standard_{}.npy'.format(i),[x_mean,x_std])

    np.save('saved_models/model_train_y_{}.npy'.format(i),train_y)

print(np.mean(baseline_elapsed_list),np.mean(new_elapsed_list))
print(min_min_y,max_max_y)

