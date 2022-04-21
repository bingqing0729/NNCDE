import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.keras_nn import hazardNN
from models.simulator import SimulatedSurvival
from models.cox import cox_pred
import tensorflow as tf

baseline = np.array(pd.read_csv("data/baseline.csv",header=None)[0])
cox = 1
generator = SimulatedSurvival(100,0.01,cox,baseline) # tau, step

N = 2500
N_sim = 100

test_N = 10
test_data,_ = generator.generate_data(N=test_N,test=True)


nn_config = {
    "hidden_layers_nodes": 64,
    "learning_rate":0.001,
    "activation": 'relu', 
    "optimizer": 'adam',
    "batch_size": 100,
    "patience": 10
}

pred_cox_df = pd.DataFrame()
pred_nn_df = pd.DataFrame()
beta_list = []
min_max_t = 100
max_max_t = 0

for i in range(N_sim):

    seed = i+1

    print("################## Simulation",i+1,"#####################")

    train_data, train_max_t = generator.generate_data(N,seed)
    valid_data, valid_max_t = generator.generate_data(N,seed+N_sim)
    valid_data['id'] = valid_data['id']+500
    max_t = max(train_max_t,valid_max_t)
    min_max_t = min(min_max_t,max_t)
    max_max_t = max(max_max_t,max_t)

    data = pd.concat([train_data,valid_data],ignore_index=True)

    # cox
    test_data, beta = cox_pred(data,test_data)

    pred_cox_df = pd.concat([pred_cox_df,test_data['pred_survival_cox']],axis=1)
    beta_list.append(beta)


    model = hazardNN(nn_config)

    x_mean, x_std, loss = model.train(
        x_train = train_data.groupby('id').apply(lambda x: x.values[:,2:]),
        fail_train = train_data.groupby('id').apply(lambda x: x.values[:,1]),
        x_valid = valid_data.groupby('id').apply(lambda x: x.values[:,2:]),
        fail_valid = valid_data.groupby('id').apply(lambda x: x.values[:,1])
    )

    x_test = test_data.groupby('id').apply(lambda x: x.values[:,1:8])
    x_test = tf.ragged.constant(x_test).to_tensor()
    dt_test = tf.expand_dims(x_test[:,:,-1]-x_test[:,:,-2],-1)  
    x_test = (x_test-x_mean)/x_std

    hazard = np.exp(model.model.predict([x_test,dt_test]).flatten())

    model2 = hazardNN(nn_config)
    x_mean, x_std, loss = model2.train(
        x_train = valid_data.groupby('id').apply(lambda x: x.values[:,2:]),
        fail_train = valid_data.groupby('id').apply(lambda x: x.values[:,1]),
        x_valid = train_data.groupby('id').apply(lambda x: x.values[:,2:]),
        fail_valid = train_data.groupby('id').apply(lambda x: x.values[:,1])
    )

    x_test = test_data.groupby('id').apply(lambda x: x.values[:,1:8])
    x_test = tf.ragged.constant(x_test).to_tensor()
    dt_test = tf.expand_dims(x_test[:,:,-1]-x_test[:,:,-2],-1)  
    x_test = (x_test-x_mean)/x_std

    hazard2 = np.exp(model.model.predict([x_test,dt_test]).flatten())


    test_data['pred_survival_nn'] = (hazard+hazard2)/2
    test_data['pred_survival_nn'] = test_data['pred_survival_nn']*(test_data['end']-test_data['start'])
    test_data['pred_survival_nn'] = test_data.groupby(['id'])['pred_survival_nn'].cumsum()
    test_data['pred_survival_nn'] = np.exp(-test_data['pred_survival_nn'])

    pred_nn_df = pd.concat([pred_nn_df,test_data['pred_survival_nn']],axis=1)


test_data['pred_cox_mean'] = pred_cox_df.mean(axis=1)
test_data['pred_cox_0.1'] = pred_cox_df.quantile(q=0.1,axis=1)
test_data['pred_cox_0.9'] = pred_cox_df.quantile(q=0.9,axis=1)

test_data['pred_nn_mean'] = pred_nn_df.mean(axis=1)
test_data['pred_nn_0.1'] = pred_nn_df.quantile(q=0.1,axis=1)
test_data['pred_nn_0.9'] = pred_nn_df.quantile(q=0.9,axis=1)


print(test_data)
#np.save('test_surv_df.npy',test_data)
print(np.mean(beta_list))

for j in range(test_N):
    plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'survival'],label="truth",color='black')
    plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_cox_mean'],'--',label="cox model",color='green')
    #plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_cox_0.1'],':',color='green')
    #plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_cox_0.9'],':',color='green')
    
    plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_nn_mean'],'-.',label="new model",color='orange')
    plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_nn_0.1'],':',label='confidence band',color='orange')
    plt.plot(test_data.loc[test_data['id']==j,'start'],test_data.loc[test_data['id']==j,'pred_nn_0.9'],':',color='orange')
    if j==0: plt.legend()
    plt.ylim(0,1)
    #plt.axvline(x=min_max_t,color='grey', linestyle='--')
    #plt.axvline(x=max_max_t,color='grey', linestyle='--')
    if cox:
        plt.savefig("results/cox_figure_{}.png".format(j))
    else:
        plt.savefig("results/noncox_figure_{}.png".format(j))
    plt.close()
    #plt.show()



