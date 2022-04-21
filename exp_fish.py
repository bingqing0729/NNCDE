import imp
import numpy as np
import pandas as pd
from tensorflow.python.ops.gen_array_ops import ListDiff
from models.keras_nn import hazardNN
from models.baseline_nn import baselineNN
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1234)

num_nodes_list = [64]
lr_list = [0.001]
batch_size_list = [100]
nn_config_list = []
for lr in lr_list:
    for batch_size in batch_size_list:
        for num_nodes in num_nodes_list:
            nn_config = {
                "hidden_layers_nodes": num_nodes,
                "learning_rate":lr,
                "activation": 'relu', 
                "optimizer": 'adam',
                "batch_size": batch_size,
                "patience": 10
            }
            nn_config_list.append(nn_config)


data = pd.read_csv('data/qsar_fish_toxicity.csv',header=None,sep=';')
data = data.sample(frac=1).reset_index(drop=True)

kf = KFold(n_splits=5)

fold_n = 0
mse_list = []
median_se_list = []
r2_list = []
pred_y_all = []
test_y_all = []

for train_index, test_index in kf.split(data):

    fold_n += 1
    print(fold_n)

    df_train = data.iloc[train_index]
    df_test = data.iloc[test_index]
    df_val = df_train.sample(frac=0.5)
    df_train = df_train.drop(df_val.index)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # expand the training data frame
    sorted_train = sorted(np.unique(df_train.loc[:,6]))
    N_train = len(df_train)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N_train):
        d = df_train.loc[i,6]
        start = 0
        for time in sorted_train:
            if time > d: break
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time
            if time < d: fail_list.append(0)
            else: fail_list.append(1)

    df = pd.DataFrame({'id':id_list,'fail':fail_list,'start':start_list,'end':end_list})
    df_train['id'] = df_train.index
    df_train = pd.merge(df_train,df)
    df_train = df_train.drop(columns=6)

    # expand the validation data frame
    sorted_val = sorted(np.unique(df_val.loc[:,6]))
    N_val = len(df_val)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N_val):
        d = df_val.loc[i,6]
        start = 0
        for time in sorted_val:
            if time > d: break
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time
            if time < d: fail_list.append(0)
            else: fail_list.append(1)

    df = pd.DataFrame({'id':id_list,'fail':fail_list,'start':start_list,'end':end_list})
    df_val['id'] = df_val.index
    df_val = pd.merge(df_val,df)
    df_val = df_val.drop(columns=6)

    # expand the test data frame by training partition
    test_y = np.array(df_test.loc[:,6])

    N_test = len(df_test)
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N_test):
        start = 0
        for time in sorted_train:
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time

    df = pd.DataFrame({'id':id_list,'start':start_list,'end':end_list})
    df_test['id'] = df_test.index
    df_test_1 = pd.merge(df_test,df)
    df_test_1 = df_test_1.drop(columns=6)

    # expand the test data frame by validation partition
    id_list = []
    start_list = []
    end_list = []
    fail_list = []

    for i in range(N_test):
        start = 0
        for time in sorted_val:
            id_list.append(i)
            start_list.append(start)
            end_list.append(time)
            start = time

    df = pd.DataFrame({'id':id_list,'start':start_list,'end':end_list})
    df_test_2 = pd.merge(df_test,df)
    df_test_2 = df_test_2.drop(columns=6)

    min_loss = np.inf
    # grid search best model
    for nn_config in nn_config_list:

        print(nn_config)
        # model 1
        model_1_tmp = hazardNN(nn_config)
        x_mean_1_tmp, x_std_1_tmp, loss_1 = model_1_tmp.train(
            x_train = df_train.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,8,9]]),
            fail_train = df_train.groupby('id').apply(lambda x: x.values[:,7]),
            x_valid = df_val.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,8,9]]),
            fail_valid = df_val.groupby('id').apply(lambda x: x.values[:,7])
            )

        # model 2
        model_2_tmp = hazardNN(nn_config)
        x_mean_2_tmp, x_std_2_tmp, loss_2 = model_2_tmp.train(
            x_train = df_val.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,8,9]]),
            fail_train = df_val.groupby('id').apply(lambda x: x.values[:,7]),
            x_valid = df_train.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,8,9]]),
            fail_valid = df_train.groupby('id').apply(lambda x: x.values[:,7])
            )

        loss = (loss_1+loss_2)/2
        if loss<min_loss:
            min_loss = loss
            model_1, model_2 = model_1_tmp, model_2_tmp
            x_mean_1, x_std_1 = x_mean_1_tmp, x_std_1_tmp
            x_mean_2, x_std_2 = x_mean_2_tmp, x_std_2_tmp


    x_test_1 = df_test_1.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,7,8]])
    x_test_1 = tf.ragged.constant(x_test_1).to_tensor()
    dt_test_1 = tf.expand_dims(x_test_1[:,:,-1]-x_test_1[:,:,-2],-1)  
    x_test_1 = (x_test_1-x_mean_1)/x_std_1

    df_test_1['pred_cdf'] = np.exp(model_1.model.predict([x_test_1,dt_test_1]).flatten())
    df_test_1['pred_cdf'] = df_test_1['pred_cdf']*(df_test_1['end']-df_test_1['start'])
    df_test_1.loc[df_test_1.groupby('id').head(1).index, 'pred_cdf'] = -np.log(1-1/N_train) # manually set the first cumulative hazard value
    df_test_1['pred_cdf'] = df_test_1.groupby(['id'])['pred_cdf'].cumsum()
    df_test_1['pred_cdf'] = 1-np.exp(-df_test_1['pred_cdf'])
    df_test_1.groupby('id').last()['pred_cdf'] = 1 # enforce the last cdf to be 1
    pred_y_1_median = df_test_1[df_test_1['pred_cdf']<=0.5].groupby(['id']).last()['end']
    df_test_1['pred_pmf'] = df_test_1.groupby(['id'])['pred_cdf'].diff().fillna(df_test_1['pred_cdf'])
    df_test_1['pred_y'] = df_test_1['pred_pmf']*df_test_1['end']
    pred_y_1 = df_test_1.groupby(['id'])['pred_y'].sum()
    cdf_1 = df_test_1.pivot_table(index='end',columns='id',values='pred_cdf')

    
    x_test_2 = df_test_2.groupby('id').apply(lambda x: x.values[:,[0,1,2,3,4,5,7,8]])
    x_test_2 = tf.ragged.constant(x_test_2).to_tensor()
    dt_test_2 = tf.expand_dims(x_test_2[:,:,-1]-x_test_2[:,:,-2],-1)  
    x_test_2 = (x_test_2-x_mean_2)/x_std_2

    df_test_2['pred_cdf'] = np.exp(model_2.model.predict([x_test_2,dt_test_2]).flatten())
    df_test_2['pred_cdf'] = df_test_2['pred_cdf']*(df_test_2['end']-df_test_2['start'])
    df_test_2.loc[df_test_2.groupby('id').head(1).index, 'pred_cdf'] = -np.log(1-1/N_val) # manually set the first cumulative hazard value
    df_test_2['pred_cdf'] = df_test_2.groupby(['id'])['pred_cdf'].cumsum()
    df_test_2['pred_cdf'] = 1-np.exp(-df_test_2['pred_cdf'])
    df_test_2.groupby('id').last()['pred_cdf'] = 1 # enforce the last cdf to be 1
    pred_y_2_median = df_test_2[df_test_2['pred_cdf']<=0.5].groupby(['id']).last()['end']
    df_test_2['pred_pmf'] = df_test_2.groupby(['id'])['pred_cdf'].diff().fillna(df_test_2['pred_cdf'])
    df_test_2['pred_y'] = df_test_2['pred_pmf']*df_test_2['end']
    pred_y_2 = df_test_2.groupby(['id'])['pred_y'].sum()
    cdf_2 = df_test_2.pivot_table(index='end',columns='id',values='pred_cdf')


    for i in range(len(cdf_2)):
        try:
            cdf_2.iloc[i,:] = cdf_1.loc[cdf_1.index<=cdf_2.index[i],:].iloc[-1,:]
        except IndexError:
            cdf_2.iloc[i,:] = 0

    cdf = pd.concat([cdf_1,cdf_2])

    cdf_2 = df_test_2.pivot_table(index='end',columns='id',values='pred_cdf')

    for i in range(len(cdf_1)):
        try:
            cdf_1.iloc[i,:] = cdf_2.loc[cdf_2.index<=cdf_1.index[i],:].iloc[-1,:]
        except IndexError:
            cdf_1.iloc[i,:] = 0

    cdf = (pd.concat([cdf_1,cdf_2])+cdf)/2
    cdf = cdf.sort_index()
    pred_y = (pred_y_1+pred_y_2)/2
    pred_y_median = (pred_y_1_median+pred_y_2_median)/2

    plt.plot(cdf.index,cdf.loc[:,0])
    plt.vlines(pred_y[0],0,1,colors='orange',ls='--',label='predicted mean')
    plt.vlines(pred_y_median[0],0,1,colors='green',ls=':',label='predicted median')
    plt.vlines(test_y[0],0,1,colors='gray',label='observed value')
    plt.legend()
    plt.show()

    pred_y_all = pred_y_all + list(pred_y)
    test_y_all = test_y_all + list(test_y)

    mse = np.mean((pred_y - test_y)**2)
    median_se = np.median((pred_y - test_y)**2)
    r2 = r2_score(test_y,pred_y)
    mse_list.append(mse)
    median_se_list.append(median_se)
    r2_list.append(r2)
    print(mse,median_se,r2)


print(np.mean(mse_list),np.mean(median_se_list),np.mean(r2_list))