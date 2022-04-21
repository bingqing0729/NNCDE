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

num_nodes_list = [64,64,64,64,64]
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

    test_y = np.array(df_test.loc[:,6])
    x_test = np.array(df_test.loc[:,[0,1,2,3,4,5]])

    min_loss = np.inf
    # grid search best model
    for nn_config in nn_config_list:

        print(nn_config)
        
        # model 1
        model_1_tmp = baselineNN(nn_config)
        x_mean_1_tmp, x_std_1_tmp, loss_1 = model_1_tmp.train(
            x_train = np.array(df_train.loc[:,[0,1,2,3,4,5]]),
            y_train = np.array(df_train.loc[:,6]),
            x_valid = np.array(df_val.loc[:,[0,1,2,3,4,5]]),
            y_valid = np.array(df_val.loc[:,6])
            )

        # model 2
        model_2_tmp = baselineNN(nn_config)
        x_mean_2_tmp, x_std_2_tmp, loss_2 = model_2_tmp.train(
            x_train = np.array(df_val.loc[:,[0,1,2,3,4,5]]),
            y_train = np.array(df_val.loc[:,6]),
            x_valid = np.array(df_train.loc[:,[0,1,2,3,4,5]]),
            y_valid = np.array(df_train.loc[:,6])
            )

        loss = (loss_1+loss_2)/2
        if loss<min_loss:
            min_loss = loss
            model_1, model_2 = model_1_tmp, model_2_tmp
            x_mean_1, x_std_1 = x_mean_1_tmp, x_std_1_tmp
            x_mean_2, x_std_2 = x_mean_2_tmp, x_std_2_tmp

    x_test_1 = (x_test-x_mean_1)/x_std_1
    pred_y_1  = model_1.model.predict(x_test_1).flatten()
    x_test_2 = (x_test-x_mean_2)/x_std_2
    pred_y_2  = model_2.model.predict(x_test_2).flatten()

    pred_y = (pred_y_1+pred_y_2)/2

    print(pred_y)
    print(test_y)

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
plt.plot(pred_y_all,test_y_all,'o')
plt.xlabel("predicted mean")
plt.ylabel("true response")
