from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.keras_nn import hazardNN
from models.baseline_nn import baselineNN
from models.simulator import SimulatedData
import tensorflow as tf



N = 2500                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
N_sim = 100

generator = SimulatedData()
min_min_y = -2.2
max_max_y = 15.2
##### testing procedure

test_N = 500
test_cdf_N = 10
test_data_origin, test_y = generator.generate_data(test_N,seed=0,test=True)
# cdf of 10 samples
#test_cdf_df = generator.calculate_cdf(test_data_origin.iloc[0:test_cdf_N,],min_min_y,max_max_y)


empirical_cdf_df = pd.DataFrame() # when theoritical cdf can't be obtained, use empirical
for j in range(1):
    test_cdf_df = generator.calculate_cdf_empirical(test_data_origin.iloc[0:test_cdf_N,],min_min_y,max_max_y)
    cdf = test_cdf_df['cdf'] 
    empirical_cdf_df = pd.concat([empirical_cdf_df,cdf],axis=1) 
test_cdf_df['cdf'] = empirical_cdf_df.mean(axis=1)



coverage_rate_90_list = []
coverage_rate_95_list = []
median_se_list =[]
baseline_median_se_list = []
mean_se_list =[]
baseline_mean_se_list = []

baseline_pred_cdf_df = pd.DataFrame()
pred_cdf_df = pd.DataFrame()

for i in range(N_sim):

    print("################## Simulation",i+1,"#####################")
    

    train_y = np.load('saved_models/model_train_y_{}.npy'.format(i))

    ### baseline model
    baseline_model = tf.keras.models.load_model('saved_models/l2_model_{}.h5'.format(i),compile=False)
    x_mean, x_std = np.load('saved_models/l2_model_standard_{}.npy'.format(i))
    x_test = test_data_origin.groupby('id').first()
    x_test = np.array(x_test)
    x_test = (x_test-x_mean)/x_std
    baseline_pred_y = baseline_model.predict(x_test).flatten()

    baseline_residual = baseline_pred_y-test_y
    test_cdf_df['res'] = test_cdf_df['end'] - baseline_pred_y[test_cdf_df['id']]
    test_cdf_df['pred_cdf_baseline'] = test_cdf_df['res'].apply(lambda x: sum(x>=baseline_residual)/test_N)
    baseline_pred_cdf_df = pd.concat([baseline_pred_cdf_df,test_cdf_df['pred_cdf_baseline']],axis=1)
  
    ### new model
    model = tf.keras.models.load_model('saved_models/model_{}.h5'.format(i),compile=False)

    ## cdf
    x_test = test_cdf_df.groupby('id').apply(lambda x: x.values[:,1:8])
    x_test = tf.ragged.constant(x_test).to_tensor()
    dt_test = tf.expand_dims(x_test[:,:,-1]-x_test[:,:,-2],-1)
    x_mean, x_std = np.load('saved_models/model_standard_{}.npy'.format(i))    
    x_test = (x_test-x_mean)/x_std
    
    hazard = np.exp(model.predict([x_test,dt_test]).flatten())
    test_cdf_df['pred_hazard'] = hazard
    test_cdf_df.loc[test_cdf_df['start']<min(train_y),'pred_hazard'] = 0  # do not make prediction beyond the left end
    test_cdf_df['pred_ch'] = test_cdf_df['pred_hazard']*(test_cdf_df['end']-test_cdf_df['start'])
    test_cdf_df.loc[test_cdf_df.groupby('id').head(1).index, 'pred_ch'] = -np.log(1-1/N) # manually add the cumulative hazard
    test_cdf_df['pred_ch'] = test_cdf_df.groupby(['id'])['pred_ch'].cumsum()
    test_cdf_df['pred_cdf'] = 1-np.exp(-test_cdf_df['pred_ch'])
    pred_cdf_df = pd.concat([pred_cdf_df,test_cdf_df['pred_cdf']],axis=1)
    
    ## mse 
    test_data = generator.expand(test_data_origin, train_y)
    x_test = test_data.groupby('id').apply(lambda x: x.values[:,1:8])
    x_test = tf.ragged.constant(x_test).to_tensor()
    dt_test = tf.expand_dims(x_test[:,:,-1]-x_test[:,:,-2],-1)
    x_test = (x_test-x_mean)/x_std
    hazard = np.exp(model.predict([x_test,dt_test]).flatten())
    test_data['pred_hazard'] = hazard
    test_data['pred_ch'] = test_data['pred_hazard']*(test_data['end']-test_data['start'])
    test_data.loc[test_data.groupby('id').head(1).index, 'pred_ch'] = -np.log(1-1/N) # manually set the first cumulative hazard value
    test_data['pred_ch'] = test_data.groupby(['id'])['pred_ch'].cumsum()
    test_data['pred_cdf'] = 1-np.exp(-test_data['pred_ch'])


    lower_bound_90 = test_data[(test_data['pred_cdf']<=0.95) & (test_data['pred_cdf']>=0.05)].groupby(['id']).first()['end']
    upper_bound_90 = test_data[(test_data['pred_cdf']<=0.95) & (test_data['pred_cdf']>=0.05)].groupby(['id']).last()['end']

    lower_bound_95 = test_data[(test_data['pred_cdf']<=0.975) & (test_data['pred_cdf']>=0.025)].groupby(['id']).first()['end']
    upper_bound_95 = test_data[(test_data['pred_cdf']<=0.975) & (test_data['pred_cdf']>=0.025)].groupby(['id']).last()['end']

    coverage_rate_90 = sum((test_y>=lower_bound_90) & (test_y<=upper_bound_90))/test_N
    coverage_rate_90_list.append(coverage_rate_90)
    coverage_rate_95 = sum((test_y>=lower_bound_95) & (test_y<=upper_bound_95))/test_N
    coverage_rate_95_list.append(coverage_rate_95)

    test_data['pred_pmf'] = test_data.groupby(['id'])['pred_cdf'].diff().fillna(test_data['pred_cdf'])
    test_data['pred_y'] = test_data['pred_pmf']*test_data['end']
    pred_y = test_data.groupby(['id'])['pred_y'].sum()

    median_se = median((pred_y-test_y)**2)
    baseline_median_se = median((baseline_pred_y-test_y)**2)
    mean_se = mean((pred_y-test_y)**2)
    baseline_mean_se = mean((baseline_pred_y-test_y)**2)

    median_se_list.append(median_se)
    baseline_median_se_list.append(baseline_median_se)
    mean_se_list.append(mean_se)
    baseline_mean_se_list.append(baseline_mean_se)

    


mean_coverage_rate_90 = np.mean(coverage_rate_90_list,0)
mean_coverage_rate_95 = np.mean(coverage_rate_95_list,0)

mean_median_se = np.mean(median_se_list,0)
mean_baseline_median_se = np.mean(baseline_median_se_list,0)
mean_mean_se = np.mean(mean_se_list,0)
mean_baseline_mean_se = np.mean(baseline_mean_se_list,0)

print("median square error:", mean_median_se, ", mean square error:", mean_mean_se, ", baseline median:", \
    mean_baseline_median_se, ", baseline mean:", mean_baseline_mean_se)

print("coverage rate:", mean_coverage_rate_90,mean_coverage_rate_95)

""""
residual = pred_y-test_y
baseline_residual = baseline_pred_y-test_y
max_residual = max(max(residual),max(baseline_residual))
min_residual = min(min(residual),min(baseline_residual))

plt.hist(baseline_residual,300,alpha=0.5,label='L2',range=(min_residual,max_residual))
plt.hist(residual,300,alpha=0.5,label='new')
plt.legend(loc='upper right')
plt.savefig("results/residuals.png")
plt.close()
"""

test_cdf_df['pred_cdf_mean'] = pred_cdf_df.mean(axis=1)
test_cdf_df['pred_cdf_0.1'] = pred_cdf_df.quantile(q=0.1,axis=1)
test_cdf_df['pred_cdf_0.9'] = pred_cdf_df.quantile(q=0.9,axis=1)

test_cdf_df['base_pred_cdf_mean'] = baseline_pred_cdf_df.mean(axis=1)
test_cdf_df['base_pred_cdf_0.1'] = baseline_pred_cdf_df.quantile(q=0.1,axis=1)
test_cdf_df['base_pred_cdf_0.9'] = baseline_pred_cdf_df.quantile(q=0.9,axis=1)



test_cdf_df = test_cdf_df[(test_cdf_df['cdf']<0.9999) & (test_cdf_df['cdf']>0.0001)]
np.save('test_cdf_df.npy',test_cdf_df)

for j in range(test_cdf_N):
    plt.plot(test_cdf_df.loc[test_cdf_df['id']==j,'end'],test_cdf_df.loc[test_cdf_df['id']==j,'cdf'],label="truth",color='black')
    plt.plot(test_cdf_df.loc[test_cdf_df['id']==j,'end'],test_cdf_df.loc[test_cdf_df['id']==j,'pred_cdf_mean'],'-.',label="new model",color='orange')
    plt.plot(test_cdf_df.loc[test_cdf_df['id']==j,'end'],test_cdf_df.loc[test_cdf_df['id']==j,'base_pred_cdf_mean'],'--',label="L2 model",color='green')
    plt.plot(test_cdf_df.loc[test_cdf_df['id']==j,'end'],test_cdf_df.loc[test_cdf_df['id']==j,'pred_cdf_0.1'],':',label="confidence band",color='orange')
    plt.plot(test_cdf_df.loc[test_cdf_df['id']==j,'end'],test_cdf_df.loc[test_cdf_df['id']==j,'pred_cdf_0.9'],':',color='orange')
    if j==0: plt.legend()
    plt.ylim(0,1)
    plt.savefig("results/figure_{}.png".format(j))
    plt.close()
