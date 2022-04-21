from lifelines import CoxTimeVaryingFitter
import pandas as pd
import numpy  as np

def cox_pred(data,test_data):
    ctv = CoxTimeVaryingFitter()
    ctv.fit(data, id_col="id", event_col="fail", start_col="start", stop_col="end", show_progress=True) # regard everything else as covariate
    beta = ctv.params_[0]
    base_ch = ctv.baseline_cumulative_hazard_/np.exp(np.mean(data['X1']+data['X2']+data['X3']+data['X4']+data['X5'])*beta)
    base_ch = base_ch.diff().fillna(base_ch) # from cumulative hazard to hazard
    test_data['pred_hazard'] = 0
    for t in base_ch.index:
        location = (test_data['start']<t)&(test_data['end']>=t)
        test_data.loc[location,'pred_hazard'] += \
            base_ch['baseline hazard'][t]*np.exp(beta*(test_data.loc[location,'X1']+
            test_data.loc[location,'X2']+test_data.loc[location,'X3']+
            test_data.loc[location,'X4']+test_data.loc[location,'X5']))
    test_data['pred_survival_cox'] = test_data.groupby(['id'])['pred_hazard'].cumsum()
    test_data['pred_survival_cox'] = np.exp(-test_data['pred_survival_cox'])
    del test_data['pred_hazard']
    return test_data, beta



