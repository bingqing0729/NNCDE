from math import *
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import norm, chi2, beta
import scipy.stats


class SimulatedSurvival:

    def __init__(self,T,step,cox,baseline=None,k=10):
        self.T = T #100
        self.step = step #0.01
        self.cox = cox
        if baseline is None:
            #lt = [0]+sorted(random.sample(range(2,self.T),k=k))+[self.T]
            #lh = [0]+sorted([random.random() for i in range(k)])+[1]
            #spline = scipy.interpolate.PchipInterpolator(lt,lh)
            t = np.arange(0,self.T,self.step)
            #cdf = spline(t)
            #pdf = np.append(np.diff(cdf),1-cdf[-1])/self.step
            cdf = beta.cdf(t/self.T,8,1)
            pdf = beta.pdf(t/self.T,8,1)
            self.baseline = pdf/(1-cdf)
            np.savetxt("baseline.csv", self.baseline)
        else:
            self.baseline = baseline
            

    def generate_data(self, N, seed=0, test=False):

        np.random.seed(seed)
        t = np.arange(0,self.T,self.step) #[0,100,0.01]
        alpha_list = []
        q_list = []
        x3_list = []
        x4_list = []
        x5_list = []
        Y_list = [] #observation time
        delta_list = []

        for i in range(N):
            alpha = np.random.uniform(0,1,5)
            alpha_list.append(alpha)
            q = np.random.uniform(0,self.T,1)
            q_list.append(q)
            X1 = alpha[0]+alpha[1]*np.sin(2*pi*t/self.T)+alpha[2]*np.cos(2*pi*t/self.T)+alpha[3]*np.sin(4*pi*t/self.T)+alpha[4]*np.cos(4*pi*t/self.T)
            X2 = (t>q)*1
            X3 = np.random.binomial(1,0.6)
            while 1:
                X4 = np.random.poisson(lam=2)
                if X4 < 5: break
            X5 = np.random.beta(2,5)
            x3_list.append(X3)
            x4_list.append(X4)
            x5_list.append(X5)
            if self.cox:
                h = self.baseline*np.exp(2*(X1+X2+X3+X4+X5)) # cox assumption
            else:
                h = self.baseline*np.exp(2*(X1**2+X2+X3*X4+X5)) # non-cox setting
                #h = self.baseline*(X1**2+X2+X3*X4+X5) # non-cox setting
            ch = np.cumsum(h)*self.step
            S = np.exp(-ch)
            u = np.random.uniform(0,1,1)[0]
            index = np.sum(s>u for s in S) #S[index-1]>u, S[index]<=u
            if index==len(t):
                fT = t[-1]
            elif index==0:
                fT = 0.0001
            else:
                fT = t[index-1]+self.step*(S[index-1]-u)/(S[index-1]-S[index])
            #C = min(np.random.exponential(50,1)[0],self.T-1)
            C = np.random.exponential(100,1)[0]
            Y_list.append(min(fT,C)) #observation time
            delta_list.append((fT<C)*1)

        print('Failue rate:', sum(delta_list)/len(delta_list))

        sorted_t = sorted(Y_list)
        sorted_t = list(OrderedDict.fromkeys(sorted_t))

        id_list = []
        fail_list = []
        end_list = []
        start_list = []
        X1_list = []
        X2_list = []
        X3_list = []
        X4_list = []
        X5_list = []
        cum_hazard_list = []

        for i in range(N):
            Y = Y_list[i]
            alpha = alpha_list[i]
            q = q_list[i]
            X3 = x3_list[i]
            X4 = x4_list[i]
            X5 = x5_list[i]
            cum_hazard = 0
            start = 0
            if test:
                for time in t[1:]:
                    X1 = alpha[0]+alpha[1]*np.sin(2*pi*start/self.T)+alpha[2]*np.cos(2*pi*start/self.T)+alpha[3]*np.sin(4*pi*start/self.T)+alpha[4]*np.cos(4*pi*start/self.T)
                    X2 = ((start>q)*1)[0]
                    if self.cox:
                        cum_hazard += self.step*self.baseline[floor(start/self.step)]*np.exp(2*(X1+X2+X3+X4+X5)) # cox assumption
                    else:
                        cum_hazard += self.step*self.baseline[floor(start/self.step)]*np.exp(2*(X1**2+X2+X3*X4+X5)) # non-cox
                        #cum_hazard += self.step*self.baseline[floor(start/self.step)]*(X1**2+X2+X3*X4+X5) # non-cox
                    id_list.append(i)
                    X1_list.append(X1)
                    X2_list.append(X2)
                    X3_list.append(X3)
                    X4_list.append(X4)
                    X5_list.append(X5)
                    cum_hazard_list.append(cum_hazard)
                    end_list.append(time)
                    start_list.append(start)
                    start = time
            else:
                for time in [sorted_t[i] for i in range(len(sorted_t)) if sorted_t[i]<=Y]:
                    X1 = alpha[0]+alpha[1]*np.sin(2*pi*start/self.T)+alpha[2]*np.cos(2*pi*start/self.T)+alpha[3]*np.sin(4*pi*start/self.T)+alpha[4]*np.cos(4*pi*start/self.T)
                    X2 = ((start>q)*1)[0]
                    id_list.append(i)
                    X1_list.append(X1)
                    X2_list.append(X2)
                    X3_list.append(X3)
                    X4_list.append(X4)
                    X5_list.append(X5)
                    end_list.append(time)
                    start_list.append(start)
                    if time < Y:
                        fail_list.append(0)
                    elif time == Y:
                        fail_list.append(delta_list[i])
                    start = time
        
        if test:
            df = pd.DataFrame({'id':id_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,'start':start_list,'end':end_list,'cum_hazard':cum_hazard_list})
            df['survival'] = np.exp(-df['cum_hazard'])

        else:
            df = pd.DataFrame({'id':id_list,'fail':fail_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,'start':start_list,'end':end_list})

        return df, sorted_t[-1]


class SimulatedData:

    # simulate variable Y conditioning on covariate X. 

    def __init__(self):
        pass
            
    def generate_data(self, N, seed=0, test=False):
        
        np.random.seed(seed)
        X_list = []
        Y_list = []
        signal_list  = []
        e_list = []
        for i in range(N):
            while 1: # truncate X1
                X1 = np.random.normal(0,1)
                if abs(X1)<3: break
            X2= np.random.uniform(0,1)
            X3 = np.random.beta(0.5,0.5)
            X4 = np.random.binomial(1,0.5)
            while 1:
                X5 = np.random.poisson(lam=2)
                if X5 < 5: break
            w = np.random.multinomial(1,[0.1,0.7,0.2])
            while 1: # truncate error
                #X6 = np.random.normal(1+0.5*X1,3/4) # cov = [[1,0.5],[0.5,1]]
                X6 = np.random.normal(1,1)
                a = w[0]*np.random.normal(-2,1) + w[1]*np.random.normal(0,1) + w[2]*(0.5*X6**2)
                #a = w[0]*scipy.stats.t.rvs(df=2,loc=-2) + w[1]*scipy.stats.t.rvs(df=2,loc=0) + w[2]*(0.5*X6**2)
                if abs(a)<np.inf: break

            #gX = 0.5*X1**2 # a positive function
            gX = 0.5
            e = gX*a
                
            
            signal = X1**2+X2*X3+X3*X4+X5
  
            Y = signal+e

            X_list.append([X1,X2,X3,X4,X5])
            Y_list.append(Y)
            signal_list.append(signal)
            e_list.append(e)


        sorted_t = sorted(Y_list)
        sorted_t = list(OrderedDict.fromkeys(sorted_t))

        id_list = []
        fail_list = []
        end_list = []
        start_list = []
        X1_list = []
        X2_list = []
        X3_list = []
        X4_list = []
        X5_list = []

        if test:
            df = pd.DataFrame({'id':range(N),'X1':[a[0] for a in X_list],'X2':[a[1] for a in X_list],
            'X3':[a[2] for a in X_list],'X4':[a[3] for a in X_list],'X5':[a[4] for a in X_list]})


        else:

            for i in range(N):
                Y = Y_list[i]
                X1, X2, X3, X4, X5 = X_list[i][0], X_list[i][1], X_list[i][2], X_list[i][3], X_list[i][4]
                start = -np.inf # The first time point t0
                for time in [sorted_t[i] for i in range(len(sorted_t)) if sorted_t[i]<=Y]:
                    id_list.append(i)
                    X1_list.append(X1)
                    X2_list.append(X2)
                    X3_list.append(X3)
                    X4_list.append(X4)
                    X5_list.append(X5)
                    end_list.append(time)
                    start_list.append(start)
                    if time < Y:
                        fail_list.append(0)
                    elif time == Y:
                        fail_list.append(1)
                    start = time

            df = pd.DataFrame({'id':id_list,'fail':fail_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,
            'start':start_list,'end':end_list})
        
        return df, Y_list

    def expand(self, test_data, train_y):

        # expand the dataframe, repeat the covariate for each time point.

        sorted_train = sorted(train_y)
        sorted_train = list(OrderedDict.fromkeys(sorted_train))
        N = len(test_data)
        id_list = []
        X1_list = []
        X2_list = []
        X3_list = []
        X4_list = []
        X5_list = []
        end_list = []
        start_list = []

        for i in range(N):
            X1, X2, X3, X4, X5 = test_data.iloc[i,1], test_data.iloc[i,2], test_data.iloc[i,3], test_data.iloc[i,4],test_data.iloc[i,5]
            start = -np.inf
            for time in sorted_train:
                id_list.append(i)
                X1_list.append(X1)
                X2_list.append(X2)
                X3_list.append(X3)
                X4_list.append(X4)
                X5_list.append(X5)
                end_list.append(time)
                start_list.append(start)
                start = time

        df = pd.DataFrame({'id':id_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,'start':start_list,'end':end_list})
        return df

    def calculate_cdf(self, test_data, min_y, max_y,step=0.1):
        # calculate the true cdf for test data, according to the settings in generate_data.
        w = [0.1,0.7,0.2]
        t = np.arange(min_y,max_y,step)
        
        N = test_data.shape[0]
        
        id_list = []
        X1_list = []
        X2_list = []
        X3_list = []
        X4_list = []
        X5_list = []
        end_list = []
        start_list = []
        cdf_list = []

        for i in range(N):

            X1, X2, X3, X4, X5 = test_data.iloc[i,1], test_data.iloc[i,2], test_data.iloc[i,3],test_data.iloc[i,4],test_data.iloc[i,5]
            start = t[0]

            for time in t[1:]:
                x = (time-X1**2-X2*X3-X3*X4-X5)/X1**2/0.5 # a 
                #x = (time-X1**2-X2*X3-X3*X4-X5)
                cdf = w[0]*norm.cdf(x,-2,1)+ w[1]*norm.cdf(x,0,1) + w[2]*chi2.cdf(2*x,1,1+0.5*X1,3/4)
                #cdf = w[0]*norm.cdf(x,-2,1)+ w[1]*norm.cdf(x,0,1) + w[2]*chi2.cdf(2*x,1,1,1)
                id_list.append(i)
                X1_list.append(X1)
                X2_list.append(X2)
                X3_list.append(X3)
                X4_list.append(X4)
                X5_list.append(X5)
                end_list.append(time)
                start_list.append(start)
                cdf_list.append(cdf)
                start = time

        df = pd.DataFrame({'id':id_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,'start':start_list,'end':end_list,'cdf':cdf_list})
        return df

    def calculate_cdf_empirical(self, test_data, min_y, max_y,step=0.1):
        # calculate the empirical cdf for test data, according to the settings in generate_data.
        t = np.arange(min_y,max_y,step)
        
        N = test_data.shape[0]
        
        id_list = []
        X1_list = []
        X2_list = []
        X3_list = []
        X4_list = []
        X5_list = []
        end_list = []
        start_list = []
        cdf_list = []
        n = 2000

        for i in range(N):
            j = 0
            a_list = []
            X1, X2, X3, X4, X5 = test_data.iloc[i,1], test_data.iloc[i,2], test_data.iloc[i,3],test_data.iloc[i,4],test_data.iloc[i,5]
            while j < n:
                #X6 = np.random.normal(1+0.5*X1,3/4) # cov = [[1,0.5],[0.5,1]] 
                X6 = np.random.normal(1,1)
                w = np.random.multinomial(1,[0.1,0.7,0.2])
                a = w[0]*np.random.normal(-2,1) + w[1]*np.random.normal(0,1) + w[2]*(0.5*X6**2)
                #a = w[0]*scipy.stats.t.rvs(df=2,loc=-2) + w[1]*scipy.stats.t.rvs(df=2,loc=0) + w[2]*(0.5*X6**2)
                if abs(a) < np.inf:
                    a_list.append(a)
                    j += 1

            start = t[0]

            for time in t[1:]:
                #x = (time-X1**2-X2*X3-X3*X4-X5)/X1**2/0.5 # a
                x = (time-X1**2-X2*X3-X3*X4-X5)/0.5
                cdf = sum(a_list<x)/n
                id_list.append(i) 
                X1_list.append(X1)
                X2_list.append(X2)
                X3_list.append(X3)
                X4_list.append(X4)
                X5_list.append(X5)
                end_list.append(time)
                start_list.append(start)
                cdf_list.append(cdf)
                start = time

        df = pd.DataFrame({'id':id_list,'X1':X1_list,'X2':X2_list,'X3':X3_list,'X4':X4_list,'X5':X5_list,'start':start_list,'end':end_list,'cdf':cdf_list})
        return df

