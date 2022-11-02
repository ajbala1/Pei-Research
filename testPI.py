#Predictability Improvement Model - Adapted
#Alex Bala
import math
from tkinter.dnd import dnd_start
import numpy as np
import scipy
from scipy import stats
# x = 1:0.1:10
# y = sin(x^2 + 1)
# y = rand(size(x))
def testPI(x,y,alpha): #x,y: List[???] alpha = float

    
    #Determine embedding dimension d and time delay tau for My
    def getDimensions(x,y):
        taus = [1] 
        ds = list(range(1,11)) #list of integers 1 to 10 inclusive
        T = len(y) #int = length of time sequence
        neis = list(range(1,11)) #list of integers 1 to 10 inclusive: neighbors
        e_min = float('inf') #current minimum

        
        for i in ds: 
            d = i
            for j in range(len(taus)):
                tau = taus[j]
                My = np.zeros((T - (d - 1) * tau - 1, d)) #matrix of zeros 
                targets_y = np.zeros(T - (d - 1) * tau - 1)
                for t in range(len(targets_y)):
                    k = y[t: (t + (d - 1) * tau) + 1: tau]
                    My[t] = y[t: (t + (d - 1) * tau) + 1: tau] #y[start:end:skip]. This should have length d
                    targets_y[t] = y[t + (d - 1) * tau] #last element in row of My[t]

                for k in range(len(neis)):
                    nei = neis[k] #number of neighbors in method of analogs
                    #predict y
                    ey = np.zeros(T - (d - 1) * tau - 1)
                    for t in range(len(ey)): #t 0 -> T-(d-1)*tau-2 inclusive
                        Y = My[t]
                        truth = targets_y[t] 
                        My_t = My
                        np.delete(My_t, t, axis=0) #CHECK AND MAKE SURE THESE THREE LINES ARE CORRECT
                        targets_t = targets_y
                        np.delete(targets_t, t, axis=0)
                        targets_t = np.reshape(targets_t, (-1, 1))
                        pred = analogs(Y, My_t, targets_t, nei) #Y = deleted row, My_t = My w/ delete, targets_t = targets_y with delete, nei = #of neighbors
                        ey[t] = abs(truth - pred)
                    ey_mean = np.mean(ey)
                    if ey_mean < e_min:
                        e_min = ey_mean
                        ey_best = ey
                        dy_best = d
                        neiy_best = nei
                        tauy_best = tau
        
        #Determine embedding dimension d and time delay tau for Mx (similar setup)
        taus = [1] 
        ds = list(range(1,11)) #list of integers 1 to 10 inclusive
        T = len(x) #int = length of time sequence
        neis = list(range(1,11)) #list of integers 1 to 10 inclusive
        e_min = float('inf') #current minimum
        for i in range(len(ds)): #0 -> 10
            d = ds[i]
            for j in range(len(taus)): 
                tau = taus[j]
                Mx = np.zeros((T - (d - 1) * tau - 1, d))
                targets_x = np.zeros(T - (d-1) * tau - 1)
                for t in range(len(targets_x)): #FIX THIS
                    Mx[t] = x[t: (t + (d - 1) * tau) + 1: tau] #x[start:end:skip]. This should have length d
                    targets_x[t] = x[t + (d - 1) * tau] #last element in row of Mx[t]
                for k in range(len(neis)):
                    nei = neis[k]
                    ex = np.zeros(T - (d-1) * tau -1)
                    for t in range(len(ex)):
                        Y = Mx[t, :]
                        truth = targets_x[t]
                        Mx_t = Mx
                        np.delete(Mx_t, t, axis=0)
                        targets_t = targets_x
                        np.delete(targets_t, t, axis=0)
                        targets_t = np.reshape(targets_t, (-1, 1))
                        pred = analogs(Y, Mx_t, targets_t, nei)
                        ex[t] = abs(truth - pred)
                ex_mean = np.mean(ex)
                if ex_mean < e_min:
                    e_min = ex_mean
                    dx_best = d
                    neix_best = nei
                    taux_best = tau

        return dy_best, dx_best, neiy_best, neix_best, tauy_best, taux_best, ey_best

    #call helper function to get relevant info
    d, dx_best, nei, neix_best, tau, taux_best, ey = getDimensions(x,y)

    T = len(y) #length of time series
    #tau = tauy_best
    #d = dy_best
    #nei = neiy_best
    #ey = ey_best (array)
    print(d)
    print(tau)
    print(dx_best)
    print(taux_best)
    print(T)

    
    My = np.zeros((T - (d-1) * tau - 1, d)) #zero array with embedding dimensions
    targets_y = np.zeros(T - (d - 1) * tau - 1)
    for t in range(len(targets_y)):
        My[t] = y[t: t + (d-1)* tau + 1: tau]
        targets_y[t] = y[t + (d-1) * tau + 1]
    '''
    ey = np.zeros(T - (d - 1) * tau - 1)
    for t in range(len(ey)):
        Y = My[t]
        truth = targets_y[t] 
        My_t = My
        np.delete(My_t, t, axis=0) #CHECK AND MAKE SURE THESE THREE LINES ARE CORRECT
        targets_t = targets_y
        np.delete(targets_t, t, axis=0)
        targets_t = np.reshape(targets_t, (-1, 1))
        pred = analogs(Y, My_t, targets_t, nei) #Y = deleted row, My_t = My w/ delete, targets_t = targets_y with delete, nei = #of neighbors
        ey[t] = abs(truth - pred)
    '''
    #Predict y with x
    #Set up
    Mxy = np.zeros((T - (d - 1) * tau - 1, d + dx_best))
    targets_y = np.zeros(T - (d - 1) * tau - 1)
    for t in range(len(targets_y)):
        tcnt = t + (d - 1) * tau #-1???
        k = (tcnt - (dx_best - 1) * taux_best)
        #print(k)
        #print(tcnt)
        txidx_hold = list(range(k, tcnt + 1, taux_best)) #start, stop, step
        #txidx_hold.reverse()
        txidx = []
        for x in txidx_hold:
            if x < 1: 
                txidx.append(int(1))
            else: 
                txidx.append(x)
        #print(txidx)
        temp = y[t: tcnt + 1: tau]
        Mxy[t] = y[t: t + (d-1)* tau + 1: tau] + txidx #FIX THIS
        targets_y[t] = y[t + (d-1) * tau + 1]

    print(Mxy)
    exy = np.zeros(T - (d - 1) * tau - 1)
    for t in range(len(ey)):
        Y = Mxy[t]
        truth = targets_y[t] 
        Mxy_t = Mxy
        np.delete(Mxy_t, t, axis=0) #CHECK AND MAKE SURE THESE THREE LINES ARE CORRECT
        targets_t = targets_y
        np.delete(targets_t, t, axis=0)
        targets_t = np.reshape(targets_t, (-1, 1))
        pred = analogs(Y, Mxy_t, targets_t, nei) #Y = deleted row, My_t = My w/ delete, targets_t = targets_y with delete, nei = #of neighbors
        exy[t] = abs(truth - pred)


    #test
    [t, p] = scipy.stats.ttest_ind(exy, ey, equal_var=True)
    if p < alpha:
        print("accept")
    else:
        print("reject")
    return True

def analogs(Y, My_t, targets_t, nei):
    #Y = deleted row
    #My_t = My without row Y
    #targets_t = targets_y with deleted element
    #nei = # of neighbors
    dist = np.subtract(My_t, np.ones((len(My_t), len(My_t[0]))) * Y)
    dist = np.power(np.sum(np.power(dist, 2), axis=1), 0.5) #maybe reshape here???
    dist = np.reshape(dist, (-1, 1))
    dist = np.append(dist, targets_t, 1)
    dist = dist[dist[:, 0].argsort()]
    weight = dist[0:nei, 0] #check if this indexing is correct???
    exp_sum = 0
    for i in weight:
        exp_sum += math.exp(-i)
    for i in range(len(weight)):
        weight[i] = math.exp(-weight[i]) / exp_sum
    weight = np.reshape(weight, (-1, 1))
    sample = np.reshape(dist[0:nei, 1], (-1, 1))
    pred = int(np.matmul(np.transpose(weight), sample)[0,0])
    #print(dist)
    return pred
    
if __name__ == '__main__':
    x = np.arange(1.0, 10, 0.1)
    #x = np.reshape(x, (-1, 1))
    #y = [math.sin(i**2 + 1) for i in x ]
    #y = [np.random.poisson(i) for i in x]
    y = [i for i in x]
    #print(x, y)
    j = testPI(x, y, 0.05)
    