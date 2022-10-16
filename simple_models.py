import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import numpy as np
import json
import random
from math import sin, pi, log, exp, sqrt, floor, ceil, log1p, isnan
import matplotlib.pyplot as plt
import seaborn as sns

def simple_SIR(tm_strt, tm_end, step, S0, I0, R0, N, D, L, beta): #SIRS Model
#function to calculate next step in SIR model using Runge-Kutta
#inputs: tm_strt, tm_end: start and end times of model, respectively
#        tm_step: size of step
#        S0, I0: Initial S and I of population
#        N = pop size. D = infection period. L = immune period. (measured in days)
#output: arrays of S, I, and R over time interval

    def beta_t(t, eps = 0.5, psi = 365): #right now this is hardcoded based off of previous papers/emails. I can create a dynamic function later.
    #eps = amp of the seasonal adjustment, psi = period 
        return max(0 , beta + eps * sin(2.0 * pi / psi * (t - 3*psi/4)))

    
    i = 0
    times = [*range(tm_strt, tm_end + 1, step)] #inclusive of the endpoint
    times_size = len(times)
   

    S = [0 for i in range(times_size)]
    I = [0 for i in range(times_size)]
    dI = [0 for i in range(times_size)]
    R = [0 for i in range(times_size)]
    R[0] = R0
    S[0] = round(S0, 0)
    I[0] = round(I0, 0)
    

    for t in range(times_size - 1):
        #print(t)
        #Round 1
        Eimmloss = max(0, step * (1 / L  * (N- S[i] - I[i]))) #expected immunity loss
        Einf = max(0, step * (beta_t(t) * I[i] * S[i] / N)) #expected infections
        Erecov = max(0, step * (1 / D * I[i])) #expected recoveries
        smcl = np.random.poisson(Eimmloss)
        smci = np.random.poisson(Einf)
        smcr = np.random.poisson(Erecov)

        sk1 = smcl - smci #dS = newly sus. - infected
        ik1 = smci - smcr #dI = new infections - recovery
        ik1a = smci #new infections
        Tsk2 = S[i] + round(sk1 / 2, 0) #s-value for next step
        Tik2 = I[i] + round(ik1 /2, 0) #i-value for next step

        #Round 2
        Eimmloss = max(0, step * (1 / L  * (N- Tsk2 - Tik2)))
        Einf = max(0, step * (beta_t(t) * Tik2 * Tsk2 / N))
        Erecov = max(0, step * (1 / D * Tik2))
        smcl = np.random.poisson(Eimmloss)
        smci = np.random.poisson(Einf)
        smcr = np.random.poisson(Erecov)

        sk2 = smcl - smci #dS = newly sus. - infected
        ik2 = smci - smcr #dI = new infections - recovery
        ik2a = smci #new infections
        Tsk3 = S[i] + round(sk2 / 2, 0)
        Tik3 = I[i] + round(ik2 /2, 0)

        #Round 3
        Eimmloss = max(0, step * (1 / L  * (N- Tsk3 - Tik3)))
        Einf = max(0, step * (beta_t(t) * Tik3 * Tsk3 / N))
        Erecov = max(0, step * (1 / D * Tik3))
        smcl = np.random.poisson(Eimmloss)
        smci = np.random.poisson(Einf)
        smcr = np.random.poisson(Erecov)

        sk3 = smcl - smci #dS = newly sus. - infected
        ik3 = smci - smcr #dI = new infections - recovery
        ik3a = smci #new infections
        Tsk4 = S[i] + round(sk3, 0)
        Tik4 = I[i] + round(ik3, 0)
        
        #Round 4
        Eimmloss = max(0, step * (1 / L  * (N - Tsk4 - Tik4)))
        Einf = max(0, step * (beta_t(t) * Tik4 * Tsk4 / N))
        Erecov = max(0, step * (1 / D * Tik4))
        smcl = np.random.poisson(Eimmloss)
        smci = np.random.poisson(Einf)
        smcr = np.random.poisson(Erecov)

        sk4 = smcl - smci #dS = newly sus. - infected
        ik4 = smci - smcr #dI = new infections - recovery
        ik4a = smci #new infections

        #Final Calculations
        i += 1
        seed = np.random.poisson(0.1)
        S[i] = S[i - 1] + round(sk1/6+sk2/3+sk3/3+sk4/6,0) - seed
        I[i] = I[i-1] + round(ik1/6+ik2/3+ik3/3+ik4/6, 0) + seed
        dI[i] = round(ik1/6+ik2/3+ik3/3+ik4/6, 0) + seed
        R[i] = N - S[i] - I[i] 
        
    return S, I, dI, R

def simple_multistrain_RK_model(tm_strt, tm_end, tm_step, S0, I0, R0, N, mu, beta, nu, sigma): #SIR model
#S0/I0 = array of initial susceptible/infected
#N = size of population
#beta = array of beta values
#nu = array of nu values (recovery rate)
#sigma = matrix of sigma values. sigma[i][j] = cross immunity strain j drives strain i
    times = [*range(tm_strt, tm_end, tm_step)] #inclusive of the endpoint
    times_size = len(times)
    
    assert (len(S0) == len(I0)) and (len(S0) == len(beta)) and (len(S0) == len(nu)) #verify all dimensions are correct
    num_pathogens = len(S0)
    S = [S0]
    I = [I0]
    R = [R0]
    dI = [[0,0]]

    i = 0 #keeps track of iteration number
    for t in times: #use t in case beta becomes function of t later
        curr_S = S[i] #array of current S values
        curr_I = I[i] #array of current I values
        curr_R = R[i] #array of current R values
        curr_dI = dI[i]

        next_S, next_I, next_dI, next_R = RK_method(curr_S, curr_I, curr_dI, curr_R, N, beta, nu, mu, sigma, i, tm_step, t)

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
        dI.append(next_dI)

        i += 1

    return S, I, dI, R

def RK_method(S, I, dI, R, N, beta, nu, mu, sigma, curr, step, t):

#helper function to do RK method at given time 
#returns four tuples: dS, dI, dR, dCC
#NOTE: S/I/R/beta/nu all arrays, sigma matrix, mu/i/step
    num_pathogens = len(S) #instead of being passed from base function

    def beta_t(t, id, eps = 0.5, psi = 365): #right now this is hardcoded based off of previous papers/emails. I can create a dynamic function later.
    #eps = amp of the seasonal adjustment, psi = period
        return beta0[id] + (eps * sin(2.0 * pi / psi * (t - psi)))
    
    #first round
    sk1 = [] #sus
    ik1 = [] #infected
    rk1 = [] #recovered
    for i in range(num_pathogens):
        sig_sum = 0 #sigma summation in ODE model
        for j in range(num_pathogens):
            sig_sum +=  S[i] * sigma[i][j] * beta_t(t,j) * I[j] / N
        bsi = beta_t(t,i) * S[i] * I[i] / N
        nui = nu[i] * I[i] 
        sk1.append(step * (mu * N - sig_sum - mu * S[i]))
        ik1.append(step * (bsi - nui - mu * I[i]))
        rk1.append(step * (nui + sig_sum - bsi - mu * R[i]))

    Tsk1 = [0 for i in range(num_pathogens)]
    Tik1 = [0 for i in range(num_pathogens)]
    Trk1 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk1[j] = S[j] + round(sk1[j] / 2, 6) #s-value for next step
        Tik1[j] = I[j] + round(ik1[j] /2, 6) #i-value for next step
        Trk1[j] = R[j] + round(rk1[j] / 2, 6) #r-value for next step

    #second round
    sk2 = []
    ik2 = []
    rk2 = []
    for i in range(num_pathogens):
        sig_sum = 0
        for j in range(num_pathogens):
            sig_sum += Tsk1[i] * sigma[i][j] * beta_t(t,j) * Tik1[j] / N
        sig_sum = max(0, sig_sum)
        bsi = beta_t(t,i) * Tsk1[i] * Tik1[i] / N
        nui = nu[i] * Tik1[i]
        sk2.append(step * (mu * N - sig_sum - mu * Tsk1[i]))
        ik2.append(step * (bsi - nui - mu * Tik1[i]))
        rk2.append(step * (nui + sig_sum - bsi - mu * Trk1[i]))

    Tsk2 = [0 for i in range(num_pathogens)]
    Tik2 = [0 for i in range(num_pathogens)]
    Trk2 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk2[j] = S[j] + round(sk2[j] / 2, 6) #s-value for next step
        Tik2[j] = I[j] + round(ik2[j] /2, 6) #i-value for next step
        Trk2[j] = R[j] + round(rk2[j] / 2, 6) #r-value for next step

    #third round
    sk3 = []
    ik3 = []
    rk3 = []
    for i in range(num_pathogens):
        sig_sum = 0
        for j in range(num_pathogens):
            sig_sum += Tsk2[i] * sigma[i][j] * beta_t(t,j) * Tik2[j] / N
        sig_sum = max(0, sig_sum)
        bsi = beta_t(t,i) * Tsk2[i] * Tik2[i] / N
        nui = nu[i] * Tik2[i]
        sk3.append(step * (mu * N - sig_sum - mu * Tsk2[i]))
        ik3.append(step * (bsi - nui - mu * Tik2[i]))
        rk3.append(step * (nui + sig_sum - bsi - mu * Trk2[i]))
        

    Tsk3 = [0 for i in range(num_pathogens)]
    Tik3 = [0 for i in range(num_pathogens)]
    Trk3 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk3[j] = S[j] + round(sk3[j], 6) #s-value for next step
        Tik3[j] = I[j] + round(ik3[j], 6) #i-value for next step
        Trk3[j] = R[j] + round(rk3[j], 6) #r-value for next step
    
    #fourth round
    sk4 = []
    ik4 = []
    rk4 = []
    for i in range(num_pathogens):
        sig_sum = 0
        for j in range(num_pathogens):
            sig_sum += Tsk3[i] * sigma[i][j] * beta_t(t,j) * Tik3[j] / N
        sig_sum = max(0, sig_sum)
        bsi = beta_t(t,i) * Tsk3[i] * Tik3[i] / N
        nui = nu[i] * Tik3[i]
        sk4.append(step * (mu * N - sig_sum - mu * Tsk3[i]))
        ik4.append(step * (bsi - nui - mu * Tik3[i]))
        rk4.append(step * (nui + sig_sum - bsi - mu * Trk3[i]))

    #print("round 4", sk1, ik1, rk1, Tsk1, Tik1, Trk1)

    #final calculations
    next_S = []
    next_I = []
    next_R = []
    next_dI = []
    for i in range(num_pathogens):
        '''next_S.append(np.random.poisson(round(S[i] + sk1[i]/6+sk2[i]/3+sk3[i]/3+sk4[i]/6, 0)))
        next_I.append(np.random.poisson(round(I[i] + ik1[i]/6+ik2[i]/3+ik3[i]/3+ik4[i]/6, 0)))
        next_R.append(np.random.poisson(round(R[i] + rk1[i]/6+rk2[i]/3+rk3[i]/3+rk4[i]/6, 0)))'''
        next_S.append(round(S[i] + sk1[i] / 6+ sk2[i] / 3 + sk3[i] / 3 + sk4[i] / 6, 0))
        next_I.append(round(I[i] + ik1[i] / 6 + ik2[i] / 3 + ik3[i] / 3 + ik4[i] / 6, 0))
        next_R.append(round(R[i] + rk1[i] / 6 + rk2[i] / 3 + rk3[i] / 3 + rk4[i] / 6, 0))
        next_dI.append(round(ik1[i] / 6 + ik2[i] / 3 + ik3[i] / 3 + ik4[i] / 6, 0))

    return next_S, next_I, next_dI, next_R

def Plot_SIR(S, I, R, beta = "n/a", dI = None, plot_I = True, pathogens = 1):
    sns.set_style("darkgrid")
    if pathogens == 1: 
        plt.plot(range(len(S)), S, label = "S")
        if plot_I:
            plt.plot(range(len(I)), I, label = "I")
        plt.plot(range(len(R)), R, label = "R")
        if dI:
            plt.plot(range(len(dI)), dI, label = "dI")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        title = "SIR Model Test: Beta = " + str(beta)
        subtitle = "S0 = " + str(S[0]) + " I0 = " + str(I[0]) + " N = " + str(S[0] + I[0] + R[0])
        plt.suptitle(subtitle, fontsize = 'small')
        plt.title(title)
    elif pathogens != 1:
        for j in range(len(S[0])):
            Si = []
            Ii = []
            Ri = []
            dIi = []
            for i in range(len(S)):
                Si.append(S[i][j])
                Ii.append(I[i][j])
                Ri.append(R[i][j])
                dIi.append(dI[i][j])
            plt.subplot(1,2, j + 1)
            plt.plot(range(len(Ii)), Ii, label = "I")
            plt.plot(range(len(Si)), Si, label = "S")
            plt.plot(range(len(Ri)), Ri, label = "R")
            plt.plot(range(len(dIi)), dIi, label = "dI")

        plt.subplot(1,2,1)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        plt.title('SIRS Model')

    plt.show()


def beta_t(eps = 0.5, psi = 365): #right now this is hardcoded based off of previous papers/emails. I can create a dynamic function later.
    #eps = amp of the seasonal adjustment, psi = period 
    betas = []
    for t in range(365):
        betas.append(max(0, 0.4 + eps * sin(2.0 * pi / psi * (t -  3 * psi / 4))))
    return betas

if __name__ == '__main__':

    
    #------TEST simple_SIR------
    #simple_SIR(tm_strt, tm_end, step, S0, I0, R0, N, D, L, beta):
    
    #Testing different betas
    S, I, dI, R = simple_SIR(0, 365, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.4)
    Plot_SIR(S,I,R, 0.4, dI)

    plt.plot(range(len(dI)), dI)
    #plt.show()

    betas = beta_t()
    #plt.plot(range(len(betas)), betas)
    #plt.show()
    
    S, I, dI, R = simple_SIR(0, 1080, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.4)
    Plot_SIR(S,I,R, 0.4, dI)
    
    #Plotting with Multiple Betas
    S, I, dI1, R = simple_SIR(0, 365, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.4)
    S, I, dI2, R = simple_SIR(0, 365, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.7)
    S, I, dI3, R = simple_SIR(0, 365, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.1)
    sns.set_style("darkgrid")
    plt.plot(range(len(dI1)), dI1, label = "dI1")
    plt.plot(range(len(dI2)), dI2, label = "dI2")
    plt.plot(range(len(dI3)), dI3, label = "dI3")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.show()
    
    #-------TEST simple_multistrain_RK_model------
    
    #one-way cross-immunity
    S=[1000000, 1000000]
    I= [1000, 1000]
    R = [0, 0]
    beta = [1.1, 0.8]
    mu = 1/ 30.0  #as proportion of pop
    nu = [0.26, 0.26]
    sigma=[[1, 0.4], [0, 1]]
    beta0 = [0.6, 0.8]

    #def simple_SIR(tm_strt, tm_end, step, S0, I0, N, D, L, beta, real_data = False, discrete = True): 
    S,I,dI,R = simple_multistrain_RK_model(0, 365, 1, S, I, R, S[0] + R[0] + I[0], mu, beta, nu, sigma)
    Plot_SIR(S,I,R, dI = dI, pathogens = 2)

    #two-way immunity
    S=[1000000, 1000000]
    I= [1000, 1000]
    R = [0, 0]
    beta = [1.1, 0.8]
    mu = 1/ 30.0  #as proportion of pop
    nu = [0.26, 0.26]
    sigma=[[1, 0.4], [0.6, 1]]
    beta0 = [0.6, 0.8]

    #def simple_SIR(tm_strt, tm_end, step, S0, I0, N, D, L, beta, real_data = False, discrete = True): 
    S,I,dI,R = simple_multistrain_RK_model(0, 365, 1, S, I, R, S[0] + R[0] + I[0], mu, beta, nu, sigma)
    Plot_SIR(S,I,R, dI = dI, pathogens = 2)
    
    