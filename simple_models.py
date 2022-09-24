import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import numpy as np
import json
import random
from math import sin, pi, log, exp, sqrt, floor, ceil, log1p, isnan
import matplotlib.pyplot as plt

def simple_SIR(tm_strt, tm_end, step, S0, I0, R0, N, D, L, beta): #SIRS Model
#function to calculate next step in SIR model using Runge-Kutta
#inputs: tm_strt, tm_end: start and end times of model, respectively
#        tm_step: size of step
#        S0, I0: Initial S and I of population
#        N = pop size. D = infection period. L = immune period. (measured in days)
#output: arrays of S, I, and R over time interval

    def beta_t(t, eps = 0.5, psi = 365): #right now this is hardcoded based off of previous papers/emails. I can create a dynamic function later.
    #eps = amp of the seasonal adjustment, psi = period 
        return max(0, beta + (1.0 + eps * sin(2.0 * pi / psi * (t - psi))))

    i = 0
    times = [*range(tm_strt, tm_end + 1, step)] #inclusive of the endpoint
    times_size = len(times)
   

    S = [0 for i in range(times_size)]
    I = [0 for i in range(times_size)]
    R = [0 for i in range(times_size)]
    S[0] = S0
    I[0] = I0
    R[0] = R0


    
    S[0] = round(S[0], 0)
    I[0] = round(I[0], 0)

    for t in range(times_size - 1):
        #print(t)
        #Round 1
        Eimmloss = max(0, step * (1 / L  * (N- S[i] - I[i])))
        Einf = max(0, step * (beta_t(t) * I[i] * S[i] / N))
        Erecov = max(0, step * (1 / D * I[i]))
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
        I[i] = I[i-1]+round(ik1/6+ik2/3+ik3/3+ik4/6, 0) + seed
        R[i] = N - S[i] - I[i] 
        


    return S, I, R

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
    CC = [[0 for i in range(num_pathogens)]] #cummulative cases = S->I; initially assume all 0

    i = 0 #keeps track of iteration number
    for t in times: #use t in case beta becomes function of t later
        curr_S = S[i] #array of current S values
        curr_I = I[i] #array of current I values
        curr_R = R[i] #array of current R values
        curr_CC = CC[i]

        next_S, next_I, next_R, next_CC = RK_method(curr_S, curr_I, curr_R, curr_CC, N, beta, nu, mu, sigma, i, tm_step, t)

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
        CC.append(next_CC)

        i += 1

    return S, I, R, CC

def RK_method(S, I, R, CC, N, beta, nu, mu, sigma, curr, step, t):
#helper function to do RK method at given time 
#returns four tuples: dS, dI, dR, dCC
#NOTE: S/I/R/beta/nu all arrays, sigma matrix, mu/i/step
    num_pathogens = len(S) #instead of being passed from base function

    def beta_t(t, id, eps = 0.5, psi = 365): #right now this is hardcoded based off of previous papers/emails. I can create a dynamic function later.
    #eps = amp of the seasonal adjustment, psi = period
        return beta0[id] + (1.0 + eps * sin(2.0 * pi / psi * (t - psi)))
    
    #first round
    sk1 = [] 
    ik1 = []
    rk1 = []
    cck1 = []
    for i in range(num_pathogens):
        sig_sum = 0
        for j in range(num_pathogens):
            sig_sum +=  S[i] * sigma[i][j] * beta_t(t,j) * I[j] / N
        bsi = beta_t(t,i) * S[i] * I[i] / N
        nui = nu[i] * I[i] 
        sk1.append(step * (mu * N - sig_sum - mu * S[i]))
        ik1.append(step * (bsi - nui - mu * I[i]))
        rk1.append(step * (nui + sig_sum - bsi - mu * R[i]))
        cck1.append(step * bsi)

    Tsk1 = [0 for i in range(num_pathogens)]
    Tik1 = [0 for i in range(num_pathogens)]
    Trk1 = [0 for i in range(num_pathogens)]
    Tcck1 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk1[j] = S[j] + round(sk1[j] / 2, 6) #s-value for next step
        Tik1[j] = I[j] + round(ik1[j] /2, 6) #i-value for next step
        Trk1[j] = R[j] + round(rk1[j] / 2, 6) #r-value for next step
        Tcck1[j] = CC[j] + round(cck1[j] / 2, 6) #cummulative cases-value for next step


    #second round
    sk2 = []
    ik2 = []
    rk2 = []
    cck2 = []
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
        cck2.append(step * bsi)

    Tsk2 = [0 for i in range(num_pathogens)]
    Tik2 = [0 for i in range(num_pathogens)]
    Trk2 = [0 for i in range(num_pathogens)]
    Tcck2 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk2[j] = S[j] + round(sk2[j] / 2, 6) #s-value for next step
        Tik2[j] = I[j] + round(ik2[j] /2, 6) #i-value for next step
        Trk2[j] = R[j] + round(rk2[j] / 2, 6) #r-value for next step
        Tcck2[j] = CC[j] + round(cck2[j] / 2, 6)

    
    #third round
    sk3 = []
    ik3 = []
    rk3 = []
    cck3 = []
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
        cck3.append(step * bsi)

    Tsk3 = [0 for i in range(num_pathogens)]
    Tik3 = [0 for i in range(num_pathogens)]
    Trk3 = [0 for i in range(num_pathogens)]
    Tcck3 = [0 for i in range(num_pathogens)]

    for j in range(num_pathogens):
        Tsk3[j] = S[j] + round(sk3[j], 6) #s-value for next step
        Tik3[j] = I[j] + round(ik3[j], 6) #i-value for next step
        Trk3[j] = R[j] + round(rk3[j], 6) #r-value for next step
        Tcck3[j] = CC[j] + round(cck3[j], 6)
    

    #fourth round
    sk4 = []
    ik4 = []
    rk4 = []
    cck4 = []
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
        cck4.append(step * bsi)

    #print("round 4", sk1, ik1, rk1, Tsk1, Tik1, Trk1)

    #final calculations
    next_S = []
    next_I = []
    next_R = []
    next_CC = []
    for i in range(num_pathogens):
        '''next_S.append(np.random.poisson(round(S[i] + sk1[i]/6+sk2[i]/3+sk3[i]/3+sk4[i]/6, 0)))
        next_CC.append(np.random.poisson(round(CC[i] + cck1[i]/6+cck2[i]/3+cck3[i]/3+cck4[i]/6, 0)))
        next_I.append(np.random.poisson(round(I[i] + ik1[i]/6+ik2[i]/3+ik3[i]/3+ik4[i]/6, 0)))
        next_R.append(np.random.poisson(round(R[i] + rk1[i]/6+rk2[i]/3+rk3[i]/3+rk4[i]/6, 0)))'''
        next_S.append(round(S[i] + sk1[i] / 6+ sk2[i] / 3 + sk3[i] / 3 + sk4[i] / 6, 0))
        next_CC.append(round(CC[i] + cck1[i] / 6 + cck2[i] / 3 + cck3[i] / 3 + cck4[i] / 6, 0))
        next_I.append(round(I[i] + ik1[i] / 6 + ik2[i] / 3 + ik3[i] / 3 + ik4[i] / 6, 0))
        next_R.append(round(R[i] + rk1[i] / 6 + rk2[i] / 3 + rk3[i] / 3 + rk4[i] / 6, 0))
        
    return next_S, next_I, next_R, next_CC

if __name__ == '__main__':

    
    #------TEST simple_SIR------
    #simple_SIR(tm_strt, tm_end, step, S0, I0, R0, N, D, L, beta):
    
    #Testing different betas
    S, I, R = simple_SIR(0, 50, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.4)
    print(I)
    plt.figure(1)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 1: Beta = 0.4')
    plt.show()


    S, I, R = simple_SIR(0, 50, 1, 1000000, 1000, 0, 1001000, 5, 90, 1)
    plt.figure(2)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 2: Beta = 1')
    plt.show()

    S, I , R = simple_SIR(0, 50, 1, 1000000, 1000, 0, 1001000, 5, 90, 1.6)
    plt.figure(3)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 3: Beta = 1.6')
    plt.show()

    S, I, R = simple_SIR(180, 250, 1, 1000000, 1000, 0, 1001000, 5, 90, 0.4)
    plt.figure(4)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 4: Beta = 0.4')
    plt.show()


    S, I, R = simple_SIR(180, 250, 1, 1000000, 1000, 0, 1001000, 5, 90, 1)
    plt.figure(5)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 5: Beta = 1')
    plt.show()

    S, I , R = simple_SIR(180, 250, 1, 1000000, 1000, 0, 1001000, 5, 90, 1.6)
    plt.figure(6)
    plt.plot(range(len(S)), S, label = "S")
    plt.plot(range(len(I)), I, label = "I")
    plt.plot(range(len(R)), R, label = "R")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('SIR Model Test 6: Beta = 1.6')
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
    S,I,R,CC = simple_multistrain_RK_model(0, 35, 1, S, I, R, S[0] + R[0] + I[0], mu, beta, nu, sigma)
    
    for j in range(len(S[0])):
        Si = []
        Ii = []
        Ri = []
        CCi = []
        for i in range(len(S)):
            Si.append(S[i][j])
            Ii.append(I[i][j])
            Ri.append(R[i][j])
            CCi.append(CC[i][j])
        plt.figure(j+1)
        plt.plot(range(len(Ii)), Ii, label = "I")
        plt.plot(range(len(Si)), Si, label = "S")
        plt.plot(range(len(Ri)), Ri, label = "R")
        plt.plot(range(len(CCi)), CCi, label = "Cummulative Cases")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        id = str(j + 1)
        plt.title('SIR Model (one-way cross-immunity): Pathogen ' + id)
    plt.show()
    

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
    S,I,R,CC = simple_multistrain_RK_model(0, 35, 1, S, I, R, S[0] + R[0] + I[0], mu, beta, nu, sigma)
    
    for j in range(len(S[0])):
        Si = []
        Ii = []
        Ri = []
        CCi = []
        for i in range(len(S)):
            Si.append(S[i][j])
            Ii.append(I[i][j])
            Ri.append(R[i][j])
            CCi.append(CC[i][j])
        plt.figure(j+1)
        plt.plot(range(len(Ii)), Ii, label = "I")
        plt.plot(range(len(Si)), Si, label = "S")
        plt.plot(range(len(Ri)), Ri, label = "R")
        plt.plot(range(len(CCi)), CCi, label = "Cummulative Cases")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        id = str(j + 1)
        plt.title('SIR Model (one-way cross-immunity): Pathogen ' + id)
    plt.show()