def simple_stochastic_multistrain_sde(
    beta = None, #array of beta values independent of time
    random_seed=None, #how the randoms are calculated. integer
    dt_euler=None,
    t_end=None, #time end. int
    dt_output=None, #step size. int
    n_pathogens=None, #number of pathogens we are concerned with. int
    S_init=None,
    I_init=None,
    mu=None, #single value
    nu=None,
    gamma=None, #array of gamma values
    sigma=None,
    
    corr_proc=None,
    sd_proc=None,
    
    shared_obs=False, 
    sd_obs=None, #array of something?
    
    shared_obs_C=False,
    sd_obs_C=None, #array of something?
    
    tol=None
):
    #initializes random class stuff
    if random_seed is None:
        sys_rand = random.SystemRandom()
        random_seed = sys_rand.randint(0, 2**31 - 1)
    rng = random.Random()
    rng.seed(random_seed)
    
    #returns sequence of integers 0 -> n_pathogens non-inclusive
    pathogen_ids = range(n_pathogens)
    
    #boolean like to determine if stochastic?
    stochastic = sum([sd_proc[i] > 0.0 for i in pathogen_ids]) > 0
    
    #no idea what this is supposed to mean
    has_obs_error = (sd_obs is not None) and (sum([sd_obs[i] > 0.0 for i in pathogen_ids]) > 0)
    has_obs_error_C = (sd_obs_C is not None) and (sum([sd_obs_C[i] > 0.0 for i in pathogen_ids]) > 0)
    
    #log mu; log_gamma = array with -inf if gamma of pathogen 0, log(gamma of pathogen) otherwise
    log_mu = log(mu)
    log_gamma = [float('-inf') if gamma[i] == 0.0 else log(gamma[i]) for i in range(n_pathogens)]
    
    #this should be number of intervals
    n_output = int(ceil(t_end / dt_output))
    
    def step(t, h, logS, logI, CC):
        neg_inf = float('-inf')
        
        sqrt_h = sqrt(h)
        
        log_betas = [log(beta[i]) for i in pathogen_ids]
        try:
            logR = [log1p(-(exp(logS[i]) + exp(logI[i]))) for i in pathogen_ids]
        except:
            R = [max(0.0, 1.0 - exp(logS[i]) - exp(logI[i])) for i in pathogen_ids]
            logR = [neg_inf if R[i] == 0 else log(R[i]) for i in pathogen_ids]
        
        if corr_proc == 1.0:
            noise = [rng.gauss(0.0, 1.0)] * n_pathogens
        else:
            noise = [rng.gauss(0.0, 1.0) for i in pathogen_ids]
            if corr_proc > 0.0:
                assert n_pathogens == 2
                noise[1] = corr_proc * noise[0] + sqrt(1 - corr_proc*corr_proc) * noise[1]
        for i in pathogen_ids:
            noise[i] *= sd_proc[i]
        
        dlogS = [0.0 for i in pathogen_ids]
        dlogI = [0.0 for i in pathogen_ids]
        dCC = [0.0 for i in pathogen_ids]
        
        for i in pathogen_ids:
            dlogS[i] += (exp(log_mu - logS[i]) - mu) * h
            if gamma[i] > 0.0 and logR[i] > neg_inf:
                dlogS[i] += exp(log_gamma[i] + logR[i] - logS[i]) * h
            for j in pathogen_ids:
                if i != j:
                    dlogSRij = sigma[i][j] * exp(log_betas[j] + logI[j])
                    dlogS[i] -=  dlogSRij * h
                    if stochastic:
                        dlogS[i] -= dlogSRij * noise[j] * sqrt_h
            dlogS[i] -= exp(log_betas[i] + logI[i]) * h
            dlogI[i] += exp(log_betas[i] + logS[i]) * h
            dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * h
            
            dlogS[i] -= exp(log_betas[i] + logI[i]) * noise[i] * sqrt_h #some goes back I->S
            dlogI[i] += exp(log_betas[i] + logS[i]) * noise[i] * sqrt_h #some goes S->I
            dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * noise[i] * sqrt_h #multiply two previous terms
            
            dlogI[i] -= (nu[i] + mu) * h
            
        return [logS[i] + dlogS[i] for i in pathogen_ids], \
            [logI[i] + dlogI[i] for i in pathogen_ids], \
            [CC[i] + dCC[i] for i in pathogen_ids]
    
    logS = [log(S_init[i]) for i in pathogen_ids]
    logI = [log(I_init[i]) for i in pathogen_ids]
    CC = [0.0 for i in pathogen_ids] 
    h = dt_euler
    
    
    ts = [0.0]
    logSs = [logS]
    logIs = [logI]
    CCs = [CC]
    Cs = [CC]
    
    for output_iter in range(n_output): #output_iter = 0,1,..., number of intervals
        min_h = h #initialized as dt_euler
        
        t = output_iter * dt_output #the curr interval point
        t_next_output = (output_iter + 1) * dt_output #next interval point
        
        while t < t_next_output: 
            if h < min_h:
                min_h = h
            
            t_next = t + h
            if t_next > t_next_output:
                t_next = t_next_output
            logS_full, logI_full, CC_full = step(t, t_next - t, logS, logI, CC)
        
            logS = logS_full
            logI = logI_full
            CC = CC_full
            t = t_next
        ts.append(t)
        logSs.append(logS)
        if not has_obs_error:
            logIs.append(logI)
        else:
            if shared_obs:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs[i]) for i in pathogen_ids]
            logIs.append([logI[i] + obs_errs[i] for i in pathogen_ids])
        CCs.append(CC)
        if has_obs_error_C:
            if shared_obs_C:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs_C[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs_C[i]) for i in pathogen_ids]
        else:
            obs_errs = [0.0 for i in pathogen_ids]
        Cs.append([max(0.0, CCs[-1][i] - CCs[-2][i] + obs_errs[i]) for i in pathogen_ids])

    result = OrderedDict([
        ('t', ts),
        ('logS', logSs),
        ('logI', logIs),
        ('C', Cs),
        ('random_seed', random_seed),
        ('pathogen_ids', pathogen_ids)
    ])
    return result