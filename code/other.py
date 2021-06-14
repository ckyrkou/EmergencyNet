import math

def cosine_decay(epochs_tot=500,initial_lrate=1e-1,warmup=False,period=500,fade=True,fade_factor=1.,min_lr=1e-6):

    def coside_decay_full(epoch,lr,epochs_tot=epochs_tot,initial_lrate=initial_lrate,warmup=warmup,fade=fade,fade_factor=fade_factor):
        lrate = 0.5 * (1 + math.cos(((epoch * math.pi) / (epochs_tot//period)))) * initial_lrate

        if(fade):
          lrate*=math.pow(fade_factor,epoch)

        if(warmup and epoch <40):
            lrate = 1e-5
        if(lrate < min_lr):
            lrate = min_lr
        return lrate
    return coside_decay_full

def cosine_annealing(epochs_tot=500, eta_min=1e-6, eta_max=2e-4, T_max=10, fade=False):
    def coside_annealing_full(epoch, lr, epochs_tot=epochs_tot, eta_min=eta_min, eta_max=eta_max, T_max=T_max,fade=fade):
        lrate = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        if(fade == True):
            lrate = lrate * (1. - 0.5*(float(epoch) / float(epochs_tot)))
        return lrate

    return coside_annealing_full

def scheduler_step(epoch,lr):
   initial_lrate = 0.1
   drop = 0.2
   epochs_drop = 30.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

def exp_decay(k=0.025,initial_rate=0.1):
    def ed(epoch,lr,k=k,initial_rate=initial_rate):
        t=epoch
        lrate = initial_rate*math.exp(-k*t)
        if(lrate < 1e-6):
            lrate
        return lrate
    return ed