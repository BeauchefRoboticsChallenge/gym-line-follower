#!/usr/bin/env python
#Diferential evolution
import sys
import numpy as np
from numba import jit
from tqdm import tqdm

@jit(nopython=True)
def derangementNumba(n):
    v=np.arange(n)
    num=np.arange(n)
    while True:
        np.random.shuffle(v)
        if np.all((v-num)!=0):
            break
    return v

#Diferential Evolution
def diff_evolution(fobj,obj, bounds, mut=0.8, crossp=0.7, popsize=20, its=100, stopf=None, bar=False):
    ind_size=len(bounds)
    pop = np.random.rand(popsize, ind_size)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    Pfitness = np.asarray([fobj(ind,obj) for ind in pop_denorm])
    best_idx = np.argmin(Pfitness)
    best = pop_denorm[best_idx]
    for i in tqdm(range(its),disable=(not bar),leave=False,unit=" gen",file=sys.stdout):
        #Elegir padres
        ina=derangementNumba(popsize)
        inb=derangementNumba(popsize)
        inc=derangementNumba(popsize)
        #crear mutantes
        mut=np.clip((pop[ina]+mut*(pop[inb]-pop[inc])),0,1)
        cross_points = np.random.rand(popsize,ind_size) < crossp
        #Ver cross
        f=np.nonzero(cross_points.sum(1)==0)
        cross_points[f,np.random.randint(ind_size,size=len(f))]=1
        trial=np.where(cross_points,mut,pop)
        #Probar nuevo fitness
        trial_denorm=min_b + trial * diff
        fitnessT=np.zeros(popsize)
        for j in range(popsize):
            fitnessT[j]=fobj(trial_denorm[j],obj)
        #Actualizar generacion
        test=fitnessT<Pfitness
        Pfitness=np.where(test,fitnessT,Pfitness)
        indn=test.nonzero()
        pop[indn]=trial[indn]
        best_idx = np.argmin(Pfitness)
        pop_denorm = min_b + pop * diff
        best = np.copy(pop_denorm[best_idx])
        if stopf is not None:
            if Pfitness[best_idx] <= stopf:
                break
        #print(best)
    return best, Pfitness[best_idx]
