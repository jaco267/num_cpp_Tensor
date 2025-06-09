
import numpy as np 
import os
import pyrallis
from dataclasses import dataclass
from config import TrainConfig
  
import random
import math
from numpy.random import randint

e2max = (2 ** 32) - 1
seed = 42
np.random.seed(seed)
random.seed(seed)
def generate_gaussian_noise(mean, stddev):
  #https://stackoverflow.com/questions/65871948/same-random-numbers-in-c-as-computed-by-python3-numpy-random-rand
  # Generate two uniformly distributed random numbers between 0 and 1
  i0 = randint(256**4, dtype='<u4', size=1)[0]
  i1 = randint(256**4, dtype='<u4', size=1)[0]
  u1 = i0/e2max
  u2 = i1/e2max
  # Apply Box-Muller transform
  z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
  return z0 * stddev + mean

@pyrallis.wrap()    
def main(cfg: TrainConfig):
  if cfg.opt == 0:
    np.random.seed(42)
    print(np.random.randint(256**4, dtype='<u4', size=1)[0])
    print(np.random.randint(256**4, dtype='<u4', size=1)[0])
    print(generate_gaussian_noise(0,0.5))
    print(generate_gaussian_noise(0,0.5))
  else:
    pass

if __name__ == '__main__':
  main()

  