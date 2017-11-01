import numpy as np
from data_process import batch_generator

names = np.array(["a","b","c","d","e"])
labels = np.array([1,2,2,2,1])
gen = batch_generator(data=[names,labels],batch_size=2)

for item in gen:
    print(item)