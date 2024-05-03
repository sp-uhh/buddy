import numpy as np

def worker_init_fn(worker_id):
    st=np.random.get_state()[2]
    np.random.seed( st+ worker_id)
