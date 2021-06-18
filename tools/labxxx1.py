import sqlite3
import multiprocessing
import numpy as np
def updater1(i):    
    print('UPDATER 1: %s'%i)
    return

def updater2(i):    
    print('UPDATER 2: %s'%i)
    return

if __name__=='__main__':
    a = np.arange(0,11)
    b = a**2
    pool = multiprocessing.Pool(2)

    
    pool.map_async(updater1, a)
    pool.map_async(updater2, b)

    pool.close()
    pool.join()