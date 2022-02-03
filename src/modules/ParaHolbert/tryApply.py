from multiprocessing import Pool
import time
import os

def initializer():
    print("In initializer pid is {} ppid is {}".format(os.getpid(),os.getppid()))
    # apple = str(os.getpid())
    

def f(x):
    print("In f pid is {} ppid is {}".format(os.getpid(),os.getppid()))
    # print('hey:', apple)
    return x*x

if __name__ == '__main__':
    print("In main pid is {} ppid is {}".format(os.getpid(), os.getppid()))
    with Pool(processes=4, initializer=initializer) as pool:  # start 4 worker processes
        result = pool.apply(f, (10,)) # evaluate "f(10)" in a single process
        pool.apply(f, (11,)) 
        print(result)

        #result = pool.apply_async(f, (10,)) # evaluate "f(10)" in a single process
        #print(result.get())