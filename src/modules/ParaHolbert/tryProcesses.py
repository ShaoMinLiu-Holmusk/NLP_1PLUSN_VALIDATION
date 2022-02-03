from multiprocessing import Process, Queue
import random
from time import sleep
from os import getpid
from tqdm import tqdm

def rand_num(queue):
    num = random.random()
    queue.put(num)


def doingWork(queueObject, task):
    # print('working on:', task, type(queueObject))
    for i in range(task):
        sleep(1)
        queueObject.put(getpid()) # task is done
    return task


if __name__ == "__main__":
    myTasks = [4,5,6,3,6,6,4,7,8,9,12,6]
    
    
    queue = Queue()

    processes = [Process(target=doingWork, args=(queue,10)) for x in range(4)]

    for p in processes:
        p.start()

        
    for i in tqdm(range(40)):
        queue.get()
    # results = [queue.get() for p in processes]