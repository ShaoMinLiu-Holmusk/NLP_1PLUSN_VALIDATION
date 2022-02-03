from multiprocessing import Manager
from multiprocessing import Queue
from multiprocessing import Pool
from unittest import result
from tqdm import tqdm
from time import sleep
from itertools import repeat
from os import getpid

def updateTqdm(queueObject):
    for i in tqdm(total=10):
        queueObject.get()
    

'''
           4,5,6,3,6,6,4,7,8,9,12,6,
           4,5,6,3,6,6,4,7,8,9,12,6,
           4,5,6,3,6,6,4,7,8,9,12,6]
'''


def doingWork(lock, queueObject, task):
    
    # print('working on:', task, type(queueObject))
    
    for i in range(task):
        sleep(1)
        lock.acquire()
        queueObject.put(getpid()) # task is done
        lock.release()
    return task

def doingWorkNoLock(queueObject, task):
    
    # print('working on:', task, type(queueObject))
    
    for i in range(task):
        sleep(1)
        queueObject.put(getpid()) # task is done
    return task

def doingWorkSinArg(inputArgs):
    queueObject, task = inputArgs
    
    # print('working on:', task, type(queueObject))
    
    for i in range(task):
        sleep(1)
        queueObject.put(getpid()) # task is done
    return task

def simplePrint(task):
    print(task)

if __name__ == '__main__':
    myTasks = [4,5,6,3,6,6,4,7,8,9,12,6]
    
    thisManage = Manager()
    progressReporter = thisManage.Queue()
    # progressReporter = Queue()
    myLock = thisManage.Lock()
    finalResult = 0
    
    def mycallBack(returnObj):
        global finalResult
        finalResult = returnObj
        print('final Result is:', finalResult)
    
    mypool = Pool(4)
    inputs = zip(repeat(myLock),
                 repeat(progressReporter), 
                 myTasks)
    inputs = zip(repeat(progressReporter), 
                myTasks)
    processRun = mypool.starmap_async(doingWorkNoLock, inputs, callback=mycallBack)
    # processRun = mypool.starmap_async(doingWork, inputs, callback=mycallBack)
    # processRun = mypool.map_async(simplePrint, myTasks)
    mypool.close()
    
    # mypool.join()
    progressBar = tqdm(range(sum(myTasks)))
    print(processRun.ready())
    # print(processRun.successful())
    
    for i in progressBar:
        # print(processRun.ready())
        # thisResult = processRun.get()
        # print(thisResult)
        '''
        while True:
            if progressReporter.empty():
                print('prgress is empty')
                sleep(2)
            else:
                # myLock.acquire()
                
                # myLock.release()
                break
        '''
        pid = progressReporter.get()
        progressBar.set_postfix_str(str(pid))
        
    print(progressReporter.empty())
        
    print('is result correct:', sum(finalResult) == sum(myTasks))