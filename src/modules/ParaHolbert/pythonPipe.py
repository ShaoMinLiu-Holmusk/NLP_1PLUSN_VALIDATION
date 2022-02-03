# using a pipe for two-way communication
# from multiprocessing.connection import wait
import multiprocessing
from time import sleep
from datetime import datetime

def sender(conn, msgs):
    for msg in msgs:
        sleep(len(msg))
        conn.send(msg)
        print("Sent the message: {}".format(msg))
    conn.close()
  
def receiver(conn):
    while 1:
        waiting = datetime.now()
        msg = conn.recv() # going to wait till message is in
        timeTaken = datetime.now() - waiting
        if msg == "END":
            break
        
        print("Received the message: {}".format(msg))
        print('time taken:', timeTaken)
  
if __name__ == "__main__":
    # messages to be sent
    msgs = ["hello", "hey", "hru?", "END"]
  
    # creating a pipe
    parent_conn, child_conn = multiprocessing.Pipe()
  
    # creating new processes
    p1 = multiprocessing.Process(target=sender, args=(parent_conn,msgs))
    p2 = multiprocessing.Process(target=receiver, args=(child_conn,))
    # p3 = multiprocessing.Process(target=receiver, args=(child_conn,))
  
    # running processes
    p1.start()
    p2.start()
    # p3.start()
  
    # wait until processes finish
    p1.join()
    p2.join()
    # p3.join()