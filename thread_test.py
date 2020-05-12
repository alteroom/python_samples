import threading
import time

def process():
    while True:
        print("Process")
        time.sleep(1)

thread = threading.Thread(target=process, args=())
thread.start()

