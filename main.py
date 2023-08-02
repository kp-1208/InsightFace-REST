import multiprocessing

from MultipleIPCameras import main as producer_main
from Multi_IP_demo_client_stream import main as consumer_main
from SharedBuffer import SHARED_BUFFER

if __name__ == "__main__":
    # producer_main(SHARED_BUFFER)
    # consumer_main(SHARED_BUFFER)
    producer_process = multiprocessing.Process(target=producer_main, args=(SHARED_BUFFER,))
    #consumer_process = multiprocessing.Process(target=consumer_main, args=(SHARED_BUFFER,))

    producer_process.start()
    #consumer_process.start()

    consumer_main(SHARED_BUFFER)

    producer_process.join()
    #consumer_process.join()
