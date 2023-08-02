"""
Reference for Multi-Processing Locking:
https://stackoverflow.com/questions/57095895/how-to-use-multiprocessing-queue-with-lock
"""
import queue
import multiprocessing
import threading

import cv2
import time
import base64

from custom_logging import logger
from parameters import IP_CAMERAS, FRAME_WIDTH, FRAME_HEIGHT, IP_CAM_REINIT_WAIT_DURATION
from utils import roi_face_detection, create_mask
from locks import lock


# Define a class to handle camera streaming in a separate thread
class CameraStream:
    def __init__(self, camera_name: str, camera_ip: str, shared_buffer):
        self.frame = None
        self.camera_name = camera_name
        self.camera_ip = camera_ip
        self.shared_buffer = shared_buffer
        self.stream = cv2.VideoCapture(self.camera_ip)
        self.grabbed, _ = self.stream.read()
        self.initialized = self.grabbed
        self.process_this_frame = True
        if not self.grabbed:
            logger.error(f'Camera stream from {self.camera_name} (url: {self.camera_ip})) unable to initialize')
        else:
            logger.info(f'Camera stream from {self.camera_name} (url: {self.camera_ip}) initialized')

    def _read_frame(self):
        """
        Reads a frame from the stream
        """
        self.grabbed, self.frame = self.stream.read()

    def _discard_frame(self):
        """
        Reads and discards one frame
        """
        _, _ = self.stream.read()

    def release(self):
        """Releases the camera stream"""
        self.stream.release()

    def place_frame_in_buffer(self):
        """
        Places the frame in the shared buffer
        :return:
        """
        if self.process_this_frame:
            self._read_frame()
            if not self.grabbed:
                # if the frame was not grabbed, then we have reached the end of the stream
                logger.error(
                    f'Could not read a frame from the camera stream from {self.camera_name} (url: {self.camera_ip})). Releasing the stream...')
                self.release()
                self.initialized = False
            else:
                # Resize the frame if the frame size is larger than the frame size specified in parameters.py
                self.frame = cv2.resize(self.frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # Separate Region of Interest from the frame
                roi_width, roi_height, roi_x, roi_y = IP_CAMERAS[self.camera_name][1]
                roi_left, roi_top, roi_right, roi_bottom = roi_face_detection(
                    roi_width,
                    roi_height,
                    self.frame.shape[1],
                    self.frame.shape[0],
                    roi_x,
                    roi_y
                )
                mask = create_mask(self.frame, roi_left, roi_top, roi_right, roi_bottom)
                # Apply mask to the frame
                self.frame = self.frame * mask

                # _, self.frame = cv2.imencode(".jpg", self.frame)
                # self.frame = base64.b64encode(self.frame).decode("ascii")

                if isinstance(self.shared_buffer, queue.Queue):
                    with lock:
                        self.shared_buffer.put(self.frame)
                elif isinstance(self.shared_buffer, multiprocessing.queues.Queue):
                    self.shared_buffer.put(self.frame)
        else:
            self._discard_frame()

        # toggle the flag to process alternate frames to improve the performance
        self.process_this_frame = not self.process_this_frame


def create_camera(name, ip, buffer):
    camera = CameraStream(camera_name=name, camera_ip=ip, shared_buffer=buffer)

    while True:
        if camera.initialized:
            try:
                camera.place_frame_in_buffer()
            except Exception as exception:
                logger.error(f"Exception raised while placing the frame in the buffer from {camera.camera_name} (url: {camera.camera_ip})). Releasing the stream...")
                camera.stream.release()
                camera.is_initialized = False
        else:
            logger.error(f"Camera stream from {camera.camera_name} (url: {camera.camera_ip})) is not accessible. Destroying the camera object...")
            del camera
            # buffer.put(None)
            logger.info(f"Putting the thread to sleep for {name} (url: {ip})) for {IP_CAM_REINIT_WAIT_DURATION} seconds...")
            time.sleep(IP_CAM_REINIT_WAIT_DURATION)
            logger.info(f'Creating a new camera object for {name} (url: {ip}))...')
            camera = CameraStream(camera_name=name, camera_ip=ip, shared_buffer=buffer)


def main(shared_buffer):
    print("Process 1 Started")
    for camera_name in IP_CAMERAS:
        camera_ip = IP_CAMERAS[camera_name][0]
        camera_thread = threading.Thread(target=create_camera, args=(camera_name, camera_ip, shared_buffer))
        camera_thread.start()
