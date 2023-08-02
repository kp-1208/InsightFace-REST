import argparse
import base64
import glob
import cv2
import PIL
import io
import logging
import multiprocessing
import os
import shutil
import time
from distutils import util
from functools import partial
from itertools import chain, islice, cycle

import msgpack
import numpy as np
import requests
import ujson
import logging
logging.basicConfig(level=logging.DEBUG)


dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()


def to_bool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False


class IFRClient:

    def __init__(self, host: str = 'http://localhost', port: int = '18081'):
        self.server = f'{host}:{port}'
        self.sess = requests.Session()
        
    def detect_from_live_stream(self, mode='data', threshold=0.6, extract_embedding=True, return_face_data=True,
                                return_landmarks=True, embed_only=False, limit_faces=0, use_msgpack=True):
        # Open the video capture (assuming the default webcam, change the parameter if needed)
        cap = cv2.VideoCapture("http://192.168.15.6:4747/video")
        frames_buffer = []
        batch_size = 16
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 360))
            if not ret:
                break
            #print(type(frame),"    ", frame)
            # Convert the frame to base64 data
            _, buffer = cv2.imencode('.jpg', frame)
            data = base64.b64encode(buffer).decode('ascii')
            frames_buffer.append(data)
            #print("Buffer Size:", len(frames_buffer), "    ", data)
            #data = buffer
            if len(frames_buffer) >= batch_size:
                # Call the extract method to detect faces on the current batch of frames
                faces_data = self.extract(frames_buffer, mode=mode, server=self.server, threshold=threshold,
                                          extract_embedding=extract_embedding, return_face_data=return_face_data,
                                          return_landmarks=return_landmarks, embed_only=embed_only,
                                          limit_faces=limit_faces, use_msgpack=use_msgpack)

                # Draw bounding boxes around the detected faces in each frame of the batch
                for frame_data, faces in zip(frames_buffer, faces_data['data']):
                    for face_data in faces['faces']:
                        bbox = face_data['bbox']
                        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                        print("Face Detection Area:", x,y,w,h)
                        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                    # Display the frame with detected faces
                    cv2.imshow('Live Stream Face Detection', frame)
                frames_buffer.clear()
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                break


            '''# Display the frame (optional, for visualization)
            cv2.imshow('Live Stream Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                break'''

        # Release the video capture and close any remaining windows
        cap.release()
        cv2.destroyAllWindows()


    def server_info(self, server: str = None, show=True):
        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        info = self.sess.get(info_uri).json()

        if show:
            server_uri = self.server
            backend_name = info['models']['inference_backend']
            det_name = info['models']['det_name']
            rec_name = info['models']['rec_name']
            rec_batch_size = info['models']['rec_batch_size']
            det_batch_size = info['models']['det_batch_size']
            det_max_size = info['models']['max_size']

            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection image size:   {det_max_size}\n'
                  f'    Detection batch size:   {det_batch_size}\n'
                  f'    Recognition model:      {rec_name}\n'
                  f'    Recognition batch size: {rec_batch_size}')

        return info

    def extract(self, data: list,
                mode: str = 'paths',
                server: str = None,
                threshold: float = 0.6,
                extract_embedding=True,
                return_face_data=False,
                return_landmarks=False,
                embed_only=False,
                limit_faces=0,
                use_msgpack=True):
        #print(type(data), data)
        start = time.time()
        if server is None:
            server = self.server

        #print(data)
        
        extract_uri = f'{server}/extract'

        if mode == 'data':
            images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)
            
        print("IMAGES:", images.keys())
        print(len(images['data']))
        print(type(images['data'][0]))
        print(len(images['data'][0]))
        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   return_landmarks=return_landmarks,
                   embed_only=embed_only,  # If set to true API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   use_rotation=True,
                   msgpack=use_msgpack,
                   )

        resp = self.sess.post(extract_uri, json=req, timeout=120)
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)
        print(content.get('data'))
        images = content.get('data')
        #print(type(images),"    ", images)
        for im in images:
            status = im.get('status')
            print(status)
            if status != 'ok':
                print(content.get('traceback'))
                break
            faces = im.get('faces', [])
            for i, face in enumerate(faces):
                #print(face)
                norm = face.get('norm', 0)
                prob = face.get('prob')
                size = face.get('size')
                facedata = face.get('facedata')
                #print(type(facedata))
                if facedata:
                    if size > 20 and norm > 14:
                        save_crop(facedata, f'crops/{i}_{size}_{norm:2.0f}_{prob}.jpg')
                        
        end = time.time()
        print("Detections Done!")
        print(f"Processing time for batch of {len(images)} frames: {(end-start):.2f} seconds")

        return content


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='defa')

    parser.add_argument('-p', '--port', default=18081, type=int, help='Port')
    parser.add_argument('-u', '--uri', default='http://localhost', type=str, help='Server hostname or ip with protocol')
    parser.add_argument('-i', '--iters', default=10, type=int, help='Number of iterations')
    parser.add_argument('-t', '--threads', default=12, type=int, help='Number of threads')
    parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
    parser.add_argument('-d', '--dir', default=None, type=str, help='Path to directory with images')
    parser.add_argument('-n', '--num_files', default=10000, type=int, help='Number of files per test')
    parser.add_argument('-lf', '--limit_faces', default=0, type=int, help='Number of files per test')
    parser.add_argument('--embed', default='True', type=str, help='Extract embeddings, otherwise run detection only')
    parser.add_argument('--embed_only', default='False', type=str,
                        help='Omit detection step. Expects already cropped 112x112 images')

    args = parser.parse_args()

    allowed_ext = '.jpeg .jpg .bmp .png .webp .tiff'.split()

    client = IFRClient(host=args.uri, port=args.port)
    
    client.detect_from_live_stream(mode='data', threshold=0.6, extract_embedding=to_bool(args.embed),
                                       return_face_data=False, return_landmarks=False, embed_only=False,
                                       limit_faces=args.limit_faces, use_msgpack=True)
    
