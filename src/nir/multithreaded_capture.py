"""BSD 2-Clause License

Copyright (c) 2019, Allied Vision Technologies GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import copy
import cv2
import threading
import queue
import numpy
import os
import datetime
import time
import glob

from typing import Optional

import numpy as np
from vimba import *

FRAME_QUEUE_SIZE = 10
HEIGHT = 1080
WIDTH = 1920

#ALVIUM U-501 NIR: 2592 (H) × 1944 (V)
OFFSET_X = (2592 - WIDTH) // 2
OFFSET_Y = (1944 - HEIGHT) // 2

DATA_FOLDER = "..\data\\recordings"
BACKGROUND_FOLDER = "..\data\\calibration\\background"

AUTO_SAVE = True
AUTO_SAVE_INTERVAL = 2
FREERUN = False

VIDEO = True
FPS = 40.0
AUTO_EXPOSURE = False
EXPOSURE = 1000

FOURCC = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')

def print_preamble():
    print('////////////////////////////////////////////')
    print('/// Vimba API Multithreading Example ///////')
    print('////////////////////////////////////////////\n')
    print(flush=True)


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    # Helper function inserting 'cam_id' into given frame. This function
    # manipulates the original image buffer inside frame object.
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame


def resize_if_required(frame: Frame) -> numpy.ndarray:
    # Helper function resizing the given frame, if it has not the required dimensions.
    # On resizing, the image data is copied and resized, the image inside the frame object
    # is untouched.
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != HEIGHT) or (frame.get_width() != WIDTH):
        cv_frame = cv2.resize(cv_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame


def create_dummy_frame() -> numpy.ndarray:
    cv_frame = numpy.zeros((50, 640, 1), numpy.uint8)
    cv_frame[:] = 0

    cv2.putText(cv_frame, 'No Stream available. Please connect a Camera.', org=(30, 30),
                fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))

    except queue.Full:
        pass


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    # Helper function that tries to set a given value. If setting of the initial value failed
    # it calculates the nearest valid value and sets the result. This function is intended to
    # be used with Height and Width Features because not all Cameras allow the same values
    # for height and width.
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)

    except VimbaFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        if feat_value <= min_:
            val = min_

        elif feat_value >= max_:
            val = max_

        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        feat.set(val)

        msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
               'Using nearest valid value \'{}\'. Note that, this causes resizing '
               'during processing, reducing the frame rate.')
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


# Thread Objects
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        # This method is executed within VimbaC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:

            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', HEIGHT)
        set_nearest_value(self.cam, 'Width', WIDTH)
        set_nearest_value(self.cam, 'OffsetX', OFFSET_X)
        set_nearest_value(self.cam, 'OffsetY', OFFSET_Y)

        # Try to enable automatic exposure time setting
        try:
            if AUTO_EXPOSURE:
                self.cam.GainAuto.set('Off')
                self.cam.Gain.set(0)
                self.cam.ExposureAuto.set('Once')
            else:
                self.cam.ExposureAuto.set('Off')
                self.cam.ExposureTime.set(EXPOSURE)
                # self.cam.GainAuto.set('Once')

        except (AttributeError, VimbaFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
                self.cam.get_id()))

        self.cam.set_pixel_format(PixelFormat.Mono8)
        if FREERUN:
            self.cam.TriggerMode.set('Off')
        else:
            self.cam.TriggerSource.set('Line0')
            self.cam.TriggerSelector.set('FrameStart')
            self.cam.TriggerMode.set('On')

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

        try:
            with self.cam:
                self.setup_camera()

                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()

                finally:
                    self.cam.stop_streaming()

        except VimbaCameraError:
            pass

        finally:
            try_put_frame(self.frame_queue, self.cam, None)

        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))


class FrameConsumer(threading.Thread):
    def __init__(self, data_path: str, background_path: str, frame_queue: queue.Queue, saving_queue_dict: dict[str, queue.Queue]):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.saving_queue_dict = saving_queue_dict
        self.data_path = data_path
        self.background_path = background_path

    def load_lf_images(self,path):

        try:
            lst = []
            all_paths = sorted(glob.glob(path))

            for curr_path in all_paths:
                im = cv2.imread(curr_path)

                if im is None:
                    continue
                lst.append(im)

            return lst

        except Exception:
            return []

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        counter = 0
        counter_bg = 0

        frames = {}
        alive = True
        save = False
        record = False
        lf_div = False

        lf_images = []

        if VIDEO:
            fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
            writers = {}

        self.log.info('Thread \'FrameConsumer\' started.')

        t0 = time.perf_counter()
        frame_last = -1

        while alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()

                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame

                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                if lf_div and len(lf_images) > 0:
                    cv_images_temp = [np.divide(im.astype(float),lf_im.astype(float)) for im,lf_im in zip(cv_images,lf_images)]
                    cv_images = [(np.clip(im*128,0,255)).astype(np.uint8) for im in cv_images_temp]


                concat_images = numpy.concatenate(cv_images, axis=1)
                concat_images = cv2.resize(concat_images, None, fx=0.25, fy=0.25)
                #concat_images = cv2.cvtColor(concat_images, cv2.COLOR_GRAY2BGR)

                if VIDEO:
                    if record:
                        #cv2.putText(concat_images, 'recording', (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                        t_now = time.perf_counter()
                        frame_now = int((t_now - t0) * FPS)
                        if frame_now > frame_last:
                            frame_last = frame_now
                            for im, cam_id in zip(cv_images, sorted(frames.keys())):
                                self.saving_queue_dict[cam_id].put(im,block=True)

                elif save or (AUTO_SAVE and time.perf_counter() - t0 > AUTO_SAVE_INTERVAL):
                    print(f'saving frame {counter:04d}...')
                    if not os.path.exists(self.data_path):
                        os.mkdir(self.data_path)

                    for im, cam_id in zip(cv_images, sorted(frames.keys())):
                        cv2.imwrite(os.path.join(self.data_path, f'{cam_id}_{counter:04d}.png'), im)

                    counter = counter + 1
                    save = False
                    t0 = time.perf_counter()

                cv2.imshow(IMAGE_CAPTION, concat_images)

            # If there are no frames available, show dummy image instead
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            # Check for shutdown condition
            key = cv2.waitKey(1) & 255
            if key == 255:
                continue

            elif KEY_CODE_ENTER == key:
                cv2.destroyAllWindows()
                alive = False
                for cam_id in writers:
                    writers[cam_id].release()

            elif key == ord('s'):
                save = True

            elif key == ord('b'):

                if lf_div:
                    self.log.warning('Cannot refresh background while lightfield compensation is turned on, please turn this off first...')
                    continue

                self.log.info('updating and storing new backgrounds...')

                counter_bg = counter_bg + 1
                for im, cam_id in zip(cv_images, sorted(frames.keys())):
                    cv2.imwrite(os.path.join(self.background_path, f'{cam_id}_{counter_bg:04d}.png'), im)

                lf_path = f'{self.background_path}\*_{counter_bg:04d}.png'
                lf_images = self.load_lf_images(lf_path)

            elif key == ord('l'):
                lf_div = ~lf_div

            elif key == ord('r'):
                if VIDEO:
                    record = ~record
                    self.log.info('start recording...' if record else 'stopped recording')

        self.log.info('Thread \'FrameConsumer\' terminated.')


class FrameRecorder(threading.Thread):
    def __init__(self, cam_id: str, data_path: str, saving_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.saving_queue = saving_queue
        self.killswitch = threading.Event()
        self.cam_id = cam_id
        self.data_path = data_path
        full_data_path = os.path.join(data_path, f'{cam_id}.mp4')
        #self.writer = cv2.VideoWriter(full_data_path, FOURCC, FPS, (WIDTH, HEIGHT))

    def stop(self):
        self.killswitch.set()

    def run(self):

        alive = True

        self.log.info('Thread \'FrameRecorder({})\' started.'.format(self.cam_id))

        frame = 0
        while alive:
            try:
                saving_frame = self.saving_queue.get(block=True, timeout=1.0)
                frame = frame + 1
                cv2.imwrite(os.path.join(self.data_path,f'{self.cam_id}_{frame:06d}.png'), saving_frame)
                #self.writer.write(saving_frame)

            except Exception:
                #print ('no frame')
                pass

            alive = not(self.killswitch.is_set())

        #self.writer.release()
        self.log.info('Thread \'FrameRecorder({})\' terminated.'.format(self.cam_id))


class MainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()

        self.recorders = {}
        self.saving_queue_dict = {}

        now = datetime.datetime.now()
        self.data_path = os.path.join(DATA_FOLDER, now.strftime('%Y%m%d_%H%M%S'))
        print(self.data_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.background_path = os.path.join(BACKGROUND_FOLDER, now.strftime('%Y%m%d_%H%M%S'))
        print(self.background_path)
        if not os.path.exists(self.background_path):
            os.mkdir(self.background_path)

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()
                self.saving_queue_dict[cam.get_id()] = queue.Queue(maxsize=1)
                self.recorders[cam.get_id()] = FrameRecorder(cam.get_id(), self.data_path, self.saving_queue_dict[id])
                self.recorders[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()
                recorder = self.recorders.pop(cam.get_id())
                recorder.stop()
                recorder.join()
                self.saving_queue_dict.pop(cam.get_id())

    def run(self):
        log = Log.get_instance()
        consumer = FrameConsumer(self.data_path, self.background_path, self.frame_queue, self.saving_queue_dict)

        vimba = Vimba.get_instance()
        vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('Thread \'MainThread\' started.')

        with vimba:
            # Construct FrameProducer threads for all detected cameras
            for cam in vimba.get_all_cameras():
                cam_id = cam.get_id()
                self.producers[cam_id] = FrameProducer(cam, self.frame_queue)
                self.saving_queue_dict[cam_id] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
                self.recorders[cam_id] = FrameRecorder(cam_id, self.data_path, self.saving_queue_dict[cam_id])

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()
                for recorder in self.recorders.values():
                    recorder.start()

            # Start and wait for consumer to terminate
            vimba.register_camera_change_handler(self)
            consumer.start()
            consumer.join()
            vimba.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()

                # Initiate concurrent shutdown
                for recorder in self.recorders.values():
                    recorder.stop()

                # Wait for shutdown to complete
                for recorder in self.recorders.values():
                    recorder.join()

        log.info('Thread \'MainThread\' terminated.')

if __name__ == '__main__':
    print_preamble()
    main = MainThread()
    main.start()
    main.join()
