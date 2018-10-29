#!/usr/bin/env python

import argparse
import collections
import contextlib
import io
import logging
import math
import os
import queue
import signal
import sys
import threading
import time

import io
import os
import shutil
from subprocess import Popen, PIPE
from string import Template
from struct import Struct
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
from wsgiref.simple_server import make_server

import picamera
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import (
    WSGIServer,
    WebSocketWSGIHandler,
    WebSocketWSGIRequestHandler,
)

from ws4py.server.wsgiutils import WebSocketWSGIApplication
from aiy.leds import Leds
from aiy.leds import Pattern
from aiy.leds import PrivacyLed
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.streaming.server import StreamingServer, InferenceData

from gpiozero import Button

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

###########################################
# CONFIGURATION
WIDTH=1280
HEIGHT=720
FRAMERATE = 24
HTTP_PORT = 8082
WS_PORT = 8084
COLOR = u'#444'
BGCOLOR = u'#333'
JSMPEG_MAGIC = b'jsmp'
JSMPEG_HEADER = Struct('>4sHH')
VFLIP = False
HFLIP = False

###########################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RED_COLOR = (255, 0, 0)

JOY_COLOR = (255, 70, 0)
SAD_COLOR = (0, 0, 64)

JOY_SCORE_PEAK = 0.50
JOY_SCORE_MIN = 0.05

JOY_SOUND = ('C5q', 'E5q', 'C6q')
SAD_SOUND = ('C6q', 'E5q', 'C5q')
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')

FONT_FILE = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

BUZZER_GPIO = 22
BUTTON_GPIO = 23

@contextlib.contextmanager
def stopwatch(message):
    try:
        logger.info('%s...', message)
        begin = time.time()
        yield
    finally:
        end = time.time()
        logger.info('%s done. (%fs)', message, end - begin)


def blend(color_a, color_b, alpha):
    return tuple([math.ceil(alpha * color_a[i] + (1.0 - alpha) * color_b[i]) for i in range(3)])


def average_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0


def draw_rectangle(draw, x0, y0, x1, y1, border, fill=None, outline=None):
    assert border % 2 == 1
    for i in range(-border // 2, border // 2 + 1):
        draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i), fill=fill, outline=outline)


def normalize_bounding_box(bounding_box, width, height):
    x, y, w, h = bounding_box
    return (x / width, y / height, w / width, h / height)


def server_inference_data(width, height, faces, joy_score):
    data = InferenceData()
    for face in faces:
        x, y, w, h = normalize_bounding_box(face.bounding_box, width, height)
        color = blend(JOY_COLOR, SAD_COLOR, face.joy_score)
        data.add_rectangle(x, y, w, h, color, round(face.joy_score * 10))
        data.add_label("%.2f" % face.joy_score, x, y, color, 1)
    data.add_label('Faces: %d Avg. score: %.2f' % (len(faces), joy_score),
                   0, 0, blend(JOY_COLOR, SAD_COLOR, joy_score), 2)
    return data


class AtomicValue:

    def __init__(self, value):
        self._lock = threading.Lock()
        self._value = value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, value):
        with self._lock:
            self._value = value


class MovingAverage:

    def __init__(self, size):
        self._window = collections.deque(maxlen=size)

    def next(self, value):
        self._window.append(value)
        return sum(self._window) / len(self._window)


class Service:

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is None:
                break
            self.process(request)
            self._requests.task_done()

    def process(self, request):
        pass

    def submit(self, request):
        self._requests.put(request)

    def close(self):
        self._requests.put(None)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class Player(Service):
    """Controls buzzer."""

    def __init__(self, gpio, bpm):
        super().__init__()
        self._toneplayer = TonePlayer(gpio, bpm)

    def process(self, sound):
        self._toneplayer.play(*sound)

    def play(self, sound):
        self.submit(sound)


class Photographer(Service):
    """Saves photographs to disk."""

    def __init__(self, format, folder):
        super().__init__()
        assert format in ('jpeg', 'bmp', 'png')

        self._font = ImageFont.truetype(FONT_FILE, size=25)
        self._faces = AtomicValue(())
        self._format = format
        self._folder = folder

    def _make_filename(self, timestamp, annotated):
        path = '%s/%s_annotated.%s' if annotated else '%s/%s.%s'
        return os.path.expanduser(path % (self._folder, timestamp, self._format))

    def _draw_face(self, draw, face):
        x, y, width, height = face.bounding_box
        text = 'Joy: %.2f' % face.joy_score
        _, text_height = self._font.getsize(text)
        margin = 3
        bottom = y + height
        text_bottom = bottom + margin + text_height + margin
        draw_rectangle(draw, x, y, x + width, bottom, 3, outline='white')
        draw_rectangle(draw, x, bottom, x + width, text_bottom, 3, fill='white', outline='white')
        draw.text((x + 1 + margin, y + height + 1 + margin), text, font=self._font, fill='black')

    def process(self, camera):
        faces = self._faces.value
        timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')

        stream = io.BytesIO()
        with stopwatch('Taking photo'):
            camera.capture(stream, format=self._format, use_video_port=True)

        filename = self._make_filename(timestamp, annotated=False)
        with stopwatch('Saving original %s' % filename):
            stream.seek(0)
            with open(filename, 'wb') as file:
                file.write(stream.read())

#         if faces:
#             filename = self._make_filename(timestamp, annotated=True)
#             with stopwatch('Saving annotated %s' % filename):
#                 stream.seek(0)
#                 image = Image.open(stream)
#                 draw = ImageDraw.Draw(image)
#                 for face in faces:
#                     self._draw_face(draw, face)
#                 del draw
#                 image.save(filename)

    def update_faces(self, faces):
        self._faces.value = faces

    def shoot(self, camera):
        self.submit(camera)


class Animator(Service):
    """Controls RGB LEDs."""

    def __init__(self, leds):
        super().__init__()
        self._leds = leds

    def process(self, joy_score):
        if joy_score > 0:
            self._leds.update(Leds.rgb_on(blend(JOY_COLOR, SAD_COLOR, joy_score)))
        else:
            self._leds.update(Leds.rgb_off())

    def update_joy_score(self, joy_score):
        self.submit(joy_score)


class JoyDetector:

    def __init__(self, camera, args):
        self._done = threading.Event()
        self.camera = camera
        self.args = args

        signal.signal(signal.SIGINT, lambda signal, frame: self.stop())
        signal.signal(signal.SIGTERM, lambda signal, frame: self.stop())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        logger.info('Stopping...')
        self._done.set()

    def _run(self):
        logger.info('Starting...')
        leds = Leds()

        with contextlib.ExitStack() as stack:
            player = stack.enter_context(Player(gpio=BUZZER_GPIO, bpm=10))
            photographer = stack.enter_context(Photographer(self.args.image_format, self.args.image_folder))
            animator = stack.enter_context(Animator(leds))
            stack.enter_context(PrivacyLed(leds))

            server = None
            if self.args.enable_streaming:
                server = stack.enter_context(StreamingServer(self.camera))
                server.run()

            def take_photo():
                logger.info('Button pressed.')
                player.play(BEEP_SOUND)
                photographer.shoot(self.camera)

            button = Button(BUTTON_GPIO)
            button.when_pressed = take_photo
            
            joy_score_moving_average = MovingAverage(10)
            prev_joy_score = 0.0
            with CameraInference(face_detection.model()) as inference:
                logger.info('Model loaded.')
                player.play(MODEL_LOAD_SOUND)
                for i, result in enumerate(inference.run()):
                    faces = face_detection.get_faces(result)
                    photographer.update_faces(faces)
                    avg_joy_score = average_joy_score(faces)
                    joy_score = joy_score_moving_average.next(avg_joy_score)
                    animator.update_joy_score(joy_score)
                    if server:
                        data = server_inference_data(result.width, result.height, faces, joy_score)
                        server.send_inference_data(data)
                    if avg_joy_score > JOY_SCORE_MIN:
                        photographer.shoot(self.camera)

#                    if joy_score > JOY_SCORE_PEAK > prev_joy_score:
 #                       player.play(JOY_SOUND)
 #                   elif joy_score < JOY_SCORE_MIN < prev_joy_score:
 #                       player.play(SAD_SOUND)

                    prev_joy_score = joy_score

                    if self._done.is_set() or i == self.args.num_frames:
                        break


class StreamingHttpHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
            return
        elif self.path == '/jsmpg.js':
            content_type = 'application/javascript'
            content = self.server.jsmpg_content
        elif self.path == '/index.html':
            content_type = 'text/html; charset=utf-8'
            tpl = Template(self.server.index_template)
            content = tpl.safe_substitute(dict(
                WS_PORT=WS_PORT, WIDTH=WIDTH, HEIGHT=HEIGHT, COLOR=COLOR,
                BGCOLOR=BGCOLOR))
        else:
            self.send_error(404, 'File not found')
            return
        content = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(content))
        self.send_header('Last-Modified', self.date_time_string(time.time()))
        self.end_headers()
        if self.command == 'GET':
            self.wfile.write(content)


class StreamingHttpServer(HTTPServer):
    def __init__(self):
        super(StreamingHttpServer, self).__init__(
                ('', HTTP_PORT), StreamingHttpHandler)
        base_path = os.path.dirname(sys.argv[0])
        with io.open(base_path + '/index.html', 'r') as f:
            self.index_template = f.read()
        with io.open(base_path + '/jsmpg.js', 'r') as f:
            self.jsmpg_content = f.read()


class StreamingWebSocket(WebSocket):
    def opened(self):
        self.send(JSMPEG_HEADER.pack(JSMPEG_MAGIC, WIDTH, HEIGHT), binary=True)


class BroadcastOutput(object):
    def __init__(self, camera):
        print('Spawning background conversion process')
        self.converter = Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'yuv420p',
            '-s', '%dx%d' % camera.resolution,
            '-r', str(float(camera.framerate)),
            '-i', '-',
            '-f', 'mpeg1video',
            '-b', '800k',
            '-r', str(float(camera.framerate)),
            '-'],
            stdin=PIPE, stdout=PIPE, stderr=io.open(os.devnull, 'wb'),
            shell=False, close_fds=True)

    def write(self, b):
        self.converter.stdin.write(b)

    def flush(self):
        print('Waiting for background conversion process to exit')
        self.converter.stdin.close()
        self.converter.wait()


class BroadcastThread(Thread):
    def __init__(self, converter, websocket_server):
        super(BroadcastThread, self).__init__()
        self.converter = converter
        self.websocket_server = websocket_server

    def run(self):
        try:
            while True:
                buf = self.converter.stdout.read1(32768)
                if buf:
                    self.websocket_server.manager.broadcast(buf, binary=True)
                elif self.converter.poll() is not None:
                    break
        finally:
            self.converter.stdout.close()        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', '-n', type=int, dest='num_frames', default=-1,
                        help='Number of frames to run for, -1 to not terminate')
    parser.add_argument('--preview_alpha', '-pa', type=int, dest='preview_alpha', default=0,
                        help='Transparency value of the preview overlay (0-255).')
    parser.add_argument('--image_format', type=str, dest='image_format', default='jpeg',
                        choices=('jpeg', 'bmp', 'png'), help='Format of captured images.')
    parser.add_argument('--image_folder', type=str, dest='image_folder', default='~/Pictures',
                        help='Folder to save captured images.')
    parser.add_argument('--blink_on_error', dest='blink_on_error', default=False,
                        action='store_true', help='Blink red if error occurred.')
    parser.add_argument('--enable_streaming', dest='enable_streaming', default=False,
                        action='store_true', help='Enable streaming server.')
    parser.add_argument('--width', dest='width', default=640,
                        action='store_true', help='Streaming video width.')

    args = parser.parse_args()

    if args.preview_alpha < 0 or args.preview_alpha > 255:
        parser.error('Invalid preview_alpha value: %d' % args.preview_alpha)

    if not os.path.exists('/dev/vision_spicomm'):
        logger.error('AIY Vision Bonnet is not attached or not configured properly.')
        return 1

   
    print('Initializing camera')
    with picamera.PiCamera() as camera:
        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        # Use half of that for video streaming (820x616).
        
        camera.resolution = (WIDTH,HEIGHT)
        camera.framerate = FRAMERATE
        camera.vflip = VFLIP # flips image rightside up, as needed
        camera.hflip = HFLIP # flips image left-right, as needed
        camera.sensor_mode = 4

        time.sleep(1) # camera warm-up time
        print('Initializing websockets server on port %d' % WS_PORT)
        WebSocketWSGIHandler.http_version = '1.1'
        websocket_server = make_server(
            '', WS_PORT,
            server_class=WSGIServer,
            handler_class=WebSocketWSGIRequestHandler,
            app=WebSocketWSGIApplication(handler_cls=StreamingWebSocket))
        websocket_server.initialize_websockets_manager()
        websocket_thread = Thread(target=websocket_server.serve_forever)
        print('Initializing HTTP server on port %d' % HTTP_PORT)
        http_server = StreamingHttpServer()
        http_thread = Thread(target=http_server.serve_forever)
        print('Initializing broadcast thread')
        output = BroadcastOutput(camera)
        broadcast_thread = BroadcastThread(output.converter, websocket_server)
        print('Starting recording')
        camera.start_recording(output, 'yuv')
        print('Start Inference')
        detector = JoyDetector(camera, args)        

        try:
            print('Starting websockets thread')
            websocket_thread.start()
            print('Starting HTTP server thread')
            http_thread.start()
            print('Starting broadcast thread')
            broadcast_thread.start()
            while True:
                camera.wait_recording(1)
        except KeyboardInterrupt:
            pass
        finally:
            if args.blink_on_error:
                leds = Leds()
                leds.pattern = Pattern.blink(500)
                leds.update(Leds.rgb_pattern(RED_COLOR))
            print('Stopping recording')
            camera.stop_recording()
            print('Waiting for broadcast thread to finish')
            broadcast_thread.join()
            print('Shutting down HTTP server')
            http_server.shutdown()
            print('Shutting down websockets server')
            websocket_server.shutdown()
            print('Waiting for HTTP server thread to finish')
            http_thread.join()
            print('Waiting for websockets thread to finish')
            websocket_thread.join()


if __name__ == '__main__':
    main()
