import socket
import cv2
import numpy as np
import time
from threading import Thread, Event
from queue import Queue, Empty, Full
import logging
from vid_classes import *

class VideoServer:
    def __init__(self, host, port, video_path, packet_size=2048):
        self.host = host
        self.port = port
        self.video_path = video_path
        self.packet_size = packet_size

        self.server_socket = None
        self.client_socket = None

        self.cap = None
        self.frame_count = 0
        self.fps = 30

        self.total_bytes_sent = 0
        self.start_time = None
        self.frame_times = []

        #self.display_frame = None
        #self.display_event = Event()

        self.send_queue = Queue(maxsize=240)            # 8 seconds buffer at 30fps for network
        self.display_queue = Queue(maxsize=120)         # 4 seconds buffer for display
        self.stop_event = Event()
        self.buffer_ready = Event()
        self.buffer_low = Event()

        self.min_buffer_threshold = 0.20
        self.buffer_recovery_threshold = 0.60
        self.current_playback_speed = 1.0

        self.packet_handler = VideoPacketEncoder()

        logging.basicConfig(filename='./video/server_log.log',level=logging.INFO)
        self.logger = logging.getLogger('VideoServer')

    def setup_server(self):
        """Initialize server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.logger.info(f"Server listening on {self.host}:{self.port}")
        except socket.error as e:
            self.logger.error(f"Socket binding error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Server setup error: {e}")
            raise

    def setup_video(self):
        """Initialize video capture"""
        self.logger.info(f"Opening video file: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"Error opening video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"Video FPS: {self.fps}")

    def frame_reader(self):
        """Read frames from video file and put them in queue"""
        try:
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error(f"Read() error: {ret}")
                    break

                while not self.stop_event.is_set() and self.send_queue.full():
                    time.sleep(0.001)

                try:
                    self.display_queue.put(frame.copy(), timeout=1)
                except Full:
                    sec = time.strftime("%d %H:%M:%S", time.localtime())
                    self.logger.warning(f"{sec}: Display queue full, skipping frame")
                    continue

                if not self.buffer_ready.is_set() and \
                    self.display_queue.qsize() >= self.display_queue.maxsize * 0.8:
                    self.logger.info("Display buffer ready")
                    self.buffer_ready.set()

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])       #40% quality, fixed for now

                if not ret:
                    self.logger.error(f"Imencode() error: {ret}")
                    continue

                compressed_frame = buffer.tobytes()
                try:
                    self.send_queue.put(compressed_frame, timeout=1)
                except Full:
                    sec = time.strftime("%d %H:%M:%S", time.localtime())
                    self.logger.warning(f"{sec}: Send queue full, skipping frame")
                    continue

            self.logger.info("Finished reading video file")

        except Exception as e:
            self.logger.error(f"Error in frame reader: {e}")
        finally:
            self.send_queue.put(None)
            self.display_queue.put(None)

    def dynamic_frame_delay(self, buffer_percent):
        """Calculate frame delay based on buffer status"""
        if buffer_percent < self.min_buffer_threshold:

            self.current_playback_speed = 0.8
            if not self.buffer_low.is_set():
                sec = time.strftime("%d %H:%M:%S", time.localtime())
                self.logger.warning(f"{sec}: Buffer low, reducing playback speed")
                self.buffer_low.set()

        elif buffer_percent > self.buffer_recovery_threshold and self.buffer_low.is_set():
            self.current_playback_speed = 1.0
            self.buffer_low.clear()
            self.logger.info("Buffer recovered, resuming normal speed")

        return 1 / (self.fps * self.current_playback_speed)

    def display_frames(self):
        """Display frames from buffer with dynamic speed adjustment"""
        try:
            self.logger.info("Waiting for display buffer to fill...")
            self.buffer_ready.wait()

            cv2.namedWindow('Video Server', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Video Server', 250, 150)
            cv2.resizeWindow('Video Server', 750, 750)

            next_frame_time = time.time()
            frames_displayed = 0
            display_times = []

            while not self.stop_event.is_set():
                try:
                    buffer_percent = self.display_queue.qsize() / self.display_queue.maxsize
                    frame_delay = self.dynamic_frame_delay(buffer_percent)

                    while time.time() < next_frame_time:
                        time.sleep(0.001)

                    frame = self.display_queue.get_nowait()
                    if frame is None:
                        break

                    cv2.imshow('Video Server', frame)

                    frames_displayed += 1
                    display_times.append(time.time())

                    # Display server playback metrics
                    """
                    if frames_displayed % 30 == 0:
                        recent_frames = [t for t in display_times if t > time.time() - 1]
                        current_fps = len(recent_frames)
                        print(f"\rDisplay FPS: {current_fps:.1f}, "
                              f"Buffer: {buffer_percent*100:.1f}%, "
                              f"Speed: {self.current_playback_speed:.2f}x", end="")
                    """

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_event.set()
                        break

                except Empty:
                    if self.stop_event.is_set():
                        break
                    sec = time.strftime("%d %H:%M:%S", time.localtime())
                    self.logger.warning(f"{sec}: Buffer underrun")
                    continue

                next_frame_time += frame_delay

        except Exception as e:
            self.logger.error(f"Error in display loop: {e}")
        finally:
            cv2.destroyAllWindows()

    def send_frame(self, frame_data):
        """Split frame into packets, calculate registers and send"""

        try:
            send_start=time.time()

            for i in range(0, len(frame_data), self.packet_size):
                packet_data = frame_data[i:i + self.packet_size]
                is_last_packet = (i + self.packet_size) >= len(frame_data)

                registers, packet = self.packet_handler.prepare_packet(
                    packet_data,
                    is_last_packet
                )

                self.client_socket.sendall(registers)   #For RP, load every register individually
                self.client_socket.sendall(packet)

                self.total_bytes_sent += len(packet) + len(registers)

            elapsed = time.time() - send_start
            if elapsed < 1/self.fps:
                time.sleep(1/self.fps - elapsed)

            self.frame_count += 1
            self.frame_times.append(time.time())
            self.packet_handler.next_frame()

        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            raise

    def print_metrics(self):
        """Print transmission metrics"""
        elapsed_time = time.time() - self.start_time

        bitrate = (self.total_bytes_sent * 8) / elapsed_time if elapsed_time > 0 else 0

        recent_frames = [t for t in self.frame_times if t > time.time() - 1]
        current_fps = len(recent_frames)

        print(f"\rProgress: {self.total_bytes_sent/1024/1024:.2f} MB sent, "
              f"Bitrate: {bitrate/1024/1024:.2f} Mbps, "
              f"FPS: {current_fps}", end="")

    def run(self):
        """Main server loop"""
        try:
            self.setup_server()
            self.setup_video()

            print("Waiting for client connection...")
            try:
                self.client_socket, addr = self.server_socket.accept()
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"Connected to client at {addr}")
                self.logger.info(f"Client connected from {addr}")
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                raise

            reader_thread = Thread(target=self.frame_reader)        #Use threads to generate and display frames
            display_thread = Thread(target=self.display_frames)
            reader_thread.start()
            display_thread.start()

            self.start_time = time.time()
            next_frame_time = self.start_time

            while not self.stop_event.is_set():
                try:
                    frame_data = self.send_queue.get()
                    if frame_data is None:
                        self.logger.error(f"Queue get() error: {frame_data}")
                        break

                    while time.time() < next_frame_time:
                        time.sleep(0.001)

                    self.send_frame(frame_data)
                    self.print_metrics()

                    next_frame_time += 1/self.fps

                except Exception as e:
                    self.logger.error(f"Frame processing error: {e}")
                    raise

            print("\nTransmission complete!")

        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
            raise e
        finally:
            self.stop_event.set()
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":

    HOST = '127.1.0.0'
    PORT = 65432
    VIDEO_PATH = './video/CiroyLosPersas.mp4'       # Replace with video path
    PACKET_SIZE = 2048

    server = VideoServer(HOST, PORT, VIDEO_PATH, PACKET_SIZE)
    server.run()