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

        self.frame_count = 0

        self.total_bytes_sent = 0
        self.frame_times = []

        self.send_queue = Queue()
        self.display_queue = Queue()

        self.packet_handler = VideoPacketEncoder()

        logging.basicConfig(filename='./video/server_log.log',level=logging.INFO)
        self.logger = logging.getLogger('VideoServer')

    def setup_server(self):
        """Initialize server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
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
        self.time_per_frame = 1/self.fps
        self.logger.info(f"Video FPS: {self.fps}")

    def frame_reader(self):
        """Read frames from video file and put them in queue"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error(f"Read() error: {ret}")
                break
            try:
                self.display_queue.put(frame.copy(), timeout=1)

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])       #40% quality, fixed for now

                self.send_queue.put(buffer, timeout=1)

            except Exception as e:
                self.logger.error(f"Error in frame reader: {e}")

        self.logger.info("Finished reading video file")

    def display_frames(self):
        """Display frames from buffer with dynamic speed adjustment"""
        self.logger.info("Waiting for display buffer to fill...")

        cv2.namedWindow('Video Server', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Video Server', 250, 150)
        cv2.resizeWindow('Video Server', 750, 750)

        previous_frame_time = time.time()

        while True:
            # Sleep until next frame
            if (self.time_per_frame > time.time() - previous_frame_time):
                time.sleep(abs(self.time_per_frame - (time.time() - previous_frame_time)))

            try:
                frame = self.display_queue.get(timeout=2)
            except Empty:
                break

            cv2.imshow('Video Server', frame)
            previous_frame_time = time.time()

            # Necessary for display window to work
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def send_frame(self, frame_data:np.ndarray):
        """Split frame into packets, calculate registers and send"""
        try:
            send_start=time.time()
            self.logger.info(f"Frame length in bytes: {len(frame_data)}")
            for i in range(0, len(frame_data), self.packet_size):
                packet_data = frame_data[i:i + self.packet_size]
                is_last_packet = (i + self.packet_size) >= len(frame_data)

                registers, packet = self.packet_handler.prepare_packet(
                    packet_data,
                    is_last_packet
                )

                self.client_socket.sendall(registers)   #For RP, load every register individually
                self.client_socket.sendall(packet)

                self.total_bytes_sent += len(packet)

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

            reader_thread = Thread(target=self.frame_reader)        # Use threads to generate and display frames
            display_thread = Thread(target=self.display_frames)
            reader_thread.start()
            display_thread.start()

            self.start_time = time.time()
            while True:
                try:
                    frame_data = self.send_queue.get(timeout=5)
                except Empty:
                    self.logger.info("No more data in video, exiting...")
                    break

                self.send_frame(frame_data)
                self.print_metrics()

            print("\nTransmission complete!")

        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")

        reader_thread.join(timeout=10)
        display_thread.join(timeout=10)

    def __del__(self):
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
    PACKET_SIZE = 4011

    server = VideoServer(HOST, PORT, VIDEO_PATH, PACKET_SIZE)
    server.run()