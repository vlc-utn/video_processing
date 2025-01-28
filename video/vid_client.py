import socket
import cv2
import numpy as np
import time
from threading import Thread, Event
from queue import Queue, Empty
import logging

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.frame_buffer = Queue(120)

        self.start_time = None
        self.frame_times = []
        self.frames_displayed = 0
        self.frame_packets = {}

        self.target_fps = 25.0
        self.time_per_frame = 1.0 / self.target_fps
        self.max_buffer_size = 60

        logging.basicConfig(filename='./video/client_log.log', level=logging.INFO)
        self.logger = logging.getLogger('VideoClient')

    def connect(self) -> bool:
        """Establish connection to server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.logger.info(f"Connecting to {self.host}:{self.port}")
            self.client_socket.connect((self.host, self.port))
            return True
        except Exception:
            self.logger.error("Couldn't connect to server. Exiting...")
            print("Couldn't connect to server. Exiting...")
            return False

    def receive_frame_packets(self):
        """Receive and process incoming packets"""
        while True:
            # Read registers
            regs = self.client_socket.recv(16, socket.MSG_WAITALL)
            regs = np.frombuffer(regs, np.uint32)

            packet_size = regs[0] - regs[1]
            packets_in_frame = (((regs[2] >> 24) & 0b111) << 3) | ((regs[2] >> 16) & 0b111)
            packet_number = (regs[3] >> 24) & 0b111111

            # Get data
            # NOTE: here, packet size should be read, but that is internal of the FPGA
            data = self.client_socket.recv(regs[0], socket.MSG_WAITALL)
            data = np.frombuffer(data[0:packet_size], np.uint8)
            if (packet_number == 0):
                current_frame_data = data
            else:
                current_frame_data = np.hstack([current_frame_data, data])

            if packet_number == packets_in_frame - 1:            # New frame
                frame = cv2.imdecode(current_frame_data, cv2.IMREAD_COLOR)
                current_frame_data = None
                self.frame_buffer.put(frame, timeout=1)

    def display_frames(self):
        """Optimized display loop with strict timing"""
        try:
            self.logger.info("Waiting for initial buffer fill...")

            cv2.namedWindow('Video client', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Video client', 1100, 150)
            cv2.resizeWindow('Video client', 750, 750)

            previous_frame_time = time.time()

            while True:
                if (self.time_per_frame > time.time() - previous_frame_time):
                    time.sleep(abs(self.time_per_frame - (time.time() - previous_frame_time)))

                try:
                    frame = self.frame_buffer.get(timeout=1)
                except Empty:
                    break
                cv2.imshow('Video client', frame)
                previous_frame_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Error in display loop: {e}")
        finally:
            cv2.destroyAllWindows()

    def print_metrics(self):
        """Print playback metrics"""
        recent_frames = [t for t in self.frame_times if t > time.time() - 1]
        current_fps = len(recent_frames)

        buffer_percent = (self.frame_buffer.qsize() / self.frame_buffer.maxsize) * 100

        print(f"\rPlayback FPS: {current_fps}, "
              f"Buffer: {buffer_percent:.1f}%, "
              f"Frames displayed: {self.frames_displayed}", end="")

    def run(self):
        """Main client loop"""
        if not self.connect():
            return None

        display_thread = Thread(target=self.display_frames)
        display_thread.start()

        print("Entering")
        self.receive_frame_packets()
        print("Exiting")

        display_thread.join()

        print("\nPlayback complete!")

    def __del__(self):
        if self.client_socket:
                self.client_socket.close()

if __name__ == "__main__":

    HOST = '127.1.0.0'
    PORT = 65432

    client = VideoClient(HOST, PORT)
    client.run()