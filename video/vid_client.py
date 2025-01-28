import socket
import cv2
import numpy as np
import time
from threading import Thread, Event
from queue import Queue, Empty
import logging
from vid_classes import *

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.frame_buffer = Queue(120)
        self.current_frame_data = bytearray()

        self.start_time = None
        self.frame_times = []
        self.frames_displayed = 0

        self.expected_packets = {}

        self.target_fps = 25.0
        self.time_per_frame = 1.0 / self.target_fps
        self.max_buffer_size = 60
        self.frame_count = 0
        self.drop_threshold = 2

        self.packet_decoder = VideoPacketDecoder()

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

    def receive_exact(self, size):
        """Receive exact number of bytes by using extend if incomplete"""
        data = bytearray()
        while len(data) < size:
            packet = self.client_socket.recv(size - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive_registers(self):
        """Receive and decode register data"""
        register_data = self.receive_exact(16)
        if not register_data:
            return None

        #registers = np.frombuffer(register_data, dtype=np.uint32)              #Raw register values
        packet_info = self.packet_decoder.decode_registers(register_data)

        """     Keep for debugging
        self.logger.info(f"------------------------------------")
        self.logger.info(f'Frame Number: {packet_info["frame_number"]}')
        self.logger.info(f'Packet Number: {packet_info["packet_number"]}')
        self.logger.info(f'Packet Size: {packet_info["packet_size"]}')
        self.logger.info(f'Last Packet: {packet_info["is_last_packet"]}')
        self.logger.info(f"------------------------------------")
        """

        return packet_info

    def receive_packet(self, packet_size):
        """Receive packet data"""
        return self.receive_exact(packet_size)

    def receive_frame_packets(self):
        """Receive and process incoming packets"""
        try:
            while True:
                packet_info = self.receive_registers()
                if not packet_info:
                    self.logger.error(f"Receive_registers() error: {packet_info}")
                    break

                data = self.receive_packet(packet_info['packet_size'])
                if not data:
                    self.logger.error(f"Receive_packet() error: {data}")
                    break

                if self.packet_decoder.check_frame_wraparound(packet_info['frame_number']):     # Wraparound frame n°
                    self.packet_decoder.current_frame = packet_info['frame_number']

                if packet_info['frame_number'] != self.packet_decoder.current_frame:            # New frame
                    self.complete_frame()
                    self.packet_decoder.current_frame = packet_info['frame_number']
                    self.packet_decoder.frame_packets.clear()

                self.packet_decoder.frame_packets[packet_info['packet_number']] = data

                if packet_info['is_last_packet']:
                    self.complete_frame()

        except Exception as e:
            self.logger.error(f"Error receiving packets: {e}")

    def complete_frame(self):
        """Modified frame handling with buffer management"""
        if not self.packet_decoder.frame_packets:
            return

        frame_data = self.packet_decoder.handle_missing_packets()
        if frame_data:
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    if self.frame_buffer.qsize() >= self.max_buffer_size * 0.9:
                        try:
                            self.frame_buffer.get_nowait()
                            self.logger.debug("Dropped oldest frame due to full buffer")
                        except Empty:
                            pass

                    self.frame_buffer.put(frame, timeout=1)

            except Exception as e:
                self.logger.error(f"Frame decode error: {e}")

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
                self.frame_count += 1

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
        try:
            receiver_thread = Thread(target=self.receive_frame_packets)
            receiver_thread.start()

            display_thread = Thread(target=self.display_frames)
            display_thread.start()

            receiver_thread.join()
            display_thread.join()

            print("\nPlayback complete!")

        except Exception as e:
            self.logger.error(f"Client error: {e}")

    def __del__(self):
        if self.client_socket:
                self.client_socket.close()

if __name__ == "__main__":

    HOST = '127.1.0.0'
    PORT = 65432

    client = VideoClient(HOST, PORT)
    client.run()