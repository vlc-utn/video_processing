import socket
import cv2
import numpy as np
import time
from threading import Thread, Event
from queue import Queue
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
        
        self.frame_queue = Queue(maxsize=120)  # 4 seconds buffer at 30fps
        self.stop_event = Event()
        
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
                    print("Error in read()")
                    break
                    
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])   #80% quality, fixed for now

                if not ret:
                    print("Error in imencode()")
                    continue
                    
                compressed_frame = buffer.tobytes()
                self.frame_queue.put(compressed_frame)
                
            self.logger.info("Finished reading video file")
        
        except Exception as e:
            self.logger.error(f"Error in frame reader: {e}")
        finally:
            self.frame_queue.put(None)

    def send_frame(self, frame_data):
        """Split frame into packets, calculate registers and send"""
        
        try:
            
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
                print(f"Connected to client at {addr}")
                self.logger.info(f"Client connected from {addr}")
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                raise
            
            time.sleep(0.5)     #Delay for stable connection

            reader_thread = Thread(target=self.frame_reader)        #Use threads to generate frames
            reader_thread.start()
        
            self.start_time = time.time()
            next_frame_time = self.start_time
        
            while not self.stop_event.is_set():
                try:
                    frame_data = self.frame_queue.get()
                    if frame_data is None:
                        break
                
                    while time.time() < next_frame_time:            #Try to maintain frame rate
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

if __name__ == "__main__":
    
    HOST = '127.1.0.0'
    PORT = 65432
    VIDEO_PATH = './video/CiroyLosPersas.mp4'       # Replace with your video path
    PACKET_SIZE = 2048
    
    server = VideoServer(HOST, PORT, VIDEO_PATH, PACKET_SIZE)
    server.run()