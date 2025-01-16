import socket
import cv2
import numpy as np
import time
from threading import Thread, Event
from queue import Queue, Empty
import logging
from vid_classes import *

class VideoClient:
    def __init__(self, host, port, buffer_size=120):
        self.host = host
        self.port = port
        self.client_socket = None
        
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.current_frame_data = bytearray()
        self.display_frame = None
        
        self.stop_event = Event()
        self.buffer_ready = Event()
        
        self.start_time = None
        self.frame_times = []
        self.frames_displayed = 0
        
        self.packet_decoder = VideoPacketDecoder()
        
        logging.basicConfig(filename='./video/client_log.log', level=logging.INFO)
        self.logger = logging.getLogger('VideoClient')

    def connect(self):
        """Establish connection to server"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.logger.info(f"Connecting to {self.host}:{self.port}")
        self.client_socket.connect((self.host, self.port))

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
            
        registers = np.frombuffer(register_data, dtype=np.uint32)
        packet_info = self.packet_decoder.decode_registers(register_data)
        
        """
        # Print individual register values
        print('\n-------------------------------')
        print('Register 0 =', registers[0])
        print('Register 1 =', registers[1])
        print('Register 2 =', registers[2])
        print('Register 3 =', registers[3])
        
        # Decode register information
        packet_info = self.packet_decoder.decode_registers(register_data)
        print(f'Frame Number: {packet_info["frame_number"]}')
        print(f'Packet Number: {packet_info["packet_number"]}')
        print(f'Packet Size: {packet_info["packet_size"]}')
        print(f'Last Packet: {packet_info["is_last_packet"]}')
        print('-------------------------------\n')
        """

        return packet_info
    
    def receive_packet(self, packet_size):
        """Receive packet data"""
        return self.receive_exact(packet_size)

    def receive_frame_packets(self):
        """Receive and process incoming packets"""
        try:
            while not self.stop_event.is_set():
                
                packet_info = self.receive_registers()
                if not packet_info:
                    print("Recieved wrong registers")
                    break
                
                data = self.receive_packet(packet_info['packet_size'])
                if not data:
                    print("Recieved null data")
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
        finally:
            self.stop_event.set()

    def complete_frame(self):
        """Complete current frame and add to buffer"""
        if not self.packet_decoder.frame_packets:
            print("No packets to process")
            return

        print(f"Processing frame with {len(self.packet_decoder.frame_packets)} packets")    

        frame_data = self.packet_decoder.handle_missing_packets()           #Fill frame if missing packets
        if frame_data:
            try:
                print(f"Frame data size: {len(frame_data)} bytes")
            
                nparr = np.frombuffer(frame_data, np.uint8)                
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    print(f"Decoded frame shape: {frame.shape}")
                    self.frame_buffer.put(frame)
                    print(f"Current buffer size: {self.frame_buffer.qsize()}")
                    
                    
                    if not self.buffer_ready.is_set() and \
                        self.frame_buffer.qsize() >= self.frame_buffer.maxsize // 3:            # buffer ready with enough frames
                        print("Buffer ready signal set")
                        self.buffer_ready.set()
                
                else:
                    print("Frame decode failed - frame is None")
                        
            except Exception as e:
                self.logger.error(f"Error decoding frame: {e}")
                print(f"Frame decode error: {e}")

    def display_frames(self):
        """Display frames at target FPS"""
        try:
            self.logger.info("Waiting for buffer to fill...")
            self.buffer_ready.wait()

            self.logger.info("Creating display window...")
            cv2.namedWindow('Video client', cv2.WINDOW_NORMAL)
        
            self.start_time = time.time()
            next_frame_time = self.start_time

            while not self.stop_event.is_set():
                
                while time.time() < next_frame_time:            # Try to maintain frame rate
                    time.sleep(0.001)
                
                try:
                    frame = self.frame_buffer.get_nowait()
                    print(f"Displaying frame with shape: {frame.shape}")
                    cv2.imshow('Video client', frame)
                    
                    self.frames_displayed += 1
                    self.frame_times.append(time.time())
                    
                    self.print_metrics()
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                
                next_frame_time += 1/30  # Target 30 FPS
                
        except Exception as e:
            self.logger.error(f"Error in display loop: {e}")
        finally:
            cv2.destroyAllWindows()

    def print_metrics(self):
        """Print playback metrics"""
        elapsed_time = time.time() - self.start_time
        
        recent_frames = [t for t in self.frame_times if t > time.time() - 1]
        current_fps = len(recent_frames)
        
        buffer_percent = (self.frame_buffer.qsize() / self.frame_buffer.maxsize) * 100
        
        print(f"\rPlayback FPS: {current_fps}, "
              f"Buffer: {buffer_percent:.1f}%, "
              f"Frames displayed: {self.frames_displayed}", end="")         # Not visible in console with debugging prints (may change to log)

    def run(self):
        """Main client loop"""
        try:
            self.connect()
            
            receiver_thread = Thread(target=self.receive_frame_packets)
            receiver_thread.start()
            
            display_thread = Thread(target=self.display_frames)
            display_thread.start()
            
            receiver_thread.join()
            display_thread.join()
            
            print("\nPlayback complete!")
            
        except Exception as e:
            self.logger.error(f"Client error: {e}")
        finally:
            self.stop_event.set()
            if self.client_socket:
                self.client_socket.close()

if __name__ == "__main__":
    
    HOST = '127.1.0.0'
    PORT = 65432
    
    client = VideoClient(HOST, PORT)
    client.run()