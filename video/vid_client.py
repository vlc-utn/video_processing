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

        self.client_socket = None
        self.expected_packets = {}

        self.target_fps = 30.0
        self.frame_interval = 1.0 / self.target_fps
        self.max_buffer_size = 60  
        self.frame_buffer = Queue(maxsize=self.max_buffer_size)
        self.last_frame_time = None
        self.frame_count = 0
        self.drop_threshold = 2  
        
        self.packet_decoder = VideoPacketDecoder()
        
        logging.basicConfig(filename='./video/client_log.log', level=logging.INFO)
        self.logger = logging.getLogger('VideoClient')

    def connect(self):
        """Establish connection to server"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.logger.info(f"Connecting to {self.host}:{self.port}")
        self.client_socket.connect((self.host, self.port))

    """
    def adaptive_frame_timing(self):
        current_time = time.time()
        buffer_size = self.frame_buffer.qsize()
        buffer_ratio = buffer_size / self.frame_buffer.maxsize

        # Store statistics
        self.stats['buffer_sizes'].append(buffer_ratio)
        if len(self.stats['buffer_sizes']) > 30:  # Keep last second of stats
            self.stats['buffer_sizes'].pop(0)

        # Calculate average buffer ratio over last few frames
        avg_buffer_ratio = sum(self.stats['buffer_sizes'][-5:]) / 5

        # Adjust playback speed based on buffer status
        if avg_buffer_ratio > self.buffer_high_threshold:
            adjusted_time = self.frame_time * (1.0 - self.speed_adjustment_factor)
        elif avg_buffer_ratio < self.buffer_low_threshold:
            adjusted_time = self.frame_time * (1.0 + self.speed_adjustment_factor)
        else:
            adjusted_time = self.frame_time

        return adjusted_time
        """
        
        
    def precise_sleep(self, target_time):
        """Hybrid sleep function for precise timing"""
        while time.time() < target_time:
            remaining = target_time - time.time()
            if remaining > 0.002: 
                time.sleep(0.001)
            else:
                pass  

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
            while not self.stop_event.is_set():
                
                packet_info = self.receive_registers()
                if not packet_info:
                    self.logger.error(f"Recieve_registers() error: {packet_info}")
                    break
                
                data = self.receive_packet(packet_info['packet_size'])
                if not data:
                    self.logger.error(f"Recieve_packet() error: {data}")
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
                    
                    if not self.buffer_ready.is_set() and \
                        self.frame_buffer.qsize() >= 10:  
                        self.buffer_ready.set()
                        
            except Exception as e:
                self.logger.error(f"Frame decode error: {e}")

    def dynamic_speed_adjustment(self):
        """Adjust playback speed based on buffer status"""
        buffer_percent = self.frame_buffer.qsize() / self.frame_buffer.maxsize
        
        if buffer_percent > 0.9:
            return self.frame_interval * 0.95  
        elif buffer_percent > 0.7:
            return self.frame_interval * 0.98  
        elif buffer_percent < 0.2:
            return self.frame_interval * 1.05  
        elif buffer_percent < 0.3:
            return self.frame_interval * 1.02  
        else:
            return self.frame_interval  


    def display_frames(self):
        """Optimized display loop with strict timing"""
        try:
            self.logger.info("Waiting for initial buffer fill...")
            self.buffer_ready.wait()
            
            cv2.namedWindow('Video client', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Video client', 1100, 150)
            cv2.resizeWindow('Video client', 750, 750)
            
            self.last_frame_time = time.time()
            next_frame_time = self.last_frame_time
            frames_this_second = 0
            last_fps_update = self.last_frame_time
            
            while not self.stop_event.is_set():
                
                current_time = time.time()
                frames_behind = int((current_time - next_frame_time) / self.frame_interval)
                
                if frames_behind >= self.drop_threshold:                        # Drop frames to catch up
                    for _ in range(frames_behind - 1):
                        try:
                            self.frame_buffer.get_nowait()
                            self.logger.debug(f"Dropped frame to catch up, {frames_behind} frames behind")
                        except Empty:
                            break
                    next_frame_time = current_time 
                
                
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    if sleep_time > 0.002: 
                        time.sleep(sleep_time - 0.002)
                    while time.time() < next_frame_time:
                        pass
                
                try:
                    frame = self.frame_buffer.get_nowait()
                    cv2.imshow('Video client', frame)
                    self.frame_count += 1
                    frames_this_second += 1
                    
                    # Update statistics
                    if current_time - last_fps_update >= 1.0:
                        buffer_percent = (self.frame_buffer.qsize() / self.frame_buffer.maxsize) * 100
                        print(f"\rPlayback FPS: {frames_this_second}, "
                              f"Buffer: {buffer_percent:.1f}%, "
                              f"Dropped: {frames_behind if frames_behind > 0 else 0}", end="")
                        frames_this_second = 0
                        last_fps_update = current_time
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Empty:
                    if self.stop_event.is_set():
                        break
                    self.logger.warning("Buffer underrun")
                    continue
                
                next_frame_time += self.frame_interval
                
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
              f"Frames displayed: {self.frames_displayed}", end="")         

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