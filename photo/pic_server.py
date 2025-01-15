import socket
import numpy as np
import time

class ImageServer:
    def __init__(self, host, port, packet_size):
        self.host = host
        self.port = port
        self.packet_size = packet_size
        self.server_socket = None
        self.client_socket = None
        self.total_bytes_sent = 0
        self.start_time = None
        self.sequence_number = 0

    def setup_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

    def calculate_reg0(self, packet_size):
        """Calculate number of bytes in packet (24-bit value)"""
        return np.uint32(packet_size & 0xFFFFFF)

    def calculate_reg1(self, packet_size):
        """Calculate padding and include sequence number
        Sequence number in upper 16 bits, padding in lower 16 bits
        """
        padding = (21 - (packet_size % 21)) % 21
        return np.uint32((self.sequence_number << 16) | padding)

    def calculate_reg2(self, is_last_packet):
        """Calculate reg2 with last packet flag"""
        last_packet_flag = 0xA0000000 if is_last_packet else 0
        placeholder_value = 65792  # Fixed value according to Transmitter.pdf
        return np.uint32(last_packet_flag | placeholder_value)

    def calculate_reg3(self):
        """Placeholder for reg3"""
        return np.uint32(66063)  # Fixed value according to Transmitter.pdf

    def prepare_packet(self, data, is_last_packet):
        """Prepare packet with registers and data"""

        packet_size = len(data)
        registers = np.array([
            self.calculate_reg0(packet_size),
            self.calculate_reg1(packet_size),
            self.calculate_reg2(is_last_packet),
            self.calculate_reg3()
        ], dtype=np.uint32)
        
        return registers.tobytes() + data

    def send_image(self, image_bytes):
        """Send image data in packets and update metrics"""
        self.start_time = time.time()
        self.total_bytes_sent = 0
        self.sequence_number = 0
        
        for i in range(0, len(image_bytes), self.packet_size):
            packet_data = image_bytes[i:i + self.packet_size]
            is_last_packet = (i + self.packet_size) >= len(image_bytes)
            
            packet = self.prepare_packet(packet_data, is_last_packet)
            self.client_socket.sendall(packet) 
            
            self.total_bytes_sent += len(packet)
            self.sequence_number += 1
            
            self.print_metrics()

    def print_metrics(self):
        """Print transmission metrics"""
        elapsed_time = time.time() - self.start_time
        avg_bitrate = (self.total_bytes_sent * 8) / elapsed_time if elapsed_time > 0 else 0
        current_progress = (self.total_bytes_sent / 1024)
        
        print(f"\rProgress: {current_progress:.2f} KB sent, "
              f"Average bitrate: {avg_bitrate/1024/1024:.2f} Mbps", end="")

    def run(self, image_path):
        """Reads Image file as bytes and transmits"""        
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            print("Waiting for client connection...")
            self.client_socket, addr = self.server_socket.accept()
            print(f"Connected to client at {addr}")
            
            self.send_image(image_bytes)
            
            print("\nTransmission complete!")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()

if __name__ == "__main__":
    
    HOST = '127.1.0.0'
    PORT = 65432
    PACKET_SIZE = 2048
    IMAGE_PATH = './photos/rp_conexiones.png'  # Replace with desired image path
    
    server = ImageServer(HOST, PORT, PACKET_SIZE)
    server.setup_server()
    server.run(IMAGE_PATH)