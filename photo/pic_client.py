import socket
import numpy as np

class ImageClient:
    def __init__(self, host, port, output_path):
        self.host = host
        self.port = port
        self.output_path = output_path
        self.client_socket = None
        self.previous_packet = None
        self.expected_sequence = 0
        self.image_bytes = bytearray()

    def connect(self):
        """Establish connection to server"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to {self.host}:{self.port}")
        self.client_socket.connect((self.host, self.port))

    def receive_exact(self, size):
        """Receive exact number of bytes"""
        data = bytearray()
        while len(data) < size:
            packet = self.client_socket.recv(size - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive_packet(self):
        """Receive and parse a single set of registers and a packet
        Checks for missing and incomplete packets        
        """
        register_data = self.receive_exact(16)
        if not register_data or len(register_data) != 16:
            print('Recieved incorrect registers')
            return None, None, True
            
        registers = np.frombuffer(register_data, dtype=np.uint32)

        expected_size = registers[0] & 0xFFFFFF  # reg0: 24-bit size
        sequence_num = registers[1] >> 16  # reg1: 16 MSB
        is_last_packet = (registers[2] & 0xF0000000) == 0xA0000000  # reg2: 4 MSB

        print('\n-------------------------------')
        print('Sequence Number =', sequence_num)
        print('Packet size =', expected_size)
        print('Padding =', (registers[1] & 0x0000FFFF))
        print('Reg0 =', registers[0])
        print('Reg1 =', registers[1])
        print('Reg2 =', registers[2])
        print('Reg3 =', registers[3])
        print('-------------------------------\n')
        
        if sequence_num != self.expected_sequence:
            print(f"Missing packet detected. Expected {self.expected_sequence}, got {sequence_num}")
            if self.previous_packet is not None:
                return self.previous_packet, False, False

        data = self.receive_exact(expected_size)
        if not data:
            print('Recieved NULL data')
            return None, None, True
            
        actual_size = len(data)
        if actual_size < expected_size:
            missing_bytes = expected_size - actual_size
            print(f"Incomplete packet received ({actual_size} bytes). Filling {missing_bytes} missing bytes...")
            
            repetitions = missing_bytes // actual_size + 1
            padding = (data * repetitions)[:missing_bytes]
            data = data + padding
        
        self.previous_packet = data
        self.expected_sequence += 1
        
        return data, is_last_packet, False

    def receive_image(self):
        """Receive complete image"""
        try:
            while True:
                data, is_last_packet, error = self.receive_packet()
                
                if error:
                    print("Error receiving packet")
                    break
                
                if data is not None:
                    self.image_bytes.extend(data)
                
                if is_last_packet:
                    break
            
            # Save the complete image bytes
            with open(self.output_path, 'wb') as f:
                f.write(self.image_bytes)
            
            print(f"Image saved to {self.output_path}")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()

if __name__ == "__main__":
    # Configuration
    HOST = '127.1.0.0'
    PORT = 65432
    OUTPUT_PATH = 'received_image.png'  # Replace with desired output path
    
    # Create and run client
    client = ImageClient(HOST, PORT, OUTPUT_PATH)
    client.connect()
    client.receive_image()