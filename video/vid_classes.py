import numpy as np

class VideoPacketHandler:
    """Base class with common packet handling logic"""

    @staticmethod
    def split_packet_number(packet_num):
        """Split 11-bit packet number into two ranges for reg2"""
        upper_bits = (packet_num >> 6) & 0x1F  # 5 bits for [15:11]
        lower_bits = packet_num & 0x3F         # 6 bits for [7:2]
        return upper_bits, lower_bits

    @staticmethod
    def combine_packet_number(upper_bits, lower_bits):
        """Combine split packet number back into 11-bit value"""
        return ((upper_bits & 0x1F) << 6) | (lower_bits & 0x3F)

class VideoPacketEncoder(VideoPacketHandler):
    """Handles packet creation and register generation for video frames"""

    def __init__(self, placeholder_value=65792):
        self.placeholder_value = placeholder_value
        self.frame_number = 0
        self.packet_number = 0

    def calculate_reg0(self, packet_size):
        """Calculate number of bytes in packet (24-bits)"""
        return np.uint32(packet_size & 0xFFFFFF)

    def calculate_reg1(self, packet_size):
        """Calculate reg1 with frame number and padding
        Frame number in upper 16 bits, padding in lower 16 bits
        """
        padding = (21 - (packet_size % 21)) % 21
        return np.uint32((self.frame_number << 16) | padding)

    def calculate_reg2(self, is_last_packet):
        """Calculate reg2 with packet number and last packet flag w/ fixed value"""

        upper_bits, lower_bits = self.split_packet_number(self.packet_number)

        reg2 = self.placeholder_value

        reg2 |= (upper_bits << 11)  # Bits [15:11]
        reg2 |= (lower_bits << 2)   # Bits [7:2]

        if is_last_packet:
            reg2 |= 0xA0000000

        return np.uint32(reg2)

    def calculate_reg3(self):
        """Placeholder for reg3"""
        return np.uint32(66063)  # Fixed value

    def prepare_packet(self, data, is_last_packet):
        """Prepare packet with registers and data"""
        packet_size = len(data)

        registers = np.array([
            self.calculate_reg0(packet_size),
            self.calculate_reg1(packet_size),
            self.calculate_reg2(is_last_packet),
            self.calculate_reg3()
        ], dtype=np.uint32)

        self.packet_number += 1
        return registers.tobytes(), data

    def next_frame(self):
        """Advance to next frame"""
        self.frame_number = (self.frame_number + 1) & 0xFFFF
        self.packet_number = 0

class VideoPacketDecoder(VideoPacketHandler):
    """Handles packet decoding and frame reconstruction"""

    def __init__(self):
        self.current_frame = 0
        self.expected_packet = 0
        self.frame_packets = {}

    def decode_registers(self, register_data):
        """Decode register values from received data"""
        registers = np.frombuffer(register_data, dtype=np.uint32)

        packet_size = registers[0] & 0xFFFFFF
        frame_number = registers[1] >> 16
        padding = registers[1] & 0xFFFF

        upper_bits = (registers[2] >> 11) & 0x1F
        lower_bits = (registers[2] >> 2) & 0x3F
        packet_number = self.combine_packet_number(upper_bits, lower_bits)

        is_last_packet = (registers[2] & 0xF0000000) == 0xA0000000

        return {
            'packet_size': packet_size,
            'frame_number': frame_number,
            'padding': padding,
            'packet_number': packet_number,
            'is_last_packet': is_last_packet
        }

    def check_frame_wraparound(self, new_frame):
        """Check for frame number wraparound"""
        if self.current_frame >= 0xF000 and new_frame < 0x1000:     #May change range
            return True
        return False

    def handle_missing_packets(self):
        """Fill missing packets with nearest valid packets"""
        if not self.frame_packets:
            print('Frame packets is null in missing_packets')
            return None

        max_packet = max(self.frame_packets.keys())
        result = bytearray()

        for i in range(max_packet + 1):
            if i not in self.frame_packets:
                distances = [(abs(p - i), p) for p in self.frame_packets.keys()]
                _, nearest = min(distances)
                result.extend(self.frame_packets[nearest])
            else:
                result.extend(self.frame_packets[i])

        return result

    def is_frame_complete(self, last_packet_number):
        """Check if all packets up to last_packet_number"""
        return all(i in self.frame_packets for i in range(last_packet_number + 1))