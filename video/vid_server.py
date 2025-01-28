import socket
import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue, Empty
import logging

CONST_FEC_BLOCK_SIZE = 21       # LDPC code block size (336 bits at output, fec rate 1/2)
CONST_CP = 0b001                # Cyclic prefix (CONST_CP * 8)
CONST_BAT_ID = 0b00010          # Bits per subcarrier
CONST_SI = 0b1111               # Scrambler initialization
CONST_FEC_RATE = 0b001          # UNUSED
CONST_BLOCK_SIZE = 0b00         # UNUSED

class VideoServer:
    def __init__(self, host, port, video_path, packet_size=4011, delay_per_package=2e-3, compression=40, display_video=True):
        self.host = host                            # Host address
        self.port = port                            # Socket port
        self.video_path = video_path                # Path to video file
        self.packet_size = packet_size              # [bytes] Size of the package
        self.delay_per_package = delay_per_package  # [seg] Amount of time to wait between packages sent
        self.compression = compression              # [%] Compression rate for the video
        self.display_video = display_video

        self.send_queue = Queue()       # Queue with packets to be sent
        self.display_queue = Queue(maxsize=500)    # Queue with frames for video visualization

        # Used to print metrics
        self.total_bytes_sent = 0
        self.frames_sent = 0
        logging.basicConfig(filename='./video/server_log.log',level=logging.INFO, filemode="w")
        self.logger = logging.getLogger('VideoServer')

    def prepare_packet(self, data:np.ndarray[np.uint8], packet_number, packets_in_frame):
        """Prepare packet with registers and data"""

        # Append zeros to message to make it a multiple of FEC_BLOCK_SIZE
        payload_len_in_fec_blocks = np.uint32(np.ceil(len(data)/CONST_FEC_BLOCK_SIZE))
        payload_len_in_words = payload_len_in_fec_blocks * CONST_FEC_BLOCK_SIZE
        payload_extra_words = payload_len_in_words - len(data)

        payload_out = np.zeros(payload_len_in_words, np.uint8)
        payload_out[0:len(data)] = data

        # Form registers
        regs = np.zeros(4, np.uint32)
        regs[0] = payload_len_in_words
        regs[1] = payload_extra_words

        # Replaced FEC Concatenation Factor and repetition number with "packets_in_frame"
        regs[2] = (((packets_in_frame >> 3) & 0b111 ) << 24) | ((packets_in_frame & 0b111) << 16) | (((self.fps >> 2) & 0b111) << 8) | ((self.fps & 0b11) << 0)

        # Replaced MIMO with packet number
        regs[3] = (packet_number << 24) | (CONST_CP << 16) | (CONST_BAT_ID << 8) | (CONST_SI << 0)

        return payload_out, regs

    def setup_server(self):
        """Initialize server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1000)
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

        self.fps = np.uint32(self.cap.get(cv2.CAP_PROP_FPS))
        self.time_per_frame = 1/self.fps
        self.logger.info(f"Video FPS: {self.fps}")

    def frame_reader(self):
        """Read frames from video file and put them in queue"""
        frame_counter = 0
        ret, frame = self.cap.read()
        while ret:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.compression])
            self.send_queue.put(buffer)
            if (self.display_video):
                self.display_queue.put(frame)
            frame_counter += 1
            ret, frame = self.cap.read()

        self.logger.info(f"Reached end of video at frame: {frame_counter}")

    def display_frames(self):
        """Display frames"""
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
                self.logger.info("No more frames to display, closing window...")
                break

            cv2.imshow('Video Server', frame)
            previous_frame_time = time.time()

            # Necessary for display window to work
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(self.display_queue.qsize())

    def send_frame(self) -> bool:
        """Split frame into packets, calculate registers and send"""
        try:
            frame_data = np.array(self.send_queue.get(timeout=10), dtype=np.uint8)
        except Empty:
            self.logger.info("No more data in video, exiting...")
            return False

        packets_in_frame = np.uint32(np.ceil(len(frame_data) / self.packet_size))

        for packet_number in range(0, packets_in_frame):
            packet_data = frame_data[packet_number*self.packet_size : (packet_number+1)*self.packet_size]

            packet, regs = self.prepare_packet(packet_data, packet_number, packets_in_frame)

            # Send packet through socket
            self.client_socket.sendall(regs)
            self.client_socket.sendall(packet)
            time.sleep(self.delay_per_package)

            self.total_bytes_sent += len(packet)

        self.frames_sent += 1
        return True

    def print_metrics(self):
        """Print transmission metrics"""
        bitrate = (self.total_bytes_sent * 8) / (time.time() - self.start_time)
        print(f"\rProgress: {self.total_bytes_sent*1e-6:.2f} MB sent, "
              f"Bitrate: {bitrate*1e-6:.2f} Mbps, "
              f"Frames sent: {self.frames_sent}", end="")

    def run(self):
        """Main server loop"""
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
            raise e

        reader_thread = Thread(target=self.frame_reader)        # Use threads to generate and display frames
        reader_thread.start()
        if (self.display_video):
            display_thread = Thread(target=self.display_frames)
            display_thread.start()

        self.start_time = time.time()
        while self.send_frame():
            self.print_metrics()

        self.logger.info("Waiting to join threads...")

        reader_thread.join()
        if (self.display_video):
            display_thread.join()

    def __del__(self):
        self.client_socket.close()
        self.server_socket.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    HOST = '127.1.0.0'
    PORT = 65432
    #VIDEO_PATH = './video/CiroyLosPersas.mp4'       # Replace with video path
    VIDEO_PATH = "./video/SampleVideo_1280x720_10mb.mp4"
    #VIDEO_PATH = "./video/1_hour_timer.webm"
    PACKET_SIZE = 4011

    server = VideoServer(HOST, PORT, VIDEO_PATH, PACKET_SIZE, delay_per_package=1e-3, compression=40, display_video=True)
    server.run()