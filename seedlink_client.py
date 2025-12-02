#!/usr/bin/env python3
"""
SeedLink Client Module
Real-time connection to IRIS SeedLink server with auto-reconnection
"""

import logging
import socket
import struct
import time
from threading import Thread, Lock, Event
from typing import Callable, Optional
from datetime import datetime
from io import BytesIO

logger = logging.getLogger(__name__)

# ============================================================================
# SEEDLINK PROTOCOL CONSTANTS
# ============================================================================

SEEDLINK_PROTOCOL = b"SLPROTO:3"
HELLO_MESSAGE = b"HELLO\n"
SELECT_MESSAGE = "SELECT {}\n".encode()
DATA_MESSAGE = b"DATA\n"

# ============================================================================
# SEEDLINK CLIENT
# ============================================================================

class SeedLinkClient:
    """Real-time SeedLink client with auto-reconnection"""
    
    def __init__(self, host: str, port: int, timeout: int = 30):
        """
        Initialize SeedLink client
        
        Args:
            host: SeedLink server hostname
            port: SeedLink server port
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.socket = None
        self.connected = False
        self.running = False
        
        self.lock = Lock()
        self.reconnect_event = Event()
        
        self.packet_callbacks = []  # List of (callback, args) tuples
        self.connection_attempts = 0
        self.successful_connections = 0
        self.packets_received = 0
        self.last_packet_time = None
        
        logger.info(f"🔗 SeedLink client initialized: {host}:{port}")
    
    def register_packet_callback(self, callback: Callable):
        """Register a callback for incoming packets"""
        self.packet_callbacks.append(callback)
        logger.debug(f"Registered packet callback: {callback.__name__}")
    
    def connect(self) -> bool:
        """
        Connect to SeedLink server
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"🔗 Connecting to {self.host}:{self.port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            
            # Receive and parse SeedLink header
            header = self.socket.recv(512)
            if not header.startswith(b"SeedLink"):
                logger.error(f"Invalid SeedLink header: {header[:20]}")
                self.socket.close()
                return False
            
            logger.info(f"✓ Connected: {header.decode('utf-8', errors='ignore').strip()}")
            
            # Send HELLO
            self.socket.sendall(HELLO_MESSAGE)
            
            # Receive server info
            info = self.socket.recv(512)
            logger.debug(f"Server info: {info.decode('utf-8', errors='ignore').strip()}")
            
            with self.lock:
                self.connected = True
                self.successful_connections += 1
                self.connection_attempts = 0
            
            logger.info("✓ SeedLink handshake complete")
            return True
        
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            return False
    
    def subscribe(self, stations: list) -> bool:
        """
        Subscribe to stations
        
        Args:
            stations: List of station codes (e.g., ["IU.ANMO.00.BHZ", "IU.CHTO.00.BHZ"])
            
        Returns:
            True if subscribed successfully
        """
        if not self.connected:
            logger.error("Not connected")
            return False
        
        try:
            for station in stations:
                select_msg = f"SELECT {station}\n"
                self.socket.sendall(select_msg.encode())
                logger.debug(f"Subscribed to {station}")
            
            # End subscriptions
            self.socket.sendall(b"END\n")
            logger.info(f"✓ Subscribed to {len(stations)} stations")
            return True
        
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    def start_data(self) -> bool:
        """Start receiving data stream"""
        if not self.connected:
            logger.error("Not connected")
            return False
        
        try:
            self.socket.sendall(DATA_MESSAGE)
            logger.info("✓ Starting data stream")
            return True
        except Exception as e:
            logger.error(f"Failed to start data: {e}")
            return False
    
    def receive_packets(self, callback: Optional[Callable] = None):
        """
        Receive and process packets (blocking)
        
        Args:
            callback: Optional callback for each packet
        """
        if not self.connected:
            logger.error("Not connected")
            return
        
        logger.info("🔄 Starting packet reception...")
        
        while self.running and self.connected:
            try:
                # Receive packet (512 bytes standard)
                packet = self.socket.recv(512)
                
                if not packet:
                    logger.warning("Connection closed by server")
                    with self.lock:
                        self.connected = False
                    break
                
                with self.lock:
                    self.packets_received += 1
                    self.last_packet_time = datetime.now()
                
                # Process packet
                self._process_packet(packet)
                
                # Call registered callbacks
                for cb in self.packet_callbacks:
                    try:
                        cb(packet)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                if callback:
                    callback(packet)
            
            except socket.timeout:
                logger.debug("Socket timeout (normal)")
            except Exception as e:
                logger.error(f"Packet reception error: {e}")
                with self.lock:
                    self.connected = False
                break
    
    def _process_packet(self, packet: bytes):
        """Parse and process Mini-SEED packet"""
        if len(packet) < 48:
            logger.debug("Packet too small")
            return
        
        try:
            # Mini-SEED header parsing
            sequence = packet[0:6].decode('ascii', errors='ignore')
            record_type = packet[6]
            reserved = packet[7]
            
            # Station/network codes
            net = packet[8:10].decode('ascii', errors='ignore').strip()
            sta = packet[10:12].decode('ascii', errors='ignore').strip()
            loc = packet[12:14].decode('ascii', errors='ignore').strip()
            chan = packet[14:17].decode('ascii', errors='ignore').strip()
            
            logger.debug(f"Packet: {net}.{sta}.{loc}.{chan} seq={sequence}")
        
        except Exception as e:
            logger.debug(f"Error parsing packet header: {e}")
    
    def disconnect(self):
        """Disconnect from server"""
        with self.lock:
            self.running = False
            self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
                logger.info("✓ Disconnected from SeedLink server")
            except:
                pass
    
    def get_statistics(self) -> dict:
        """Get connection statistics"""
        with self.lock:
            return {
                'connected': self.connected,
                'packets_received': self.packets_received,
                'successful_connections': self.successful_connections,
                'connection_attempts': self.connection_attempts,
                'last_packet': self.last_packet_time.isoformat() if self.last_packet_time else None
            }

# ============================================================================
# AUTO-RECONNECTING SEEDLINK LISTENER
# ============================================================================

class SeedLinkListener:
    """SeedLink client with automatic reconnection"""
    
    def __init__(self, host: str, port: int, stations: list, timeout: int = 30):
        """
        Initialize listener with auto-reconnection
        
        Args:
            host: SeedLink server hostname
            port: SeedLink server port
            stations: List of station subscriptions
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.stations = stations
        self.timeout = timeout
        
        self.client = None
        self.running = False
        self.listener_thread = None
        
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_interval = 5  # seconds
        
        logger.info(f"🎧 SeedLink listener initialized: {host}:{port}")
    
    def start(self, callback: Optional[Callable] = None) -> bool:
        """
        Start listening for packets
        
        Args:
            callback: Optional callback for each packet
            
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("Listener already running")
            return False
        
        self.running = True
        self.listener_thread = Thread(target=self._listen_loop, args=(callback,), daemon=True)
        self.listener_thread.start()
        
        logger.info("🎧 Listener started (background thread)")
        return True
    
    def _listen_loop(self, callback: Optional[Callable]):
        """Main listening loop with auto-reconnection"""
        while self.running:
            try:
                # Create client and connect
                self.client = SeedLinkClient(self.host, self.port, self.timeout)
                
                if not self.client.connect():
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                        break
                    
                    logger.info(f"Reconnecting in {self.reconnect_interval}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    time.sleep(self.reconnect_interval)
                    continue
                
                # Subscribe to stations
                if not self.client.subscribe(self.stations):
                    logger.error("Subscription failed")
                    time.sleep(self.reconnect_interval)
                    continue
                
                # Start data stream
                if not self.client.start_data():
                    logger.error("Failed to start data")
                    time.sleep(self.reconnect_interval)
                    continue
                
                # Reset reconnection counter on successful connection
                self.reconnect_attempts = 0
                
                # Receive packets (blocking)
                self.client.receive_packets(callback)
            
            except Exception as e:
                logger.error(f"Listen loop error: {e}")
                time.sleep(self.reconnect_interval)
            
            finally:
                if self.client:
                    self.client.disconnect()
    
    def stop(self):
        """Stop listening"""
        self.running = False
        if self.client:
            self.client.disconnect()
        
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
        
        logger.info("🛑 Listener stopped")
    
    def get_statistics(self) -> dict:
        """Get listener statistics"""
        if self.client:
            return self.client.get_statistics()
        return {'running': self.running, 'connected': False}

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("SEEDLINK CLIENT TEST")
    print("="*70)
    
    # Test configuration
    HOST = "rtserve.iris.washington.edu"
    PORT = 18000
    STATIONS = [
        "IU.ANMO.00.BHZ",
        "IU.CHTO.00.BHZ",
    ]
    
    print(f"\n🎧 Testing SeedLinkListener:")
    print(f"   Host: {HOST}:{PORT}")
    print(f"   Stations: {STATIONS}")
    
    # Create listener
    listener = SeedLinkListener(HOST, PORT, STATIONS, timeout=10)
    
    # Packet counter
    packet_count = [0]
    def packet_callback(packet):
        packet_count[0] += 1
        if packet_count[0] % 100 == 0:
            print(f"   Received {packet_count[0]} packets")
    
    # Start listener
    if listener.start(callback=packet_callback):
        print("   ✓ Listener started in background")
        
        # Let it run for 10 seconds
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        
        # Get statistics
        stats = listener.get_statistics()
        print(f"\n   Statistics:")
        for key, value in stats.items():
            print(f"     {key}: {value}")
    else:
        print("   ✗ Failed to start listener")
    
    # Stop listener
    listener.stop()
    
    print("\n✓ SeedLink client test completed")
    print("="*70 + "\n")
