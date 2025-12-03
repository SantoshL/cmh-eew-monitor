#!/usr/bin/env python3
"""
Real-Time SeedLink Waveform Handler for CMH EEW
Connects to IRIS SeedLink server and processes continuous waveform streams
"""

import numpy as np
from obspy.clients.seedlink.easyseedlink import create_client
from obspy import UTCDateTime
import logging
import threading
import time
from collections import deque
from typing import Dict, Callable, Optional
import pywt

logger = logging.getLogger(__name__)


class CMHStreamProcessor:
    """Process continuous waveform stream with CMH detection"""

    def __init__(self, station_id: str, callback: Callable,
                 window_sec: float = 30.0, buffer_sec: float = 120.0):
        self.station_id = station_id
        self.callback = callback
        self.window_sec = window_sec
        self.buffer_sec = buffer_sec

        # Ring buffer for continuous data
        self.buffer = deque(maxlen=int(buffer_sec * 100))  # Assume 100 Hz
        self.sample_rate = None
        self.background_cmh = None
        self.background_computed = False

        logger.info(f"Initialized processor for {station_id}")

    def add_data(self, trace):
        """Add new data to ring buffer"""
        if self.sample_rate is None:
            self.sample_rate = trace.stats.sampling_rate
            self.buffer = deque(maxlen=int(self.buffer_sec * self.sample_rate))

        # Add samples to buffer
        for sample in trace.data:
            self.buffer.append(sample)

        # Compute background CMH if we have enough data
        if not self.background_computed and len(self.buffer) >= int(30 * self.sample_rate):
            self._compute_background()

        # Check for P-wave if background is ready
        if self.background_computed:
            self._check_pwave()

    def _compute_background(self):
        """Compute background CMH from first 30 seconds"""
        try:
            bg_data = np.array(list(self.buffer)[:int(30 * self.sample_rate)])
            coeffs = pywt.wavedec(bg_data, 'db4', level=2)
            detail = np.concatenate([coeffs[i]
                                    for i in range(1, min(3, len(coeffs)))])
            hist, _ = np.histogram(np.abs(detail), bins=50, density=True)
            self.background_cmh = hist / (np.sum(hist) + 1e-10)
            self.background_computed = True
            logger.info(f"{self.station_id}: Background CMH computed")
        except Exception as e:
            logger.error(f"{self.station_id}: Error computing background: {e}")

    def _check_pwave(self):
        """Check latest window for P-wave using CMH"""
        try:
            # Get latest window
            window_samples = int(self.window_sec * self.sample_rate)
            if len(self.buffer) < window_samples:
                return

            window_data = np.array(list(self.buffer)[-window_samples:])

            # Compute CMH for this window
            coeffs = pywt.wavedec(window_data, 'db4', level=2)
            detail = np.concatenate([coeffs[i]
                                    for i in range(1, min(3, len(coeffs)))])
            hist, _ = np.histogram(np.abs(detail), bins=50, density=True)
            window_cmh = hist / (np.sum(hist) + 1e-10)

            # Compute delta CMH
            delta_cmh = np.linalg.norm(window_cmh - self.background_cmh)

            # Detection threshold
            if delta_cmh > 0.25:  # Stricter threshold to avoid false positives
                # Time-gating: only trigger if no detection in last 30 seconds
                current_time = UTCDateTime.now().timestamp
                if not hasattr(self, 'last_detection_time'):
                    self.last_detection_time = 0

                if current_time - self.last_detection_time > 30:
                    detection = {
                        'station_id': self.station_id,
                        'detection_time': current_time,
                        'delta_cmh': delta_cmh,
                        'confidence': min(delta_cmh / 0.5, 1.0)
                    }
                    self.callback(detection)
                    self.last_detection_time = current_time
                    logger.warning(
                        f"{self.station_id}: P-WAVE DETECTED! ΔCMH={delta_cmh:.3f}")

        except Exception as e:
            logger.error(f"{self.station_id}: Error in P-wave check: {e}")


class SeedLinkManager:
    """Manage SeedLink connections for multiple stations"""

    def __init__(self, detection_callback: Callable):
        self.detection_callback = detection_callback
        self.processors: Dict[str, CMHStreamProcessor] = {}
        self.client = None
        self.running = False
        self.thread = None

    def add_station(self, network: str, station: str, channel: str = "BHZ"):
        """Add a station to monitor"""
        station_id = f"{network}.{station}"

        if station_id not in self.processors:
            processor = CMHStreamProcessor(
                station_id=station_id,
                callback=self.detection_callback
            )
            self.processors[station_id] = processor
            logger.info(f"Added station: {station_id}")

    def on_data(self, trace):
        """Callback when new data arrives"""
        station_id = f"{trace.stats.network}.{trace.stats.station}"

        if station_id in self.processors:
            self.processors[station_id].add_data(trace)

    def start(self, server: str = "rtserve.iris.washington.edu", port: int = 18000):
        """Start SeedLink streaming"""
        def run():
            try:
                logger.info(f"Connecting to SeedLink: {server}:{port}")

                # Create SeedLink client
                self.client = create_client(server, self.on_data)

                # Select streams for each station
                for station_id, processor in self.processors.items():
                    network, station = station_id.split('.')
                    self.client.select_stream(network, station, 'BHZ')
                    logger.info(f"Selected stream: {network}.{station}.BHZ")

                # Start streaming
                self.running = True
                logger.info("Starting SeedLink stream...")
                self.client.run()

            except Exception as e:
                logger.error(f"SeedLink error: {e}")
                self.running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        logger.info("SeedLink manager started")

    def stop(self):
        """Stop SeedLink streaming"""
        self.running = False
        if self.client:
            self.client.close()
        logger.info("SeedLink manager stopped")


# Test function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def test_callback(detection):
        print(f"DETECTION: {detection}")

    manager = SeedLinkManager(test_callback)

    # Add test stations
    # Production stations - 20 global coverage
    stations = [
        ("JA", "KAMAE"), ("JA", "OKW"), ("JA",
                                         "WTNM"), ("JA", "MZGH"), ("JA", "SHIZ"),
        ("CI", "PAS"), ("CI", "CLC"), ("CI", "LRL"), ("CI", "SBC"), ("CI", "SMO"),
        ("BK", "FARB"), ("BK", "YBH"), ("BK",
                                        "MCCM"), ("BK", "SAO"), ("BK", "CMB"),
        ("NC", "A25K"), ("NC", "H04P"), ("NC",
                                         "O22K"), ("NC", "Y27K"), ("NC", "Z24K"),
    ]

    for network, station in stations:
        manager.add_station(network, station)

    manager.start()

    print("Monitoring stations... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        manager.stop()
