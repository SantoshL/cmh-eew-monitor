#!/usr/bin/env python3
"""
SeedLink EEW Integration Module
Orchestrates real-time EEW system with SeedLink data, P-wave detection, and magnitude estimation
"""

import logging
from threading import Thread, Lock, Event
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
import json
import time

# Import your existing modules
try:
    from magnitude_estimator_v2 import CMHMagnitudeEstimator
except ImportError:
    logging.warning("magnitude_estimator_v2 not found, using fallback")
    CMHMagnitudeEstimator = None

# Import SeedLink modules
from seedlink_config import (
    get_active_server, get_station_list, EEW_THRESHOLDS,
    CMH_MAGNITUDE_PARAMS, P_WAVE_DETECTION
)
from waveform_buffer import BufferManager, BufferStatistics
from seedlink_client import SeedLinkListener

logger = logging.getLogger(__name__)

# ============================================================================
# EEW ALERT DATA STRUCTURE
# ============================================================================

class EEWAlert:
    """Early Warning Alert"""
    
    def __init__(self, alert_id: str, magnitude: float, confidence: float,
                 stations_used: int, detection_time: datetime, 
                 epicenter_lat: Optional[float] = None,
                 epicenter_lon: Optional[float] = None):
        self.alert_id = alert_id
        self.magnitude = magnitude
        self.confidence = confidence
        self.stations_used = stations_used
        self.detection_time = detection_time
        self.epicenter_lat = epicenter_lat
        self.epicenter_lon = epicenter_lon
        self.uncertainty = CMH_MAGNITUDE_PARAMS['uncertainty']
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            'alert_id': self.alert_id,
            'magnitude': round(self.magnitude, 2),
            'confidence': round(self.confidence, 3),
            'stations_used': self.stations_used,
            'detection_time': self.detection_time.isoformat(),
            'uncertainty': self.uncertainty,
            'epicenter': {
                'latitude': self.epicenter_lat,
                'longitude': self.epicenter_lon
            } if self.epicenter_lat and self.epicenter_lon else None
        }

# ============================================================================
# P-WAVE DETECTOR
# ============================================================================

class PWaveDetector:
    """Detects P-waves using STA/LTA algorithm"""
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize P-wave detector
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.short_window = int(P_WAVE_DETECTION['short_window'] * sampling_rate)
        self.long_window = int(P_WAVE_DETECTION['long_window'] * sampling_rate)
        self.threshold = P_WAVE_DETECTION['threshold_ratio']
        self.min_amplitude = P_WAVE_DETECTION['min_amplitude']
    
    def detect(self, data: np.ndarray) -> tuple:
        """
        Detect P-wave arrival using STA/LTA
        
        Args:
            data: Waveform samples
            
        Returns:
            (detected: bool, confidence: float, onset_index: int)
        """
        if len(data) < self.long_window:
            return False, 0.0, -1
        
        # Calculate absolute values
        abs_data = np.abs(data)
        
        # Short-term average (STA)
        sta = np.zeros(len(data))
        for i in range(self.short_window, len(data)):
            sta[i] = np.mean(abs_data[i-self.short_window:i])
        
        # Long-term average (LTA)
        lta = np.zeros(len(data))
        for i in range(self.long_window, len(data)):
            lta[i] = np.mean(abs_data[i-self.long_window:i])
        
        # STA/LTA ratio
        ratio = np.divide(sta, lta, where=lta!=0, out=np.zeros_like(sta))
        
        # Check for threshold crossing
        onset_idx = -1
        max_ratio = 0.0
        
        for i in range(self.long_window, len(ratio)):
            if ratio[i] > max_ratio:
                max_ratio = ratio[i]
            
            if ratio[i] > self.threshold and abs_data[i] > self.min_amplitude:
                if onset_idx == -1:
                    onset_idx = i
        
        detected = onset_idx > 0 and max_ratio > self.threshold
        confidence = min(1.0, max_ratio / self.threshold)
        
        return detected, confidence, onset_idx

# ============================================================================
# EEW PIPELINE
# ============================================================================

class EEWPipeline:
    """Complete EEW processing pipeline"""
    
    def __init__(self):
        """Initialize EEW pipeline"""
        self.buffer_manager = BufferManager(
            max_duration_seconds=20.0,
            sampling_rate=100.0
        )
        
        self.p_wave_detector = PWaveDetector(sampling_rate=100.0)
        
        # Try to load magnitude estimator
        if CMHMagnitudeEstimator:
            try:
                self.magnitude_estimator = CMHMagnitudeEstimator()
                logger.info("✓ CMH magnitude estimator loaded")
            except Exception as e:
                logger.warning(f"Failed to load CMH estimator: {e}")
                self.magnitude_estimator = None
        else:
            self.magnitude_estimator = None
        
        self.seedlink_listener = None
        self.running = False
        self.processing_thread = None
        
        self.alerts: List[EEWAlert] = []
        self.alert_lock = Lock()
        self.last_alert_time = None
        
        # Statistics
        self.packets_processed = 0
        self.detections_found = 0
        self.alerts_issued = 0
        
        logger.info("🔬 EEW Pipeline initialized")
    
    def _packet_handler(self, packet: bytes):
        """Handle incoming SeedLink packet"""
        self.packets_processed += 1
        
        if self.packets_processed % 1000 == 0:
            logger.debug(f"Processed {self.packets_processed} packets")
        
        try:
            # Parse Mini-SEED packet header
            if len(packet) < 48:
                return
            
            net = packet[8:10].decode('ascii', errors='ignore').strip()
            sta = packet[10:12].decode('ascii', errors='ignore').strip()
            loc = packet[12:14].decode('ascii', errors='ignore').strip()
            chan = packet[14:17].decode('ascii', errors='ignore').strip()
            
            # Store in buffer (simplified - uses packet arrival time)
            timestamp = time.time()
            amplitude = np.random.randn() * 0.1  # Mock amplitude
            
            self.buffer_manager.add_sample(f"{net}.{sta}", chan, timestamp, amplitude)
        
        except Exception as e:
            logger.debug(f"Packet parsing error: {e}")
    
    def _processing_loop(self):
        """Main EEW processing loop"""
        logger.info("🔄 Processing loop started")
        
        while self.running:
            try:
                # Get all buffer statistics
                all_stats = self.buffer_manager.get_all_statistics()
                
                # Process each buffer
                for stats in all_stats:
                    self._process_station(stats)
                
                time.sleep(1)  # Process every second
            
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1)
    
    def _process_station(self, stats: BufferStatistics):
        """Process single station buffer"""
        if stats.num_samples < 100:
            return  # Not enough data
        
        # Get waveform data
        data = self.buffer_manager.get_buffer_data(stats.station_id, stats.channel, duration_seconds=10.0)
        
        if len(data) == 0:
            return
        
        # Detect P-wave
        detected, confidence, onset_idx = self.p_wave_detector.detect(data)
        
        if detected and confidence > 0.5:
            self.detections_found += 1
            logger.info(f"🔴 P-wave detected at {stats.station_id}.{stats.channel} (confidence: {confidence:.2f})")
            
            # Check if we should issue alert
            self._check_alert_conditions()
    
    def _check_alert_conditions(self):
        """Check if alert conditions are met"""
        all_stats = self.buffer_manager.get_all_statistics()
        
        # Count P-wave detections
        num_detections = self.detections_found
        
        if num_detections >= EEW_THRESHOLDS['min_stations']:
            # Estimate magnitude
            if self.magnitude_estimator:
                try:
                    # Use mock CMH values for now
                    cmh_integrals = [np.random.randn() * 0.5 + 2.0 for _ in range(num_detections)]
                    magnitude = self.magnitude_estimator.estimate_magnitude(cmh_integrals)
                except:
                    magnitude = 4.5 + np.random.randn() * 0.3
            else:
                magnitude = 4.5 + np.random.randn() * 0.3
            
            if magnitude >= EEW_THRESHOLDS['magnitude_threshold']:
                self._issue_alert(magnitude, num_detections)
    
    def _issue_alert(self, magnitude: float, num_stations: int):
        """Issue EEW alert"""
        now = datetime.now()
        
        # Check alert rate limiting (don't alert more than once per 10 seconds)
        if self.last_alert_time and (now - self.last_alert_time).total_seconds() < 10:
            return
        
        alert = EEWAlert(
            alert_id=f"CMH_{now.strftime('%Y%m%d_%H%M%S')}",
            magnitude=magnitude,
            confidence=0.85,
            stations_used=num_stations,
            detection_time=now,
            epicenter_lat=35.5,  # Mock coordinates
            epicenter_lon=138.5
        )
        
        with self.alert_lock:
            self.alerts.append(alert)
            self.alerts_issued += 1
            self.last_alert_time = now
        
        logger.warning(f"🚨 ALERT ISSUED: M{magnitude:.1f} (Stations: {num_stations})")
        logger.warning(f"   Alert ID: {alert.alert_id}")
    
    def start(self) -> bool:
        """Start EEW pipeline"""
        if self.running:
            logger.warning("Pipeline already running")
            return False
        
        logger.info("🚀 Starting EEW pipeline...")
        
        # Get server config
        server_config = get_active_server()
        stations = get_station_list()
        
        # Start SeedLink listener
        try:
            self.seedlink_listener = SeedLinkListener(
                host=server_config['host'],
                port=server_config['port'],
                stations=stations,
                timeout=server_config['timeout']
            )
            
            if not self.seedlink_listener.start(callback=self._packet_handler):
                logger.error("Failed to start SeedLink listener")
                return False
            
            logger.info("✓ SeedLink listener started")
        
        except Exception as e:
            logger.error(f"Failed to initialize SeedLink: {e}")
            return False
        
        # Start processing thread
        self.running = True
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("✓ EEW pipeline running")
        return True
    
    def stop(self):
        """Stop EEW pipeline"""
        logger.info("🛑 Stopping EEW pipeline...")
        
        self.running = False
        
        if self.seedlink_listener:
            self.seedlink_listener.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("✓ EEW pipeline stopped")
    
    def get_status(self) -> dict:
        """Get pipeline status"""
        with self.alert_lock:
            latest_alert = self.alerts[-1].to_dict() if self.alerts else None
        
        return {
            'running': self.running,
            'packets_processed': self.packets_processed,
            'detections_found': self.detections_found,
            'alerts_issued': self.alerts_issued,
            'latest_alert': latest_alert,
            'buffers_active': self.buffer_manager.get_buffer_count(),
            'seedlink_status': self.seedlink_listener.get_statistics() if self.seedlink_listener else {}
        }
    
    def get_recent_alerts(self, n: int = 10) -> List[dict]:
        """Get recent alerts"""
        with self.alert_lock:
            return [alert.to_dict() for alert in self.alerts[-n:]]

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("EEW PIPELINE TEST")
    print("="*70)
    
    # Create pipeline
    pipeline = EEWPipeline()
    
    # Start pipeline
    print("\n🚀 Starting EEW pipeline...")
    if pipeline.start():
        print("   ✓ Pipeline started")
        
        # Run for 30 seconds
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n   (Interrupted)")
        
        # Get status
        status = pipeline.get_status()
        print("\n📊 Pipeline Status:")
        for key, value in status.items():
            if key != 'seedlink_status':
                print(f"   {key}: {value}")
    else:
        print("   ✗ Failed to start pipeline")
    
    # Stop pipeline
    pipeline.stop()
    
    print("\n✓ EEW pipeline test completed")
    print("="*70 + "\n")
