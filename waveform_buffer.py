#!/usr/bin/env python3
"""
Waveform Buffer Module
Thread-safe circular buffers for real-time seismic data storage
"""

import logging
from collections import deque
from threading import Lock
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WaveformSample:
    """Single waveform sample with metadata"""
    timestamp: float           # UTC timestamp (seconds)
    amplitude: float           # Waveform amplitude
    station_id: str           # Station identifier
    channel: str              # Channel (BHZ, BHE, BHN, etc.)
    sampling_rate: float      # Samples per second

@dataclass
class BufferStatistics:
    """Statistics for a buffer"""
    station_id: str
    channel: str
    num_samples: int          # Total samples in buffer
    duration_seconds: float   # Duration of data in buffer
    min_amplitude: float
    max_amplitude: float
    mean_amplitude: float
    gaps_detected: int        # Number of data gaps
    last_update: datetime = field(default_factory=datetime.now)

# ============================================================================
# CIRCULAR BUFFER
# ============================================================================

class CircularBuffer:
    """Thread-safe circular buffer for waveform data"""
    
    def __init__(self, station_id: str, channel: str, max_duration_seconds: float = 10.0, sampling_rate: float = 100.0):
        """
        Initialize circular buffer
        
        Args:
            station_id: e.g., "IU.ANMO"
            channel: e.g., "BHZ"
            max_duration_seconds: How long to keep data (seconds)
            sampling_rate: Sampling rate in Hz
        """
        self.station_id = station_id
        self.channel = channel
        self.max_duration = max_duration_seconds
        self.sampling_rate = sampling_rate
        self.max_samples = int(max_duration_seconds * sampling_rate)
        
        # Thread-safe storage
        self.lock = Lock()
        self.samples = deque(maxlen=self.max_samples)
        self.last_timestamp = None
        self.gap_count = 0
        
        logger.info(f"📦 Buffer created: {station_id}.{channel} ({self.max_samples} samples, {max_duration_seconds}s)")
    
    def add_sample(self, timestamp: float, amplitude: float) -> bool:
        """
        Add a waveform sample to buffer
        
        Args:
            timestamp: UTC timestamp in seconds
            amplitude: Waveform amplitude value
            
        Returns:
            True if added successfully, False if out of order
        """
        with self.lock:
            # Check for data gaps or out-of-order timestamps
            if self.last_timestamp is not None:
                expected_delta = 1.0 / self.sampling_rate
                actual_delta = timestamp - self.last_timestamp
                
                if actual_delta < 0:
                    logger.warning(f"⚠️  Out-of-order timestamp: {self.station_id}.{self.channel}")
                    return False
                
                if actual_delta > expected_delta * 1.5:  # Allow 50% variation
                    self.gap_count += 1
                    logger.debug(f"🔴 Gap detected in {self.station_id}.{self.channel} (gap #{self.gap_count})")
            
            sample = WaveformSample(
                timestamp=timestamp,
                amplitude=amplitude,
                station_id=self.station_id,
                channel=self.channel,
                sampling_rate=self.sampling_rate
            )
            
            self.samples.append(sample)
            self.last_timestamp = timestamp
            return True
    
    def get_data(self, duration_seconds: Optional[float] = None) -> np.ndarray:
        """
        Get waveform data as numpy array
        
        Args:
            duration_seconds: Get last N seconds (or all if None)
            
        Returns:
            Numpy array of amplitudes
        """
        with self.lock:
            if not self.samples:
                return np.array([])
            
            if duration_seconds is None:
                data = [s.amplitude for s in self.samples]
            else:
                cutoff_time = self.last_timestamp - duration_seconds
                data = [s.amplitude for s in self.samples if s.timestamp >= cutoff_time]
            
            return np.array(data, dtype=np.float32)
    
    def get_timestamps(self) -> np.ndarray:
        """Get timestamps of all samples"""
        with self.lock:
            return np.array([s.timestamp for s in self.samples], dtype=np.float64)
    
    def get_statistics(self) -> BufferStatistics:
        """Get buffer statistics"""
        with self.lock:
            if not self.samples:
                return BufferStatistics(
                    station_id=self.station_id,
                    channel=self.channel,
                    num_samples=0,
                    duration_seconds=0.0,
                    min_amplitude=0.0,
                    max_amplitude=0.0,
                    mean_amplitude=0.0,
                    gaps_detected=0
                )
            
            amplitudes = np.array([s.amplitude for s in self.samples])
            duration = (self.last_timestamp - self.samples[0].timestamp) if len(self.samples) > 1 else 0.0
            
            return BufferStatistics(
                station_id=self.station_id,
                channel=self.channel,
                num_samples=len(self.samples),
                duration_seconds=duration,
                min_amplitude=float(np.min(amplitudes)),
                max_amplitude=float(np.max(amplitudes)),
                mean_amplitude=float(np.mean(amplitudes)),
                gaps_detected=self.gap_count
            )
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self.lock:
            return len(self.samples) >= self.max_samples
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.samples.clear()
            self.last_timestamp = None
            self.gap_count = 0

# ============================================================================
# MULTI-BUFFER MANAGER
# ============================================================================

class BufferManager:
    """Manages multiple circular buffers for different stations/channels"""
    
    def __init__(self, max_duration_seconds: float = 10.0, sampling_rate: float = 100.0):
        """
        Initialize buffer manager
        
        Args:
            max_duration_seconds: Duration to keep per buffer
            sampling_rate: Sampling rate in Hz
        """
        self.max_duration = max_duration_seconds
        self.sampling_rate = sampling_rate
        self.buffers: Dict[str, CircularBuffer] = {}
        self.lock = Lock()
        
        logger.info(f"📦 BufferManager initialized ({max_duration_seconds}s, {sampling_rate}Hz)")
    
    def get_or_create_buffer(self, station_id: str, channel: str) -> CircularBuffer:
        """Get existing buffer or create new one"""
        key = f"{station_id}.{channel}"
        
        with self.lock:
            if key not in self.buffers:
                self.buffers[key] = CircularBuffer(
                    station_id=station_id,
                    channel=channel,
                    max_duration_seconds=self.max_duration,
                    sampling_rate=self.sampling_rate
                )
            
            return self.buffers[key]
    
    def add_sample(self, station_id: str, channel: str, timestamp: float, amplitude: float) -> bool:
        """Add sample to appropriate buffer"""
        buffer = self.get_or_create_buffer(station_id, channel)
        return buffer.add_sample(timestamp, amplitude)
    
    def get_buffer_data(self, station_id: str, channel: str, duration_seconds: Optional[float] = None) -> np.ndarray:
        """Get data from specific buffer"""
        key = f"{station_id}.{channel}"
        
        with self.lock:
            if key not in self.buffers:
                return np.array([])
            
            return self.buffers[key].get_data(duration_seconds)
    
    def get_all_statistics(self) -> List[BufferStatistics]:
        """Get statistics from all buffers"""
        with self.lock:
            stats = [buffer.get_statistics() for buffer in self.buffers.values()]
        return stats
    
    def get_buffer_count(self) -> int:
        """Get number of active buffers"""
        with self.lock:
            return len(self.buffers)
    
    def clear_all(self):
        """Clear all buffers"""
        with self.lock:
            for buffer in self.buffers.values():
                buffer.clear()
            self.buffers.clear()

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("WAVEFORM BUFFER TEST")
    print("="*70)
    
    # Test single buffer
    print("\n1️⃣  Testing CircularBuffer:")
    buffer = CircularBuffer("IU.ANMO", "BHZ", max_duration_seconds=5.0, sampling_rate=100.0)
    
    # Add samples
    import time
    start_time = time.time()
    for i in range(100):
        timestamp = start_time + i / 100.0
        amplitude = np.sin(2 * np.pi * 1.0 * i / 100.0)  # 1 Hz sine wave
        buffer.add_sample(timestamp, amplitude)
    
    stats = buffer.get_statistics()
    print(f"   Samples: {stats.num_samples}")
    print(f"   Duration: {stats.duration_seconds:.2f}s")
    print(f"   Amplitude range: [{stats.min_amplitude:.3f}, {stats.max_amplitude:.3f}]")
    print(f"   Mean: {stats.mean_amplitude:.3f}")
    print(f"   Gaps: {stats.gaps_detected}")
    
    # Test buffer manager
    print("\n2️⃣  Testing BufferManager:")
    manager = BufferManager(max_duration_seconds=10.0, sampling_rate=100.0)
    
    # Add samples to multiple buffers
    for i in range(50):
        timestamp = start_time + i / 100.0
        
        # Station 1
        amp1 = np.sin(2 * np.pi * 0.5 * i / 100.0)
        manager.add_sample("IU.ANMO", "BHZ", timestamp, amp1)
        
        # Station 2
        amp2 = np.cos(2 * np.pi * 0.7 * i / 100.0)
        manager.add_sample("IU.CHTO", "BHZ", timestamp, amp2)
    
    print(f"   Active buffers: {manager.get_buffer_count()}")
    
    all_stats = manager.get_all_statistics()
    for stat in all_stats:
        print(f"   {stat.station_id}.{stat.channel}: {stat.num_samples} samples")
    
    print("\n✓ Buffer tests completed successfully")
    print("="*70 + "\n")
