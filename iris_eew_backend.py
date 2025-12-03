#!/usr/bin/env python3

"""
IRIS CMH EARTHQUAKE EARLY WARNING SYSTEM - Backend v2.0
Enhanced with Railway Volume support and research data export

NEW FEATURES:
• Railway Volume persistent storage support
• File naming convention: yymmdd-erthqk-type-vX.ext
• Data download endpoint for research
• Better startup logging
• Error handling improvements
===============================================================================
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum
import threading
import time
import os
import requests
from pathlib import Path
import zipfile
import io

# Import ObsPy for IRIS waveform retrieval
from obspy.clients.fdsn import Client as IRISClient
from seedlink_handler import SeedLinkManager
from seedlink_handler import SeedLinkManager
from obspy import UTCDateTime
# Import SeedLink EEW Pipeline
from performance_charts import PerformanceCharts  #
from seedlink_eew_integration import EEWPipeline
# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__, static_folder='.')
CORS(app)

# Keep-alive function to prevent Railway serverless sleep


def keep_alive_ping():
    """Ping external service every 8 minutes to prevent Railway sleep"""
    while True:
        try:
            time.sleep(480)  # 8 minutes
            requests.get('https://www.google.com', timeout=5)
            print("✅ Keep-alive ping sent to prevent sleep")
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")


# Start keep-alive thread
threading.Thread(target=keep_alive_ping, daemon=True).start()
print("🔄 Keep-alive thread started")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory - supports Railway Volume
DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
DATA_DIR.mkdir(exist_ok=True)

logger.info(f"📁 Data directory: {DATA_DIR.absolute()}")

# Global state
latest_alert = None
alert_history = []
eew_engine = None
monitoring_active = False
# SeedLink EEW Pipeline
seedlink_pipeline = None
seedlink_running = False

# ============================================================================
# FILE NAMING CONVENTION: yymmdd-erthqk-type-vX.ext
# ============================================================================


def get_filename(file_type, version=1, extension='json'):
    """
    Generate filename with your convention: yymmdd-erthqk-type-vX.ext

    Args:
        file_type: usgs, iris, stats, export, model
        version: version number (default 1)
        extension: file extension (default json)

    Returns:
        Formatted filename string
    """
    date_str = datetime.now().strftime('%y%m%d')
    return f"{date_str}-erthqk-{file_type}-v{version}.{extension}"

# ============================================================================
# DATA LOGGING FUNCTIONS (ENHANCED)
# ============================================================================


def log_earthquake_event(event, source='USGS'):
    """
    Log detected earthquake to daily JSON file
    Uses naming convention: yymmdd-erthqk-usgs-v1.json
    """
    try:
        log_file = DATA_DIR / get_filename('usgs', version=1, extension='json')

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'event': event,
            'system_status': {
                'active_stations': len(eew_engine.eew_system.detectors) if eew_engine else 0,
                'total_alerts': len(alert_history)
            }
        }

        # Load existing logs
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)

        # Append new entry
        logs.append(log_entry)

        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        logger.info(
            f"📝 Logged: {event.get('alert_id', 'unknown')} → {log_file.name}")

    except Exception as e:
        logger.error(f"Error logging earthquake event: {e}")


def log_iris_waveform_fetch(event_id, network, station, channel, success, error=None, n_samples=0, sampling_rate=0):
    """
    Log all IRIS waveform retrieval attempts
    Uses naming convention: yymmdd-erthqk-iris-v1.json
    """
    try:
        log_file = DATA_DIR / get_filename('iris', version=1, extension='json')

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_id': event_id,
            'network': network,
            'station': station,
            'channel': channel,
            'success': success,
            'error': error,
            'data_summary': {
                'n_samples': n_samples,
                'sampling_rate': sampling_rate
            } if success else None
        }

        # Load existing logs
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)

        # Append new entry
        logs.append(log_entry)

        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        status = "✓" if success else "✗"
        logger.info(f"📝 {status} IRIS fetch logged: {network}.{station}")

    except Exception as e:
        logger.error(f"Error logging IRIS waveform fetch: {e}")


def generate_daily_stats():
    """
    Generate daily statistics summary
    Uses naming convention: yymmdd-erthqk-stats-v1.csv
    """
    try:
        stats_file = DATA_DIR / \
            get_filename('stats', version=1, extension='csv')

        # Get today's earthquake log
        usgs_file = DATA_DIR / \
            get_filename('usgs', version=1, extension='json')
        iris_file = DATA_DIR / \
            get_filename('iris', version=1, extension='json')

        stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_earthquakes': 0,
            'total_iris_fetches': 0,
            'iris_success_rate': 0.0,
            'max_magnitude': 0.0,
            'min_magnitude': 10.0,
            'avg_magnitude': 0.0
        }

        # Count earthquakes
        if usgs_file.exists():
            with open(usgs_file, 'r') as f:
                logs = json.load(f)
                stats['total_earthquakes'] = len(logs)
                if logs:
                    mags = [log['event']['magnitude'] for log in logs]
                    stats['max_magnitude'] = max(mags)
                    stats['min_magnitude'] = min(mags)
                    stats['avg_magnitude'] = sum(mags) / len(mags)

        # Count IRIS fetches
        if iris_file.exists():
            with open(iris_file, 'r') as f:
                logs = json.load(f)
                stats['total_iris_fetches'] = len(logs)
                successful = sum(1 for log in logs if log.get('success'))
                if logs:
                    stats['iris_success_rate'] = successful / len(logs)

        # Write CSV
        with open(stats_file, 'w') as f:
            f.write(','.join(stats.keys()) + '\n')
            f.write(','.join(str(v) for v in stats.values()) + '\n')

        logger.info(f"📊 Daily stats generated: {stats_file.name}")
        return stats

    except Exception as e:
        logger.error(f"Error generating daily stats: {e}")
        return {}


def get_log_summary():
    """
    Generate summary statistics from all logs
    """
    try:
        summary = {
            'total_earthquakes_logged': 0,
            'total_iris_fetches': 0,
            'iris_success_rate': 0.0,
            'date_range': {'start': None, 'end': None},
            'data_files': []
        }

        # Count all earthquake logs
        earthquake_logs = list(DATA_DIR.glob('*-erthqk-usgs-*.json'))
        for log_file in earthquake_logs:
            summary['data_files'].append(log_file.name)
            with open(log_file, 'r') as f:
                logs = json.load(f)
                summary['total_earthquakes_logged'] += len(logs)

        # Count all IRIS logs
        iris_logs = list(DATA_DIR.glob('*-erthqk-iris-*.json'))
        for iris_log in iris_logs:
            with open(iris_log, 'r') as f:
                logs = json.load(f)
                summary['total_iris_fetches'] += len(logs)
                successful = sum(1 for log in logs if log.get('success'))
                if logs:
                    summary['iris_success_rate'] = successful / len(logs)

        return summary

    except Exception as e:
        logger.error(f"Error generating log summary: {e}")
        return {}

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class StationDetection:
    """Single station P-wave detection result"""
    station_id: str
    detection_time: float
    delta_cmh: float
    confidence: float


@dataclass
class EarthquakeAlert:
    """Earthquake early warning alert"""
    alert_id: str
    detection_time: datetime
    num_stations: int
    estimated_magnitude: float
    magnitude_uncertainty: float
    confidence: float
    epicenter_lon: Optional[float]
    epicenter_lat: Optional[float]
    stations: List[StationDetection]

# ============================================================================
# USGS EARTHQUAKE DATA FETCHER
# ============================================================================


def fetch_usgs_earthquakes(days=7, min_magnitude=4.5):
    """
    Fetch recent earthquakes from USGS GeoJSON feed
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'minmagnitude': min_magnitude,
            'orderby': 'time-asc',
            'limit': 100
        }

        logger.info(
            f"🌍 Fetching USGS earthquakes: {start_time} to {end_time}, M>={min_magnitude}")

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        events = []

        for feature in data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']

            event = {
                'alert_id': feature['id'],
                'detection_time': datetime.fromtimestamp(props['time']/1000).isoformat(),
                'magnitude': round(props['mag'], 2) if props['mag'] else 0.0,
                'latitude': round(coords[1], 4),
                'longitude': round(coords[0], 4),
                'depth_km': round(coords[2], 1) if coords[2] else 0.0,
                'location': props.get('place', 'Unknown'),
                'num_stations': props.get('nst', 1),
                'confidence': 0.95,
                'source': 'USGS'
            }

            events.append(event)
            log_earthquake_event(event, source='USGS')

        logger.info(f"✓ Fetched {len(events)} earthquakes from USGS")
        return events

    except Exception as e:
        logger.error(f"❌ Error fetching USGS data: {e}")
        return []


def populate_initial_history():
    """Load last 7 days of earthquakes on startup"""
    global alert_history

    logger.info("=" * 60)
    logger.info("LOADING HISTORICAL EARTHQUAKE DATA")
    logger.info("=" * 60)

    events = fetch_usgs_earthquakes(days=7, min_magnitude=4.5)

    if events:
        alert_history.extend(events)
        logger.info(f"✓ Loaded {len(events)} earthquakes from last 7 days")
        logger.info(
            f"  Magnitude range: M{min(e['magnitude'] for e in events):.1f} - M{max(e['magnitude'] for e in events):.1f}")
    else:
        logger.warning("⚠ No historical earthquakes loaded")

    logger.info("=" * 60)


def auto_update_earthquakes():
    """Background task to fetch new earthquakes every 10 minutes"""
    global alert_history

    logger.info("🔄 Auto-update thread started (checks every 10 minutes)")

    while monitoring_active:
        try:
            time.sleep(600)  # Wait 10 minutes

            logger.info("Checking for new earthquakes...")
            recent_events = fetch_usgs_earthquakes(
                days=0.021, min_magnitude=4.5)  # ~30 min

            # Add new events only
            existing_ids = {e.get('alert_id') for e in alert_history}
            new_events = [
                e for e in recent_events if e['alert_id'] not in existing_ids]

            if new_events:
                alert_history.extend(new_events)
                logger.info(f"✓ Added {len(new_events)} new earthquake(s):")
                for event in new_events:
                    logger.info(
                        f"  • M{event['magnitude']} - {event['location']}")

                # Update latest alert
                global latest_alert
                if new_events:
                    latest_alert = new_events[-1]
            else:
                logger.info("  No new earthquakes detected")

            # Keep last 100 events
            if len(alert_history) > 100:
                alert_history = alert_history[-100:]

        except Exception as e:
            logger.error(f"Error in auto-update: {e}")

# ============================================================================
# GEOLOCATION SUPPORT
# ============================================================================


class EarthquakeAlertWithGeolocation:
    """Calculate distance, lead time, and format coordinates"""

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate great circle distance between two points (km)"""
        R = 6371
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * \
            math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    @staticmethod
    def estimate_lead_time(distance_km, p_wave_velocity=6.0):
        """Estimate lead time to strong motion arrival (seconds)"""
        p_wave_time = distance_km / p_wave_velocity
        return max(0, p_wave_time + 2.5)

    @staticmethod
    def decimal_to_dms(decimal, is_longitude=False):
        """Convert decimal degrees to DMS format"""
        absolute_value = abs(decimal)
        degrees = int(absolute_value)
        minutes_decimal = (absolute_value - degrees) * 60
        minutes = int(minutes_decimal)
        seconds = (minutes_decimal - minutes) * 60

        direction = 'E' if (is_longitude and decimal > 0) else (
            'W' if is_longitude else ('N' if decimal > 0 else 'S'))

        return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"

# ============================================================================
# CMH DETECTOR (Placeholder for Phase 2)
# ============================================================================


class CMHDetector:
    """∆CMH Seismic Event Detector"""

    def __init__(self, station_id: str, sampling_rate: float = 100.0):
        self.station_id = station_id
        self.sampling_rate = sampling_rate
        self.background_cmh = None
        self.detected = False
        self.delta_cmh_integral = 0
        self.detection_time = None
        self.confidence = 0.0

    def process(self) -> Optional[StationDetection]:
        """Process incoming waveform data"""
        if self.detected:
            return StationDetection(
                station_id=self.station_id,
                detection_time=self.detection_time,
                delta_cmh=self.delta_cmh_integral,
                confidence=self.confidence
            )
        return None

# ============================================================================
# MULTI-STATION CONSENSUS
# ============================================================================


class MultiStationConsensus:
    """Require multiple stations for earthquake confirmation"""

    def __init__(self, min_stations: int = 3):
        self.min_stations = min_stations
        self.detections: List[StationDetection] = []

    def add_detection(self, detection: StationDetection):
        if detection not in self.detections:
            self.detections.append(detection)

    def check_consensus(self) -> bool:
        return len(self.detections) >= self.min_stations

    def get_consensus_time(self) -> float:
        if not self.detections:
            return 0.0
        times = sorted([d.detection_time for d in self.detections])
        return times[len(times) // 2]

    def get_consensus_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

# ============================================================================
# MAGNITUDE ESTIMATION
# ============================================================================


class MagnitudeEstimator:
    """Estimate magnitude from ∆CMH integral values"""

    def estimate(self, integrals: List[float]) -> tuple:
        if not integrals:
            return 0.0, 0.42

        median_integral = sorted(integrals)[len(integrals) // 2]

        a, b, c = 105.85, -9.62, 4.46
        magnitude = a * (median_integral ** b) + c
        magnitude = max(3.0, min(9.0, magnitude))
        uncertainty = 0.42

        return magnitude, uncertainty

# ============================================================================
# CMH EARLY WARNING SYSTEM
# ============================================================================


class CMHEarthquakeEarlyWarning:
    """Complete CMH EEW System"""

    def __init__(self, station_ids: List[str], sampling_rate: float = 100.0):
        self.detectors = {sid: CMHDetector(
            sid, sampling_rate) for sid in station_ids}
        self.consensus = MultiStationConsensus(min_stations=3)
        self.magnitude_estimator = MagnitudeEstimator()
        self.alert_issued = False
        self.current_alert = None

        logger.info(f"EEW system initialized with {len(station_ids)} stations")

    def process(self) -> Optional[EarthquakeAlert]:
        if self.alert_issued:
            return self.current_alert

        for station_id, detector in self.detectors.items():
            detection = detector.process()
            if detection:
                self.consensus.add_detection(detection)

        if not self.consensus.check_consensus():
            return None

        consensus_time = self.consensus.get_consensus_time()
        consensus_confidence = self.consensus.get_consensus_confidence()

        integrals = [
            self.detectors[d.station_id].delta_cmh_integral
            for d in self.consensus.detections
        ]

        magnitude, mag_uncertainty = self.magnitude_estimator.estimate(
            integrals)

        estimated_lat, estimated_lon = 35.5, 138.5
        estimated_depth = 60

        alert = EarthquakeAlert(
            alert_id=f"CMH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            detection_time=datetime.now(),
            num_stations=len(self.consensus.detections),
            estimated_magnitude=magnitude,
            magnitude_uncertainty=mag_uncertainty,
            confidence=consensus_confidence,
            epicenter_lon=estimated_lon,
            epicenter_lat=estimated_lat,
            stations=self.consensus.detections
        )

        alert.depth_km = estimated_depth

        self.alert_issued = True
        self.current_alert = alert

        logger.warning(
            f"🚨 EARTHQUAKE ALERT ISSUED: M{magnitude:.1f} ± {mag_uncertainty:.2f}")

        return alert

# ============================================================================
# EEW ENGINE
# ============================================================================


class EEWEngine:
    """Real-time earthquake monitoring engine with SeedLink"""

    def __init__(self):
        # Real stations - 20 global coverage
        self.stations = [
            ("JA", "KAMAE"), ("JA", "OKW"), ("JA",
                                             "WTNM"), ("JA", "MZGH"), ("JA", "SHIZ"),
            ("CI", "PAS"), ("CI", "CLC"), ("CI",
                                           "LRL"), ("CI", "SBC"), ("CI", "SMO"),
            ("BK", "FARB"), ("BK", "YBH"), ("BK",
                                            "MCCM"), ("BK", "SAO"), ("BK", "CMB"),
            ("NC", "A25K"), ("NC", "H04P"), ("NC",
                                             "O22K"), ("NC", "Y27K"), ("NC", "Z24K"),
        ]

        self.seedlink_manager = None
        self.detections = []
        self.last_poll_time = datetime.now()
        self.alert_count = 0

        # Create old EEW system for compatibility
        station_ids = [f"{net}.{sta}" for net, sta in self.stations]
        self.eew_system = CMHEarthquakeEarlyWarning(
            station_ids=station_ids,
            sampling_rate=100.0
        )

    def on_detection(self, detection):
        """Callback when P-wave detected"""
        logger.warning(
            f"🚨 P-WAVE: {detection['station_id']} ΔCMH={detection['delta_cmh']:.3f}")

        # Log detection
        self.detections.append(detection)

        # Keep only last 100 detections
        if len(self.detections) > 100:
            self.detections.pop(0)

        # Check for multi-station consensus (3+ stations within 10 seconds)
        recent_time = detection['detection_time']
        recent_detections = [
            d for d in self.detections
            if abs(d['detection_time'] - recent_time) < 10
        ]

        if len(recent_detections) >= 3:
            self.issue_alert(recent_detections)

    def issue_alert(self, detections):
        """Issue earthquake alert from multiple detections"""
        global latest_alert, alert_history

        # Estimate magnitude from average delta CMH
        avg_delta_cmh = sum(d['delta_cmh']
                            for d in detections) / len(detections)
        magnitude = max(3.0, min(9.0, 4.0 + (avg_delta_cmh * 10)))

        alert = {
            'alertid': f"CMH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'detectiontime': datetime.now().isoformat(),
            'numstations': len(detections),
            'estimatedmagnitude': round(magnitude, 1),
            'magnitudeuncertainty': 0.5,
            'confidence': min(avg_delta_cmh / 0.5, 1.0),
            'stations': [d['station_id'] for d in detections],
            'source': 'CMH-REALTIME'
        }

        latest_alert = alert
        alert_history.append(alert)

        logger.warning(
            f"🚨🚨🚨 EARTHQUAKE ALERT: M{magnitude:.1f} detected by {len(detections)} stations")

        # Log to file
        log_earthquake_event(alert, source='CMH-RT')

    def poll_data(self):
        """Compatibility method"""
        self.last_poll_time = datetime.now()

    def initialize(self):
        """Start SeedLink real-time monitoring"""
        logger.info("⚡ Initializing SeedLink real-time monitoring...")

        self.seedlink_manager = SeedLinkManager(self.on_detection)

        # Add all 20 stations
        for network, station in self.stations:
            self.seedlink_manager.add_station(network, station)

        # Start streaming
        self.seedlink_manager.start()

        logger.info(
            f"✅ SeedLink ACTIVE: {len(self.stations)} stations streaming real-time data")


class EEWEngine:
    """Real-time earthquake monitoring engine with SeedLink"""

    def __init__(self):
        # Real stations - 20 global coverage
        self.stations = [
            ("JA", "KAMAE"), ("JA", "OKW"), ("JA",
                                             "WTNM"), ("JA", "MZGH"), ("JA", "SHIZ"),
            ("CI", "PAS"), ("CI", "CLC"), ("CI",
                                           "LRL"), ("CI", "SBC"), ("CI", "SMO"),
            ("BK", "FARB"), ("BK", "YBH"), ("BK",
                                            "MCCM"), ("BK", "SAO"), ("BK", "CMB"),
            ("NC", "A25K"), ("NC", "H04P"), ("NC",
                                             "O22K"), ("NC", "Y27K"), ("NC", "Z24K"),
        ]

        self.seedlink_manager = None
        self.detections = []
        self.last_poll_time = datetime.now()
        self.alert_count = 0

        # Create old EEW system for compatibility
        station_ids = [f"{net}.{sta}" for net, sta in self.stations]
        self.eew_system = CMHEarthquakeEarlyWarning(
            station_ids=station_ids,
            sampling_rate=100.0
        )

    def on_detection(self, detection):
        """Callback when P-wave detected"""
        logger.warning(
            f"🚨 P-WAVE: {detection['station_id']} ΔCMH={detection['delta_cmh']:.3f}")

        # Log detection
        self.detections.append(detection)

        # Keep only last 100 detections
        if len(self.detections) > 100:
            self.detections.pop(0)

        # Check for multi-station consensus (3+ stations within 10 seconds)
        recent_time = detection['detection_time']
        recent_detections = [
            d for d in self.detections
            if abs(d['detection_time'] - recent_time) < 10
        ]

        if len(recent_detections) >= 3:
            self.issue_alert(recent_detections)

    def issue_alert(self, detections):
        """Issue earthquake alert from multiple detections"""
        global latest_alert, alert_history

        # Estimate magnitude from average delta CMH
        avg_delta_cmh = sum(d['delta_cmh']
                            for d in detections) / len(detections)
        magnitude = max(3.0, min(9.0, 4.0 + (avg_delta_cmh * 10)))

        alert = {
            'alertid': f"CMH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'detectiontime': datetime.now().isoformat(),
            'numstations': len(detections),
            'estimatedmagnitude': round(magnitude, 1),
            'magnitudeuncertainty': 0.5,
            'confidence': min(avg_delta_cmh / 0.5, 1.0),
            'stations': [d['station_id'] for d in detections],
            'source': 'CMH-REALTIME'
        }

        latest_alert = alert
        alert_history.append(alert)

        logger.warning(
            f"🚨🚨🚨 EARTHQUAKE ALERT: M{magnitude:.1f} detected by {len(detections)} stations")

        # Log to file
        log_earthquake_event(alert, source='CMH-RT')

    def poll_data(self):
        """Compatibility method"""
        self.last_poll_time = datetime.now()

    def initialize(self):
        """Start SeedLink real-time monitoring"""
        logger.info("⚡ Initializing SeedLink real-time monitoring...")

        self.seedlink_manager = SeedLinkManager(self.on_detection)

        # Add all 20 stations
        for network, station in self.stations:
            self.seedlink_manager.add_station(network, station)

        # Start streaming
        self.seedlink_manager.start()

        logger.info(
            f"✅ SeedLink ACTIVE: {len(self.stations)} stations streaming real-time data")


# ============================================================================
# FLASK REST API ENDPOINTS
# ============================================================================


@app.route('/')
def serve_dashboard():
    """Serve the main dashboard HTML"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            'status': 'running',
            'message': 'CMH EEW Backend is running. Dashboard HTML not found.',
            'endpoints': {
                'status': '/api/status',
                'latest_alert': '/api/latest-alert',
                'alert_history': '/api/alert-history',
                'test_historical': '/api/test-historical',
                'log_summary': '/api/log-summary',
                'download_data': '/api/download-data'
            }
        })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'last_poll': eew_engine.last_poll_time.isoformat(),
        'alerts_issued': len(alert_history),
        'active_stations': len(eew_engine.eew_system.detectors),
        'data_directory': str(DATA_DIR.absolute())
    })


@app.route('/api/latest-alert', methods=['GET'])
def get_latest_alert():
    """Get latest earthquake alert"""
    if latest_alert:
        return jsonify(latest_alert)
    return jsonify({'status': 'no_alerts'})


@app.route('/api/alert-history', methods=['GET'])
def get_alert_history():
    """Get recent earthquake alerts"""
    recent = alert_history[-50:][::-1]
    return jsonify(recent)


@app.route('/api/log-summary', methods=['GET'])
def get_logs_summary():
    """Get summary of logged data"""
    summary = get_log_summary()
    return jsonify(summary)


@app.route('/api/download-data', methods=['GET'])
def download_research_data():
    """
    Download all research data as ZIP file
    Filename: yymmdd-erthqk-export-v1.zip
    """
    try:
        # Create in-memory ZIP file
        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all data files
            for file in DATA_DIR.glob('*-erthqk-*.json'):
                zf.write(file, file.name)

            for file in DATA_DIR.glob('*-erthqk-*.csv'):
                zf.write(file, file.name)

            # Add summary
            summary = get_log_summary()
            zf.writestr('summary.json', json.dumps(summary, indent=2))

        memory_file.seek(0)

        zip_filename = get_filename('export', version=1, extension='zip')

        logger.info(f"📦 Data export requested: {zip_filename}")

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        logger.error(f"Error creating data export: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/test-historical', methods=['GET'])
def test_historical_quake():
    """Test with real historical IRIS data"""
    try:
        event_time = request.args.get('origin_time', '2011-03-11T05:46:18')
        net = request.args.get('network', 'IU')
        sta = request.args.get('station', 'ANMO')
        cha = request.args.get('channel', 'BHZ')
        dur = int(request.args.get('duration', 180))

        client = IRISClient("IRIS")
        t0 = UTCDateTime(event_time)
        st = client.get_waveforms(
            network=net, station=sta, location='00',
            channel=cha, starttime=t0, endtime=t0+dur
        )

        samples = st[0].data.tolist()

        # LOG SUCCESS
        log_iris_waveform_fetch(
            event_id=event_time,
            network=net,
            station=sta,
            channel=cha,
            success=True,
            n_samples=len(samples),
            sampling_rate=st[0].stats.sampling_rate
        )

        return jsonify({
            'success': True,
            'station': f"{net}.{sta}",
            'n_samples': len(samples),
            'starttime': str(st[0].stats.starttime),
            'sampling_rate': st[0].stats.sampling_rate,
            'preview_signal': samples[:500],
        })

    except Exception as e:
        # LOG FAILURE
        log_iris_waveform_fetch(
            event_id=event_time,
            network=net,
            station=sta,
            channel=cha,
            success=False,
            error=str(e)
        )

        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# BACKGROUND THREADS
# ============================================================================


def monitoring_thread():
    """Background thread for continuous monitoring"""
    global monitoring_active

    while monitoring_active:
        try:
            eew_engine.poll_data()
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
            time.sleep(5)

# ============================================================================
# APPLICATION STARTUP
# ============================================================================


@app.before_request
def initialize_app():
    """Initialize app on first request"""
    global eew_engine, monitoring_active

    if eew_engine is None:
        logger.info("=" * 80)
        logger.info("INITIALIZING CMH EARTHQUAKE EARLY WARNING SYSTEM v2.0")
        logger.info("=" * 80)

        eew_engine = EEWEngine()
        eew_engine.initialize()

        # Load last 7 days of earthquakes from USGS
        populate_initial_history()

        monitoring_active = True

        # Start monitoring thread
        thread = threading.Thread(target=monitoring_thread, daemon=True)
        thread.start()

        # Start auto-update thread
        update_thread = threading.Thread(
            target=auto_update_earthquakes, daemon=True)
        update_thread.start()

        logger.info("✓ Background monitoring started")
        logger.info("✓ Auto-update enabled (checks every 10 minutes)")
        logger.info(f"✓ Data logging enabled: {DATA_DIR.absolute()}")
        logger.info("✓ File naming: yymmdd-erthqk-type-vX.ext")
        logger.info("=" * 80)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
# ============================================================================
# SEEDLINK REAL-TIME EEW ENDPOINTS
# ============================================================================

@app.route('/api/seedlink/start', methods=['POST'])
def start_seedlink():
    """Start real-time SeedLink listener"""
    global seedlink_pipeline, seedlink_running
    try:
        if seedlink_pipeline is None:
            seedlink_pipeline = EEWPipeline()

        if seedlink_pipeline.start():
            seedlink_running = True
            logger.info("✓ SeedLink pipeline started")
            return jsonify({
                'status': 'started',
                'message': 'SeedLink listener running',
                'pipeline': seedlink_pipeline.get_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to start pipeline'
            }), 500

    except Exception as e:
        logger.error(f"Error starting SeedLink: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/seedlink/status', methods=['GET'])
def get_seedlink_status():
    """Get SeedLink pipeline status"""
    if seedlink_pipeline is None:
        return jsonify({'status': 'not_initialized'})

    return jsonify(seedlink_pipeline.get_status())


@app.route('/api/seedlink/alerts', methods=['GET'])
def get_seedlink_alerts():
    """Get recent SeedLink-based alerts"""
    if seedlink_pipeline is None:
        return jsonify([])

    n = request.args.get('limit', 10, type=int)
    return jsonify(seedlink_pipeline.get_recent_alerts(n))


@app.route('/api/seedlink/stop', methods=['POST'])
def stop_seedlink():
    """Stop SeedLink listener"""
    global seedlink_pipeline, seedlink_running

    if seedlink_pipeline:
        seedlink_pipeline.stop()
        seedlink_running = False
        logger.info("✓ SeedLink pipeline stopped")

    return jsonify({
        'status': 'stopped',
        'message': 'SeedLink listener stopped'
    })

# ============================================================================
# PERFORMANCE CHARTS ENDPOINTS
# ============================================================================


# Initialize chart generator
chart_generator = PerformanceCharts(DATA_DIR)


@app.route('/api/performance-charts', methods=['GET'])
def get_performance_charts():
    """Generate and return performance run charts"""
    try:
        days = request.args.get('days', 90, type=int)
        charts = chart_generator.generate_all_charts(days=days)

        return jsonify({
            'status': 'success',
            'charts_generated': len(charts),
            'charts': [str(c.name) for c in charts]
        })
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/charts/hart_name>', methods=['GET'])
def serve_chart(chart_name):
    """Serve generated chart images"""
    chart_path = DATA_DIR / 'charts' / chart_name

    if chart_path.exists():
        return send_file(str(chart_path), mimetype='image/png')
    else:
        return jsonify({'error': 'Chart not found'}), 404


@app.route('/api/chart-status', methods=['GET'])
def chart_status():
    """Check which charts are available"""
    charts_dir = DATA_DIR / 'charts'
    if not charts_dir.exists():
        return jsonify({'status': 'no_charts', 'charts': []})

    charts = list(charts_dir.glob('*.png'))
    return jsonify({
        'status': 'ok',
        'total_charts': len(charts),
        'charts': [c.name for c in charts]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    print("=" * 80)
    print("CMH EARTHQUAKE EARLY WARNING SYSTEM v2.0 - IRIS REAL-TIME MONITOR")
    print("=" * 80)
    print()
    print(f"Backend: Flask API on http://localhost:{port}")
    print("Endpoints:")
    print("  GET /                       - Dashboard")
    print("  GET /api/status             - System status")
    print("  GET /api/latest-alert       - Latest earthquake")
    print("  GET /api/alert-history      - Alert history (last 50)")
    print("  GET /api/test-historical    - Historical IRIS waveform demo")
    print("  GET /api/log-summary        - Data logging statistics")
    print("  GET /api/download-data      - Download research data (ZIP)")
    print()
    print("Data Sources:")
    print("  • USGS: Real-time earthquake feed (M4.5+)")
    print("  • IRIS: Waveform data for events")
    print("  • Updates: Every 10 minutes")
    print()
    print("Data Logging (yymmdd-erthqk-type-vX.ext):")
    print(f"  • Directory: {DATA_DIR.absolute()}")
    print("  • USGS events: yymmdd-erthqk-usgs-v1.json")
    print("  • IRIS waveforms: yymmdd-erthqk-iris-v1.json")
    print("  • Daily stats: yymmdd-erthqk-stats-v1.csv")
    print("  • Data export: yymmdd-erthqk-export-v1.zip")
    print()
    print("=" * 80)
    print()

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
