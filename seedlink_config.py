#!/usr/bin/env python3
"""
SeedLink Configuration Module
Defines IRIS server connections, station selections, and thresholds
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SEEDLINK SERVER CONFIGURATIONS
# ============================================================================

SEEDLINK_SERVERS = {
    'IRIS': {
        'host': 'rtserve.iris.washington.edu',
        'port': 18000,
        'timeout': 30,
        'description': 'IRIS Global Seismic Network'
    },
    'GEOFON': {
        'host': 'geofon.gfz-potsdam.de',
        'port': 18001,
        'timeout': 30,
        'description': 'GFZ GEOFON Network'
    },
    'GFZ': {
        'host': 'eida.gfz-potsdam.de',
        'port': 18000,
        'timeout': 30,
        'description': 'GFZ EIDA Network'
    },
    'ETH': {
        'host': 'eida.ethz.ch',
        'port': 18000,
        'timeout': 30,
        'description': 'ETH EIDA Network'
    }
}

# ============================================================================
# ACTIVE SERVER SELECTION
# ============================================================================

ACTIVE_SERVER = 'IRIS'  # Change to test different networks: 'IRIS', 'GEOFON', 'GFZ', 'ETH'

def get_active_server():
    """Get active SeedLink server configuration"""
    if ACTIVE_SERVER not in SEEDLINK_SERVERS:
        logger.error(f"Unknown server: {ACTIVE_SERVER}. Using IRIS.")
        return SEEDLINK_SERVERS['IRIS']
    return SEEDLINK_SERVERS[ACTIVE_SERVER]

# ============================================================================
# STATION SELECTIONS (Global Seismic Network)
# ============================================================================

STATION_SELECTIONS = [
    # Global Backbone Stations
    "IU.ANMO.00.BHZ",    # Albuquerque, USA
    "IU.CHTO.00.BHZ",    # Chiang Mai, Thailand
    "IU.NWAO.00.BHZ",    # Perth, Australia
    "IU.GUMO.00.BHZ",    # Guam
    "IU.CTAO.00.BHZ",    # Canberra, Australia
    "IU.SSPA.00.BHZ",    # South Pole
    "IU.INCN.00.BHZ",    # Inchon, South Korea
    "IU.KONO.00.BHZ",    # Kongsberg, Norway
    "IU.PEL.00.BHZ",     # Pellican Lake, Canada
    "IU.RSSD.00.BHZ",    # Black Hills, South Dakota
    
    # Regional Networks - Japan
    "JA.KAMAE.00.BHZ",   # Kamae, Japan
    "JA.OKW.00.BHZ",     # Okayama West, Japan
    "JA.WTNM.00.BHZ",    # Watashinomine, Japan
    
    # Regional Networks - USA
    "CI.PAS.00.BHZ",     # Pasadena, California
    "CI.LRL.00.BHZ",     # Long Range Lake, California
    "BK.FARB.00.BHZ",    # Farallon, California
    "NC.A25K.00.BHZ",    # Northern California
    
    # European Networks
    "GE.MORC.00.BHZ",    # Morocco
    "GE.SNAA.00.BHZ",    # Sanaa, Yemen
    "CH.LIENZ.00.BHZ",   # Lienz, Austria
]

# ============================================================================
# SEEDLINK CONNECTION PARAMETERS
# ============================================================================

SEEDLINK_CONFIG = {
    'buffer_duration_seconds': 10.0,      # Keep last 10 seconds of waveform
    'reconnect_interval': 5,               # Try reconnect every 5 seconds
    'max_reconnect_attempts': 10,          # Give up after 10 attempts
    'packet_timeout': 30,                  # Timeout waiting for packets
    'max_buffer_size': 1000000,            # Max samples in memory (1MB)
}

# ============================================================================
# EEW THRESHOLDS & PARAMETERS
# ============================================================================

EEW_THRESHOLDS = {
    'magnitude_threshold': 4.5,            # Only alert for M >= 4.5
    'min_stations': 2,                     # Need at least 2 stations
    'confidence_threshold': 0.70,          # Confidence must be > 70%
    'max_processing_delay_ms': 5000,       # Process within 5 seconds
}

# ============================================================================
# MAGNITUDE ESTIMATION PARAMETERS
# ============================================================================

CMH_MAGNITUDE_PARAMS = {
    'a': 105.85,                           # Power law coefficient
    'b': -9.62,                            # Power law exponent
    'c': 4.46,                             # Offset
    'min_magnitude': 3.0,                  # Minimum detectable magnitude
    'max_magnitude': 9.0,                  # Maximum realistic magnitude
    'uncertainty': 0.42,                   # Magnitude uncertainty (±0.42)
}

# ============================================================================
# P-WAVE DETECTION PARAMETERS
# ============================================================================

P_WAVE_DETECTION = {
    'sampling_rate': 100.0,                # Hz
    'short_window': 1.0,                   # 1 second (short-term average)
    'long_window': 10.0,                   # 10 seconds (long-term average)
    'threshold_ratio': 3.0,                # STA/LTA ratio threshold
    'min_amplitude': 0.5,                  # Minimum amplitude threshold
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'seedlink_eew.log',
    'file_size_mb': 10,
    'backup_count': 5,
}

# ============================================================================
# DATA EXPORT CONFIGURATION
# ============================================================================

DATA_EXPORT = {
    'export_dir': 'data/seedlink',
    'waveform_format': 'h5',               # HDF5 for efficient storage
    'alert_format': 'json',
    'max_export_size_mb': 500,
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check active server
    if ACTIVE_SERVER not in SEEDLINK_SERVERS:
        errors.append(f"Invalid ACTIVE_SERVER: {ACTIVE_SERVER}")
    
    # Check threshold values
    if not (3.0 <= EEW_THRESHOLDS['magnitude_threshold'] <= 9.0):
        errors.append("magnitude_threshold must be between 3.0 and 9.0")
    
    if EEW_THRESHOLDS['min_stations'] < 1:
        errors.append("min_stations must be >= 1")
    
    if not (0 < EEW_THRESHOLDS['confidence_threshold'] <= 1.0):
        errors.append("confidence_threshold must be between 0 and 1")
    
    # Check CMH parameters
    if CMH_MAGNITUDE_PARAMS['a'] <= 0:
        errors.append("CMH coefficient 'a' must be positive")
    
    # Check P-wave parameters
    if P_WAVE_DETECTION['sampling_rate'] <= 0:
        errors.append("sampling_rate must be positive")
    
    if P_WAVE_DETECTION['threshold_ratio'] <= 1.0:
        errors.append("threshold_ratio must be > 1.0")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False
    
    logger.info("✓ Configuration validation passed")
    return True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_station_list():
    """Get list of stations to monitor"""
    return STATION_SELECTIONS

def get_active_server_info():
    """Get active server information"""
    server = get_active_server()
    return {
        'server': ACTIVE_SERVER,
        'host': server['host'],
        'port': server['port'],
        'description': server['description']
    }

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("SEEDLINK CONFIGURATION TEST")
    print("="*70)
    
    print("\n📍 Active Server:")
    info = get_active_server_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n📡 Monitoring {len(STATION_SELECTIONS)} stations:")
    for i, sta in enumerate(STATION_SELECTIONS[:5], 1):
        print(f"  {i}. {sta}")
    if len(STATION_SELECTIONS) > 5:
        print(f"  ... and {len(STATION_SELECTIONS) - 5} more")
    
    print("\n⚙️  EEW Thresholds:")
    for key, value in EEW_THRESHOLDS.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Configuration loaded successfully")
    print("="*70 + "\n")
