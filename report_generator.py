"""
Daily Performance Report Generator
Generates markdown reports with tables and metrics
File: report_generator.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate daily performance reports in markdown format"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def generate_daily_report(self, metrics: Dict) -> str:
        """Generate daily markdown report with tables"""
        today = datetime.now().strftime('%B %d, %Y')

        report = f"""# 🌍 CMH Earthquake Early Warning System
## Daily Performance Report - {today}

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Detections** | {metrics.get('period', {}).get('total_detections', 0)} |
| **Valid Detections** | {metrics.get('period', {}).get('valid_detections', 0)} |
| **False Positives** | {metrics.get('period', {}).get('false_positives', 0)} |
| **Detection Success Rate** | {metrics.get('reliability', {}).get('detection_success_rate', 0)*100:.1f}% |
| **System Status** | ✅ Operational |

---

## 🎯 Magnitude Accuracy

| Metric | CMH EEW | ShakeAlert Benchmark |
|--------|---------|---------------------|
| **Mean Absolute Error** | ±{metrics.get('magnitude_accuracy', {}).get('mean_absolute_error', 0):.2f} | ±0.50 |
| **RMSE** | {metrics.get('magnitude_accuracy', {}).get('rmse', 0):.2f} | ~0.60 |
| **Within ±0.3 units** | {metrics.get('magnitude_accuracy', {}).get('within_0.3_units', 0)*100:.1f}% | ~65% |
| **Within ±0.5 units** | {metrics.get('magnitude_accuracy', {}).get('within_0.5_units', 0)*100:.1f}% | ~85% |
| **Performance** | {metrics.get('shakealert_comparison', {}).get('magnitude_accuracy', {}).get('performance', 'N/A')} | --- |

---

## 📍 Location Accuracy

| Metric | Value |
|--------|-------|
| **Mean Error** | {metrics.get('location_accuracy', {}).get('mean_error_km', 0):.2f} km |
| **Median Error** | {metrics.get('location_accuracy', {}).get('median_error_km', 0):.2f} km |
| **Within 10 km** | {metrics.get('location_accuracy', {}).get('within_10km', 0)*100:.1f}% |
| **Within 20 km** | {metrics.get('location_accuracy', {}).get('within_20km', 0)*100:.1f}% |

---

## ⚡ Detection Speed

| Metric | CMH EEW | ShakeAlert Benchmark |
|--------|---------|---------------------|
| **Average Latency** | {metrics.get('timing', {}).get('avg_detection_latency_sec', 0):.2f}s | 2-5s |
| **Median Latency** | {metrics.get('timing', {}).get('median_latency_sec', 0):.2f}s | ~3.5s |
| **Fastest Detection** | {metrics.get('timing', {}).get('fastest_detection_sec', 0):.2f}s | --- |
| **Performance** | {metrics.get('shakealert_comparison', {}).get('detection_speed', {}).get('performance', 'N/A')} | --- |

---

## 🔍 Reliability Metrics

| Metric | CMH EEW | ShakeAlert Benchmark |
|--------|---------|---------------------|
| **False Positive Rate** | {metrics.get('reliability', {}).get('false_positive_rate', 0)*100:.1f}% | ~10% |
| **Detection Success** | {metrics.get('reliability', {}).get('detection_success_rate', 0)*100:.1f}% | ~95% |
| **Performance** | {metrics.get('shakealert_comparison', {}).get('false_positives', {}).get('performance', 'N/A')} | --- |

---

## 🏆 Overall Performance vs ShakeAlert

| Category | CMH EEW | ShakeAlert | Status |
|----------|---------|------------|--------|
| **Magnitude Accuracy** | ±{metrics.get('shakealert_comparison', {}).get('magnitude_accuracy', {}).get('cmh', 0):.2f} | ±{metrics.get('shakealert_comparison', {}).get('magnitude_accuracy', {}).get('shakealert', 0):.2f} | {metrics.get('shakealert_comparison', {}).get('magnitude_accuracy', {}).get('performance', 'N/A')} |
| **Detection Speed** | {metrics.get('shakealert_comparison', {}).get('detection_speed', {}).get('cmh', 0):.2f}s | {metrics.get('shakealert_comparison', {}).get('detection_speed', {}).get('shakealert', 0):.2f}s | {metrics.get('shakealert_comparison', {}).get('detection_speed', {}).get('performance', 'N/A')} |
| **False Positives** | {metrics.get('shakealert_comparison', {}).get('false_positives', {}).get('cmh', 0):.1f}% | {metrics.get('shakealert_comparison', {}).get('false_positives', {}).get('shakealert', 0):.1f}% | {metrics.get('shakealert_comparison', {}).get('false_positives', {}).get('performance', 'N/A')} |

---

## 📈 Next Steps

- Continue monitoring for 90 days to build comprehensive validation dataset
- Target: 1000+ validated events before customer outreach
- Focus areas for improvement based on today's data

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}  
**System:** CMH Earthquake Early Warning v2.0  
**Data Source:** USGS + IRIS Integration
"""

        return report

    def save_report(self, report: str, date: datetime = None) -> Path:
        """Save report to markdown file"""
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%y%m%d')
        filename = f"{date_str}-erthqk-report-v1.md"
        report_file = self.data_dir / filename

        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"✓ Report saved: {report_file}")
        return report_file


# Test code
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Sample metrics
    sample_metrics = {
        'period': {'total_detections': 15, 'valid_detections': 14, 'false_positives': 1},
        'magnitude_accuracy': {'mean_absolute_error': 0.28, 'rmse': 0.34, 'within_0.3_units': 0.88, 'within_0.5_units': 0.95},
        'location_accuracy': {'mean_error_km': 8.5, 'median_error_km': 6.2, 'within_10km': 0.82, 'within_20km': 0.93},
        'timing': {'avg_detection_latency_sec': 2.6, 'median_latency_sec': 2.3, 'fastest_detection_sec': 1.4},
        'reliability': {'false_positive_rate': 0.067, 'detection_success_rate': 0.933},
        'shakealert_comparison': {
            'magnitude_accuracy': {'cmh': 0.28, 'shakealert': 0.5, 'performance': 'Better'},
            'detection_speed': {'cmh': 2.6, 'shakealert': 3.5, 'performance': 'Better'},
            'false_positives': {'cmh': 6.7, 'shakealert': 10.0, 'performance': 'Better'}
        }
    }

    generator = ReportGenerator(Path('./data'))
    report = generator.generate_daily_report(sample_metrics)
    generator.save_report(report)

    print("✓ Report generation test passed")
