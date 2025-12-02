import logging
import csv
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Ensures non-interactive chart generation on servers


class PerformanceCharts:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.charts_dir = self.data_dir / 'charts'
        self.charts_dir.mkdir(exist_ok=True)

        self.shakealert_benchmarks = {
            'magnitude_error': 0.5,
            'detection_latency': 3.5,
            'false_positive_rate': 10.0,
        }
        self.google_benchmarks = {
            'magnitude_error': 0.38,
            'detection_latency': 4.0,
            'false_positive_rate': 8.0,
        }

    def load_performance_data(self, days=90):
        cutoff = datetime.now() - timedelta(days=days)
        all_data = []
        for file in sorted(self.data_dir.glob('*-erthqk-performance-v1.csv')):
            try:
                with open(file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            timestamp = datetime.fromisoformat(
                                row['timestamp'])
                            if timestamp > cutoff:
                                all_data.append(row)
                        except Exception:
                            continue
            except Exception:
                continue
        return all_data

    def aggregate_daily_metrics(self, data):
        from collections import defaultdict
        daily = defaultdict(list)
        for row in data:
            try:
                timestamp = datetime.fromisoformat(row['timestamp'])
                k = timestamp.strftime('%Y-%m-%d')
                if row.get('magnitude_error') and row['magnitude_error'] != 'N/A':
                    daily[k].append({
                        'magnitude_error': float(row['magnitude_error']),
                        'location_error_km': float(row.get('location_error_km', 0)) if row.get('location_error_km') != 'N/A' else None,
                        'detection_latency_sec': float(row.get('detection_latency_sec', 0)) if row.get('detection_latency_sec') != 'N/A' else None,
                        'false_positive': row.get('false_positive', False)
                    })
            except Exception:
                continue
        metrics = {}
        for date_str, entries in daily.items():
            mag = [x['magnitude_error'] for x in entries]
            loc = [x['location_error_km']
                   for x in entries if x['location_error_km'] is not None]
            lat = [x['detection_latency_sec']
                   for x in entries if x['detection_latency_sec'] is not None]
            fp = sum(1 for x in entries if x.get('false_positive'))
            metrics[date_str] = {
                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                'magnitude_error_avg': np.mean(mag) if mag else 0,
                'magnitude_error_std': np.std(mag) if mag else 0,
                'location_error_avg': np.mean(loc) if loc else 0,
                'detection_latency_avg': np.mean(lat) if lat else 0,
                'detection_latency_std': np.std(lat) if lat else 0,
                'false_positive_rate': (fp/len(entries)*100) if entries else 0,
                'total_detections': len(entries),
            }
        return metrics

    def create_magnitude_accuracy_chart(self, daily_metrics):
        fig, ax = plt.subplots(figsize=(14, 7))
        sorted_data = sorted(daily_metrics.items(), key=lambda x: x[1]['date'])
        dates = [item[1]['date'] for item in sorted_data]
        mag_errors = [item[1]['magnitude_error_avg'] for item in sorted_data]
        mag_stds = [item[1]['magnitude_error_std'] for item in sorted_data]
        ax.plot(dates, mag_errors, 'o-', color='#3498db',
                linewidth=2, markersize=6, label='CMH EEW', zorder=3)

        upper = [m+s for m, s in zip(mag_errors, mag_stds)]
        lower = [max(0, m-s) for m, s in zip(mag_errors, mag_stds)]
        ax.fill_between(dates, lower, upper, alpha=0.2, color='#3498db')

        ax.axhline(y=self.shakealert_benchmarks['magnitude_error'],
                   color='#e74c3c', linestyle='--', linewidth=2, label='ShakeAlert (±0.5)')
        ax.axhline(y=self.google_benchmarks['magnitude_error'], color='#f39c12',
                   linestyle='--', linewidth=2, label='Google EEW (±0.38)')
        ax.axhline(y=0.3, color='#27ae60', linestyle=':',
                   linewidth=2, label='Target (±0.3)')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mean Absolute Magnitude Error', fontsize=12)
        ax.set_title(
            'Magnitude Accuracy Over Time - CMH EEW vs Benchmarks', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.xticks(rotation=45, ha='right')

        latest_error = mag_errors[-1] if mag_errors else 0
        status = '❌ NEEDS IMPROVEMENT'
        color = '#e74c3c'
        if latest_error < 0.3:
            status = '✅ EXCELLENT'
            color = "#2a4133"
        elif latest_error < self.google_benchmarks['magnitude_error']:
            status = '✅ BEATING GOOGLE'
            color = '#f39c12'
        elif latest_error < self.shakealert_benchmarks['magnitude_error']:
            status = '⚠️ BEATING SHAKEALERT'
            color = '#e67e22'
        ax.text(0.02, 0.98, f'Current Status: {status}', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        plt.tight_layout()
        chart_path = self.charts_dir / 'magnitude_accuracy_run_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path

    # For brevity here, detection speed/false positive charts use same pattern, and combined dashboard as described in prior response.

    def create_detection_speed_chart(self, daily_metrics):
        """Create detection speed run chart"""
        fig, ax = plt.subplots(figsize=(14, 7))
        sorted_data = sorted(daily_metrics.items(), key=lambda x: x[1]['date'])
        dates = [item[1]['date'] for item in sorted_data]
        latencies = [item[1]['detection_latency_avg'] for item in sorted_data]
        latency_stds = [item[1]['detection_latency_std']
                        for item in sorted_data]

        ax.plot(dates, latencies, 's-', color='#9b59b6',
                linewidth=2, markersize=6, label='CMH EEW', zorder=3)
        upper = [l+s for l, s in zip(latencies, latency_stds)]
        lower = [max(0, l-s) for l, s in zip(latencies, latency_stds)]
        ax.fill_between(dates, lower, upper, alpha=0.2, color='#9b59b6')

        ax.axhline(y=self.shakealert_benchmarks['detection_latency'],
                   color='#e74c3c', linestyle='--', linewidth=2, label='ShakeAlert (3.5s)')
        ax.axhline(y=self.google_benchmarks['detection_latency'], color='#f39c12',
                   linestyle='--', linewidth=2, label='Google EEW (4.0s)')
        ax.axhline(y=3.0, color='#27ae60', linestyle=':',
                   linewidth=2, label='Target (3.0s)')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Detection Latency (seconds)', fontsize=12)
        ax.set_title(
            'Detection Speed Over Time - CMH EEW vs Benchmarks', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.xticks(rotation=45, ha='right')

        latest_latency = latencies[-1] if latencies else 0
        status = '❌ NEEDS IMPROVEMENT'
        color = '#e74c3c'
        if latest_latency < 3.0:
            status = '✅ EXCELLENT'
            color = '#27ae60'
        elif latest_latency < self.shakealert_benchmarks['detection_latency']:
            status = '✅ BEATING SHAKEALERT'
            color = '#f39c12'
        elif latest_latency < self.google_benchmarks['detection_latency']:
            status = '⚠️ BEATING GOOGLE'
            color = '#e67e22'

        ax.text(0.02, 0.98, f'Current Status: {status}', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        plt.tight_layout()
        chart_path = self.charts_dir / 'detection_speed_run_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path

    def create_false_positive_chart(self, daily_metrics):
        """Create false positive rate run chart"""
        fig, ax = plt.subplots(figsize=(14, 7))
        sorted_data = sorted(daily_metrics.items(), key=lambda x: x[1]['date'])
        dates = [item[1]['date'] for item in sorted_data]
        fp_rates = [item[1]['false_positive_rate'] for item in sorted_data]

        ax.plot(dates, fp_rates, '^-', color='#e74c3c', linewidth=2,
                markersize=6, label='CMH EEW', zorder=3)
        ax.axhline(y=self.shakealert_benchmarks['false_positive_rate'],
                   color='#c0392b', linestyle='--', linewidth=2, label='ShakeAlert (10%)')
        ax.axhline(y=self.google_benchmarks['false_positive_rate'],
                   color='#d35400', linestyle='--', linewidth=2, label='Google EEW (8%)')
        ax.axhline(y=5.0, color='#27ae60', linestyle=':',
                   linewidth=2, label='Target (5%)')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('False Positive Rate (%)', fontsize=12)
        ax.set_title(
            'False Positive Rate Over Time - CMH EEW vs Benchmarks', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.xticks(rotation=45, ha='right')

        latest_fp = fp_rates[-1] if fp_rates else 0
        status = '❌ NEEDS IMPROVEMENT'
        color = '#e74c3c'
        if latest_fp < 5.0:
            status = '✅ EXCELLENT'
            color = '#27ae60'
        elif latest_fp < self.google_benchmarks['false_positive_rate']:
            status = '✅ BEATING GOOGLE'
            color = '#f39c12'
        elif latest_fp < self.shakealert_benchmarks['false_positive_rate']:
            status = '⚠️ BEATING SHAKEALERT'
            color = '#e67e22'

        ax.text(0.02, 0.98, f'Current Status: {status}', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        plt.tight_layout()
        chart_path = self.charts_dir / 'false_positive_run_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path

    def create_combined_dashboard(self, daily_metrics):
        """Create combined 3-month dashboard"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        sorted_data = sorted(daily_metrics.items(), key=lambda x: x[1]['date'])
        dates = [item[1]['date'] for item in sorted_data]

        # Chart 1: Magnitude Accuracy
        ax = axes[0]
        mag_errors = [item[1]['magnitude_error_avg'] for item in sorted_data]
        ax.plot(dates, mag_errors, 'o-', color='#3498db',
                linewidth=2, markersize=5)
        ax.axhline(y=0.5, color='#e74c3c', linestyle='--', label='ShakeAlert')
        ax.axhline(y=0.38, color='#f39c12', linestyle='--', label='Google')
        ax.axhline(y=0.3, color='#27ae60', linestyle=':', label='Target')
        ax.set_ylabel('Magnitude Error', fontweight='bold')
        ax.set_title('3-Month Performance Tracking Dashboard',
                     fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Chart 2: Detection Speed
        ax = axes[1]
        latencies = [item[1]['detection_latency_avg'] for item in sorted_data]
        ax.plot(dates, latencies, 's-', color='#9b59b6',
                linewidth=2, markersize=5)
        ax.axhline(y=3.5, color='#e74c3c', linestyle='--', label='ShakeAlert')
        ax.axhline(y=4.0, color='#f39c12', linestyle='--', label='Google')
        ax.axhline(y=3.0, color='#27ae60', linestyle=':', label='Target')
        ax.set_ylabel('Detection Latency (s)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Chart 3: False Positives
        ax = axes[2]
        fp_rates = [item[1]['false_positive_rate'] for item in sorted_data]
        ax.plot(dates, fp_rates, '^-', color='#e74c3c',
                linewidth=2, markersize=5)
        ax.axhline(y=10.0, color='#c0392b', linestyle='--', label='ShakeAlert')
        ax.axhline(y=8.0, color='#d35400', linestyle='--', label='Google')
        ax.axhline(y=5.0, color='#27ae60', linestyle=':', label='Target')
        ax.set_ylabel('False Positive (%)', fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        chart_path = self.charts_dir / '3month_dashboard.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path

    def generate_all_charts(self, days=90):
        """Generate all run charts"""
        logging.info(f"📊 Generating run charts for last {days} days...")
        data = self.load_performance_data(days)
        if not data:
            logging.warning("No performance data available for charting")
            return []
        logging.info(f"✓ Loaded {len(data)} performance records")
        daily_metrics = self.aggregate_daily_metrics(data)
        logging.info(f"✓ Aggregated into {len(daily_metrics)} days")
        charts = []
        charts.append(self.create_magnitude_accuracy_chart(daily_metrics))
        charts.append(self.create_detection_speed_chart(daily_metrics))
        charts.append(self.create_false_positive_chart(daily_metrics))
        charts.append(self.create_combined_dashboard(daily_metrics))
        logging.info(f"✅ Generated {len(charts)} run charts")
        return charts


# Typical usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    chart_gen = PerformanceCharts('./data')
    charts = chart_gen.generate_all_charts(days=90)
    print(f"Charts generated: {charts}")
