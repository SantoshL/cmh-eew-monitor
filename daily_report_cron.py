"""
Daily Report Cron Job
Runs daily to generate and email performance report
File: daily_report_cron.py
"""

from email_sender import EmailSender
from report_generator import ReportGenerator
from performance_tracker import PerformanceTracker
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main cron job execution

    1. Validate recent detections against USGS
    2. Generate markdown report
    3. Email report
    """
    try:
        logger.info("="*70)
        logger.info("STARTING DAILY PERFORMANCE REPORT GENERATION")
        logger.info("="*70)

        # Initialize components
        data_dir = Path('./data')
        tracker = PerformanceTracker(data_dir)
        generator = ReportGenerator(data_dir)
        sender = EmailSender()

        # Step 1: Validate detections
        logger.info("📊 Step 1: Validating detections against USGS...")
        metrics = tracker.validate_detections(days=1)

        if not metrics or 'error' in metrics:
            logger.warning("⚠️  No valid metrics to report")
            # Still send email with status
            report = f"""# CMH EEW Daily Report - {datetime.now().strftime('%B %d, %Y')}

## Status: No New Detections

No earthquake detections in the last 24 hours to validate.

System is operational and monitoring continues.

---
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
"""
        else:
            # Step 2: Generate report
            logger.info("📝 Step 2: Generating markdown report...")
            report = generator.generate_daily_report(metrics)
            generator.save_report(report)

        # Step 3: Send email
        logger.info("📧 Step 3: Sending email report...")
        success = sender.send_markdown_report(report)

        if success:
            logger.info("✅ Daily report sent successfully!")
        else:
            logger.error("❌ Failed to send daily report")
            sys.exit(1)

        logger.info("="*70)
        logger.info("DAILY REPORT COMPLETED SUCCESSFULLY")
        logger.info("="*70)

        # Exit cleanly for Railway cron
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Error in daily report cron: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
