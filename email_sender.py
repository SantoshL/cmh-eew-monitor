"""
Email Sender for Daily Reports
Sends markdown reports via Gmail SMTP
File: email_sender.py
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EmailSender:
    """Send daily performance reports via email"""

    def __init__(self):
        # Get credentials from environment variables (Railway)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv(
            'SENDER_PASSWORD')  # Gmail App Password
        self.recipient_email = os.getenv('RECIPIENT_EMAIL')

        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            raise ValueError(
                "Missing email configuration in environment variables")

    def send_markdown_report(self, markdown_content: str, subject: str = None):
        """Send markdown report as HTML email"""
        if subject is None:
            from datetime import datetime
            subject = f"CMH EEW Daily Report - {datetime.now().strftime('%Y-%m-%d')}"

        # Convert markdown to HTML (simple conversion)
        html_content = self._markdown_to_html(markdown_content)

        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = self.sender_email
        message['To'] = self.recipient_email

        # Attach both plain text and HTML versions
        text_part = MIMEText(markdown_content, 'plain')
        html_part = MIMEText(html_content, 'html')

        message.attach(text_part)
        message.attach(html_part)

        # Send email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)

            logger.info(f"✓ Email sent successfully to {self.recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple markdown to HTML conversion"""

        html = markdown_text

        # Convert headers
        html = html.replace('# ', '<h1>').replace(
            '\n## ', '</h1>\n<h2>').replace('\n### ', '</h2>\n<h3>')

        # Convert tables (simple approach)
        lines = html.split('\n')
        in_table = False
        result = []

        for line in lines:
            if '|' in line and not line.startswith('<!'):
                if not in_table:
                    result.append(
                        '<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%; margin: 20px 0;">')
                    in_table = True

                cells = [cell.strip() for cell in line.split('|')[1:-1]]

                if '---' in line:
                    continue  # Skip separator row

                if len(cells) > 0:
                    row = '<tr>'
                    for cell in cells:
                        if cell.startswith('**'):
                            row += f'<th style="background-color: #3498db; color: white; padding: 12px;">{cell.replace("**", "")}</th>'
                        else:
                            row += f'<td style="padding: 12px; border: 1px solid #ddd;">{cell}</td>'
                    row += '</tr>'
                    result.append(row)
            else:
                if in_table:
                    result.append('</table>')
                    in_table = False
                result.append(line)

        if in_table:
            result.append('</table>')

        html = '\n'.join(result)

        # Wrap in styled HTML
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                table {{
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                tr:hover {{
                    background-color: #e8f4f8;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #ecf0f1;
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """

        return styled_html


# Test code
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sample_markdown = """
# Test Report

## Metrics

| Metric | Value |
|--------|-------|
| Detection Rate | 95% |
| Magnitude Error | ±0.28 |

✅ All systems operational
"""

    try:
        sender = EmailSender()
        print("✓ Email sender initialized")
        print(f"  Will send from: {sender.sender_email}")
        print(f"  Will send to: {sender.recipient_email}")
    except ValueError as e:
        print(f"⚠️  {e}")
        print("\nSet these Railway environment variables:")
        print("  SENDER_EMAIL=your-email@gmail.com")
        print("  SENDER_PASSWORD=your-16-char-app-password")
        print("  RECIPIENT_EMAIL=recipient@example.com")
