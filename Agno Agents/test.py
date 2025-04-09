import json
import re
from datetime import datetime
import httpx
from typing import List, Dict, Any, Optional
import os
from textwrap import dedent
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
email_address = os.environ.get('EMAIL_ADDRESS')
email_password = os.environ.get('EMAIL_PASSWORD')
smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
smtp_port = int(os.environ.get('SMTP_PORT', 587))
imap_server = os.environ.get('IMAP_SERVER', 'imap.gmail.com')

# Set up environment variables
os.environ['EMAIL_ADDRESS'] = email_address
os.environ['EMAIL_PASSWORD'] = email_password
os.environ['SMTP_SERVER'] = smtp_server
os.environ['SMTP_PORT'] = str(smtp_port)
os.environ['IMAP_SERVER'] = imap_server

#  Authentication Fixes
def validate_email_settings():
    """Check required environment variables"""
    missing = []
    if not email_address:
        missing.append('EMAIL_ADDRESS')
    if not email_password:
        missing.append('EMAIL_PASSWORD')
    if missing:
        raise ValueError(f"Missing email credentials: {', '.join(missing)}")

# Add this test code
if __name__ == "__main__":
    try:
        validate_email_settings()
        print("✅ Email credentials configured properly")
        
        # Test SMTP connection
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            print("✅ SMTP authentication successful")
            
        # Test IMAP connection
        with imaplib.IMAP4_SSL(imap_server) as mail:
            mail.login(email_address, email_password)
            print("✅ IMAP authentication successful")
            
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("Check: 1) .env file 2) App passwords 3) Server settings")