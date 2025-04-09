import json
import re
from datetime import datetime
import httpx
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
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

from agno.agent import Agent
from agno.models.groq import Groq

groq_api_key = os.environ.get('GROQ_API_KEY')
# Set the environment variables
os.environ['GROQ_API_KEY'] = groq_api_key

# Dictionary to store draft emails
draft_emails = {}
# Persistent Storage Configuration
DRAFTS_FILE = Path("email_drafts.json")

def load_drafts() -> Dict[str, Any]:
    """Load drafts from JSON file with error handling"""
    try:
        if DRAFTS_FILE.exists():
            with open(DRAFTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error loading drafts: Corrupted file - {e}")
        return {}
    except Exception as e:
        print(f"Error loading drafts: {e}")
        return {}

def save_drafts(drafts: Dict[str, Any]) -> None:
    """Save drafts to JSON file with atomic write"""
    try:
        temp_file = DRAFTS_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(drafts, f, indent=2, default=str)
        temp_file.replace(DRAFTS_FILE)
    except Exception as e:
        print(f"Error saving drafts: {e}")

# Initialize draft storage
draft_emails = load_drafts()

def compose_email(to: str, subject: str, body: str, cc: Optional[str] = None, bcc: Optional[str] = None) -> str:
    """
    Compose a new email and save it as a draft.
    
    Args:
        to (str): Recipient email address(es), comma-separated for multiple
        subject (str): Email subject
        body (str): Email body content
        cc (Optional[str]): CC email address(es), comma-separated for multiple
        bcc (Optional[str]): BCC email address(es), comma-separated for multiple
        
    Returns:
        str: JSON string containing the draft email details and a draft ID
    """
    draft_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    draft_email = {
        "draft_id": draft_id,
        "to": to,
        "subject": subject,
        "body": body,
        "cc": cc or "",
        "bcc": bcc or "",
        "created_at": datetime.now().isoformat()
    }
    
    # Save to our draft storage
    draft_emails[draft_id] = draft_email
    save_drafts(draft_emails)
    
    return json.dumps({
        "status": "success",
        "message": "Email draft created successfully",
        "draft_id": draft_id,
        "email": draft_email
    })

def preview_email(draft_id: str) -> str:
    """
    Preview an email draft.
    
    Args:
        draft_id (str): ID of the draft email to preview
        
    Returns:
        str: JSON string containing the draft email details for preview
    """
    if draft_id not in draft_emails:
        return json.dumps({
            "status": "error",
            "message": f"Draft with ID {draft_id} not found"
        })
    
    return load_drafts()[draft_id]
    # Return the draft email details

def list_drafts() -> str:
    """
    List all saved draft emails.
    
    Returns:
        str: JSON string containing all saved draft emails
    """
    return json.dumps({
        "status": "success",
        "count": len(draft_emails),
        "drafts": list(draft_emails.values())
    })

def send_email(draft_id: str) -> str:
    """
    Send an email from a saved draft.
    
    Args:
        draft_id (str): ID of the draft email to send
        
    Returns:
        str: JSON string containing the status of the email sending operation
    """
    if draft_id not in draft_emails:
        return json.dumps({
            "status": "error",
            "message": f"Draft with ID {draft_id} not found"
        })
    
    draft = draft_emails[draft_id]
    
    # Create MIME message
    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = draft['to']
    msg['Subject'] = draft['subject']
    
    if draft['cc']:
        msg['Cc'] = draft['cc']
    
    # Attach body
    msg.attach(MIMEText(draft['body'], 'plain'))
    
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_address, email_password)
        
        # Calculate all recipients
        all_recipients = draft['to'].split(',')
        if draft['cc']:
            all_recipients.extend(draft['cc'].split(','))
        if draft['bcc']:
            all_recipients.extend(draft['bcc'].split(','))
        
        # Send email
        text = msg.as_string()
        server.sendmail(email_address, all_recipients, text)
        server.quit()
        
        # Remove from drafts after sending
        del draft_emails[draft_id]
        
        return json.dumps({
            "status": "success",
            "message": "Email sent successfully",
            "to": draft['to'],
            "subject": draft['subject']
        })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to send email: {str(e)}"
        })

def update_draft(draft_id: str, to: Optional[str] = None, subject: Optional[str] = None, 
                body: Optional[str] = None, cc: Optional[str] = None, bcc: Optional[str] = None) -> str:
    """
    Update an existing email draft.
    
    Args:
        draft_id (str): ID of the draft email to update
        to (Optional[str]): New recipient email address(es)
        subject (Optional[str]): New email subject
        body (Optional[str]): New email body content
        cc (Optional[str]): New CC email address(es)
        bcc (Optional[str]): New BCC email address(es)
        
    Returns:
        str: JSON string containing the updated draft email details
    """
    if draft_id not in draft_emails:
        return json.dumps({
            "status": "error",
            "message": f"Draft with ID {draft_id} not found"
        })
    
    # Update only the provided fields
    if to is not None:
        draft_emails[draft_id]['to'] = to
    if subject is not None:
        draft_emails[draft_id]['subject'] = subject
    if body is not None:
        draft_emails[draft_id]['body'] = body
    if cc is not None:
        draft_emails[draft_id]['cc'] = cc
    if bcc is not None:
        draft_emails[draft_id]['bcc'] = bcc
    
    draft_emails[draft_id]['updated_at'] = datetime.now().isoformat()
    save_drafts(draft_emails)
    
    return json.dumps({
        "status": "success",
        "message": "Draft updated successfully",
        "email": draft_emails[draft_id]
    })

def delete_draft(draft_id: str) -> str:
    """
    Delete an email draft.
    
    Args:
        draft_id (str): ID of the draft email to delete
        
    Returns:
        str: JSON string containing the status of the deletion operation
    """
    if draft_id not in draft_emails:
        return json.dumps({
            "status": "error",
            "message": f"Draft with ID {draft_id} not found"
        })
    
    del draft_emails[draft_id]
    
    
    return json.dumps({
        "status": "success",
        "message": f"Draft with ID {draft_id} deleted successfully"
    })

def read_inbox(limit: int = 5, unread_only: bool = False) -> str:
    """
    Read emails from the inbox.
    
    Args:
        limit (int): Maximum number of emails to retrieve
        unread_only (bool): Whether to retrieve only unread emails
        
    Returns:
        str: JSON string containing the retrieved emails
    """
    try:
        # Connect to IMAP server
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_address, email_password)
        mail.select('inbox')
        
        # Search for emails
        search_criteria = 'UNSEEN' if unread_only else 'ALL'
        status, messages = mail.search(None, search_criteria)
        
        email_ids = messages[0].split()
        
        # Get the latest emails (up to the limit)
        start_idx = max(0, len(email_ids) - limit)
        latest_email_ids = email_ids[start_idx:]
        
        emails = []
        
        for e_id in latest_email_ids:
            status, msg_data = mail.fetch(e_id, '(RFC822)')
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Decode subject
                    subject, encoding = decode_header(msg['Subject'])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or 'utf-8')
                    
                    # Get sender
                    from_header, encoding = decode_header(msg['From'])[0]
                    if isinstance(from_header, bytes):
                        from_header = from_header.decode(encoding or 'utf-8')
                    
                    # Get date
                    date = msg['Date']
                    
                    # Get body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get('Content-Disposition'))
                            
                            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    
                    emails.append({
                        "id": e_id.decode(),
                        "from": from_header,
                        "subject": subject,
                        "date": date,
                        "body": body[:500] + ("..." if len(body) > 500 else "")  # Truncate long bodies
                    })
        
        mail.close()
        mail.logout()
        
        return json.dumps({
            "status": "success",
            "count": len(emails),
            "emails": emails
        })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to read emails: {str(e)}"
        })

def search_emails(query: str, folder: str = 'inbox', limit: int = 5) -> str:
    """
    Search emails in a specific folder.
    
    Args:
        query (str): Search query
        folder (str): Email folder to search in
        limit (int): Maximum number of emails to retrieve
        
    Returns:
        str: JSON string containing the search results
    """
    try:
        # Connect to IMAP server
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_address, email_password)
        mail.select(folder)
        
        # Search for emails matching the query
        status, messages = mail.search(None, f'SUBJECT "{query}" OR BODY "{query}" OR FROM "{query}" OR TO "{query}"')
        
        email_ids = messages[0].split()
        
        # Get the latest emails (up to the limit)
        start_idx = max(0, len(email_ids) - limit)
        latest_email_ids = email_ids[start_idx:]
        
        emails = []
        
        for e_id in latest_email_ids:
            status, msg_data = mail.fetch(e_id, '(RFC822)')
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Decode subject
                    subject, encoding = decode_header(msg['Subject'])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or 'utf-8')
                    
                    # Get sender
                    from_header, encoding = decode_header(msg['From'])[0]
                    if isinstance(from_header, bytes):
                        from_header = from_header.decode(encoding or 'utf-8')
                    
                    # Get date
                    date = msg['Date']
                    
                    # Get body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get('Content-Disposition'))
                            
                            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    
                    emails.append({
                        "id": e_id.decode(),
                        "from": from_header,
                        "subject": subject,
                        "date": date,
                        "body": body[:500] + ("..." if len(body) > 500 else "")  # Truncate long bodies
                    })
        
        mail.close()
        mail.logout()
        
        return json.dumps({
            "status": "success",
            "count": len(emails),
            "query": query,
            "emails": emails
        })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to search emails: {str(e)}"
        })

# Create an Email Assistant Agent
email_agent = Agent(
    model=Groq(
        id='meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    instructions=dedent("""\
    You are MailPro, an intelligent email assistant designed to help manage and optimize email workflows! 📧
    
    Your personality combines:
    - Professional efficiency with a friendly tone
    - Email management expertise with clear communication
    - Organizational skills with intuitive suggestions
    
    Capabilities:
    - Compose thoughtfully crafted emails that match the user's intent and tone
    - Preview and edit drafts before sending
    - Read and summarize inbox content
    - Organize and manage email drafts
    - Search through emails to find specific information
    
    Writing style:
    - Adapt to formal or casual tones based on the email context
    - Create clear, concise subject lines
    - Structure email bodies for optimal readability
    - Use appropriate greetings and closings
    - Maintain professional language while being personable
    
    Your response approach:
    1. Listen carefully to the user's email needs
    2. Ask clarifying questions when necessary
    3. Suggest improvements to draft emails
    4. Provide organized summaries of inbox content
    5. Efficiently handle email management tasks
    
    Remember: Your goal is to make email management effortless while ensuring all communications are professional, clear, and effective!
"""),
    tools=[
        compose_email,
        preview_email,
        list_drafts,
        send_email,
        update_draft,
        delete_draft,
        read_inbox,
        search_emails
    ],
    show_tool_calls=True,
    markdown=True,
)

# Test the agent with a sample query
if __name__ == "__main__":
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        email_agent.print_response(user_input, stream=True)