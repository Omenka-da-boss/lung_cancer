import smtplib
from email.mime.text import MIMEText

def test_email():
    sender = os.environ.get('ALERT_EMAIL_SENDER')
    password = os.environ.get('ALERT_EMAIL_PASSWORD')
    recipients = os.environ.get('ALERT_EMAIL_RECIPIENTS', '').split(',')
    if not all([sender, password, recipients]):
        print("Missing email env vars")
        return

    msg = MIMEText("Test email from Evidently")
    msg['Subject'] = "Test"
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print("✅ Test email sent")
    except Exception as e:
        print(f"❌ Email failed: {e}")

test_email()