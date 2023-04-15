from twilio.rest import Client


def send_message_over_text(msg):
    account_sid = 'AC5713b468de92a1d454a052432fb885bd'
    auth_token = '392bc0da2b84a5328acfc65192c58e84'

    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                         body=msg,
                         from_='+18646683468',  # make sure this matches the number in your twilio account
                         to='+19077825400'  # naturally, where you're sending it
                     )

