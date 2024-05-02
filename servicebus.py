from azure.servicebus import ServiceBusClient, ServiceBusMessage

# Define the connection string and the queue name
connection_str = '***'
queue_name = 'location'

servicebus_client = ServiceBusClient.from_connection_string(conn_str=connection_str, logging_enable=True)
with servicebus_client:
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:
        message = ServiceBusMessage("{'message': 'This is a test message'}", content_type="application/json")
        sender.send_messages(message)

print("Message sent successfully.")
