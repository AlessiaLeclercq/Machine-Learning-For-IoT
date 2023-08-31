import uuid
import json
import psutil 
import paho.mqtt.client as mqtt 

from time import sleep, time

client = mqtt.Client() 
MAC_ADDRESS = hex(uuid.getnode())

#Pahoo mandatory method
def on_connect(client, userdata, flags, rc):
    print(f'Connected with result code {str(rc)}')

#Bind the on_connect method to the client
client.on_connect = on_connect

#Connects to the broker 
client.connect('mqtt.eclipseprojects.io', 1883)


while True:
    output_dict = {
        "mac_address": MAC_ADDRESS,
        "timestamp" : int(time()*1000),
        "battery_level" : psutil.sensors_battery().percent, 
        "power_plugged" : int(psutil.sensors_battery().power_plugged)
    }

    output_string = json.dumps(output_dict)
    print(output_string)


    client.publish('s291871', output_string)
    sleep(1)