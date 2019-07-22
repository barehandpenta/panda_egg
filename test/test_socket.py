import socketio as socket
import time
sio = socket.Client()
@sio.event
def message(data):
    pass
@sio.on('reply')
def on_message(data):
    print('I received a message!')
    print(data)
@sio.event
def connect():
    print("Sucessfully connected!")
@sio.event
def disconnect():
    print("I'm disconnected!")
sio.connect('http://18.179.14.225:3000')
print('my sid is', sio.sid)
sio.emit('message', "0")

# while True:
#     sio.emit('pos-change', "85")
#     time.sleep(3)
#     sio.emit('pos-change', "180")
#     time.sleep(3)
while True:
    pos = input("Input position:")
    sio.emit('pos-change', str(pos))

