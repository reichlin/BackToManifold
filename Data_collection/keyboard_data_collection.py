from env import YumiFullMoveEnv
from pynput import keyboard
import numpy as np


def on_press(key):
    try:
        global key_pressed
        key_pressed = key.char
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press)
listener.start()
vel = 0.04
env = YumiFullMoveEnv()
key_pressed = ''
while key_pressed != 'z':
    if key_pressed != '':
        action = np.zeros(7)
        if key_pressed == 's':
            action[0] = vel
        elif key_pressed == 'w':
            action[0] = -vel
        elif key_pressed == 'd':
            action[1] = vel
        elif key_pressed == 'a':
            action[1] = -vel
        elif key_pressed == 'q':
            action[2] = vel*2
        elif key_pressed == 'e':
            action[2] = -vel*2
        env.step(action)

env.release()





