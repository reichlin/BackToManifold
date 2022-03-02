import os
import socket
import cv2
import time
from datetime import datetime
import numpy as np
import PIL
# todo make it gym env


class YumiFullMoveEnv:
    def __init__(self, counter=None, dagger=False):
        self.port = 11997
        self.host = '130.237.218.6'
        self.action_dim = 7 #3  # todo change it to gym action space
        self.state_dim = 7
        self.step_delay = 0.1  # seconds
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.image_directory = 'files/data/pick/'
        now = datetime.now()
        self.save_data = (counter is not None)

        if dagger:
            self.image_directory = '../files/data/pick/dataset_IL_dagger/' + str(counter)
            if not os.path.exists(self.image_directory):
                os.mkdir(self.image_directory)
        else:
            if counter is not None:
                self.image_directory += now.date().__str__() + '_'
                self.image_directory += now.time().hour.__str__() + '_'
                self.image_directory += now.time().minute.__str__()
                self.image_directory += '_' + str(counter)
                self.image_directory = 'files/data/pick/dataset_IL_raw/'+str(counter)
                if not os.path.exists(self.image_directory):
                    os.mkdir(self.image_directory)

        self.step_counter = 0
        self.max_steps = 100000
        self.save_current_frame()
        time.sleep(0.1)
        self.save_current_frame()
        self.actions = np.zeros((self.max_steps, self.action_dim))
        self.states = np.zeros((self.max_steps, self.state_dim))

        self.ready = True

    def step(self, action, deploy=False):

        state = self.transmit(action)
        self.states[self.step_counter] = state
        self.actions[self.step_counter] = action

        if self.save_data:
            frame = self.save_current_frame()
        else:
            frame = None

        self.ready = False

        if deploy:
            for t in range(3):
                _, _ = self.cap.read()
        else:
            for t in range(3):
                time.sleep(self.step_delay/3.)
                _, _ = self.cap.read()

        self.ready = True

        self.step_counter += 1
        return frame, state

    def get_state(self):
        ret, frame = self.cap.read()
        state = self.transmit_state()
        return frame, state

    def get_frame(self):
        ret, img = self.cap.read()
        if not ret:
            print()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        return img

    def transmit(self, action):
        str_action = ''
        for i in range(self.action_dim):
            str_action += '{:.4f} '.format(action[i])
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(str_action.encode('ascii'))
        obs = s.recv(100)
        s.close()
        return np.fromstring(obs, sep=' ')

    def transmit_state(self):
        msg = ' '
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.send(msg.encode('ascii'))
        obs = s.recv(100)
        s.close()
        return np.fromstring(obs, sep=' ')

    def save_current_frame(self):
        ret, frame = self.cap.read()
        cv2.imwrite(self.image_directory + '/img{:04d}.png'.format(self.step_counter), frame)
        return frame

    def release(self):
        if self.save_data:
            np.save(self.image_directory + '/actions.npy', self.actions[:self.step_counter])
            np.save(self.image_directory + '/states.npy', self.states[:self.step_counter])
            print("data successfully saved")
        self.cap.release()
        cv2.destroyAllWindows()


