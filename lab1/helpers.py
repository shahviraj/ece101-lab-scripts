import matplotlib.pyplot as plt
import numpy as np

message_len = 0

def create_figure():
    plt.figure(figsize=(10, 3))

def set_title(title):
    plt.title(title)

def show_legend():
    plt.legend()

def generate_noise(amp):
    global message_len
    return np.random.normal(0, amp, message_len * 10)

def plot_line(data, label=None, color=None):
    global message_len
    plt.xticks(np.arange(0, message_len+1, 1))
    plt.yticks(np.arange(-10, 10, 0.25))
    plt.xlabel('time (seconds)')
    plt.grid(True)
    if not color:
        color = 'red'
    plt.plot(np.linspace(0, message_len, len(data)), data, label=label, linewidth=2, color=color)   

def plot_step(data, label=None, color=None): 
    global message_len
    if type(data) is str:
        message_len = len(data)
        data = [int(x) for x in data]
    plt.xticks(np.arange(0, message_len+1, 1))
    plt.yticks(np.arange(-10, 10, 0.25))
    plt.xlabel('time (seconds)')
    plt.grid(True)
    if not color:
        color = 'blue'
    plt.step(np.arange(message_len + 1), data + [data[-1]], where='post', label=label, linewidth=3, color=color)   

def set_horizontal_line(y):
    plt.axhline(y, color='black', linestyle='--')

def add_message_and_noise(message, noise):
    return [int(message[i//10]) + noise[i] for i in range(len(noise))]
