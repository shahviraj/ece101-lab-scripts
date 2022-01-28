import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

message_len = 0
message_signal_len = 0
sampling_factor = 10

def convert_message_to_signal(message, set=False):
    global message_len, message_signal_len, sampling_factor
    if type(message) is not str:
        raise TypeError('message must be a string')
    message_len = len(message)
    ret = [int(message[i//sampling_factor]) for i in range(message_len * sampling_factor)]
    ret += [ret[-1]]
    message_signal_len = len(ret)
    return np.array(ret, dtype=np.float32)

def create_figure():
    plt.figure(figsize=(12, 5))

def set_title(title):
    plt.title(title)

def show_legend():
    plt.legend()

def generate_noise(amp):
    global message_signal_len
    return np.random.normal(0, amp * 0.607, message_signal_len)

def plot_line(data, label=None, color=None):
    global message_len
    plt.xticks(np.arange(0, message_len+1, 1))
    if max(data) - min(data) > 0.5:
        plt.yticks(np.arange(-10, 10, 0.25))
    plt.ylim(min(data) - 0.5, max(data) + 0.5)
    plt.xlabel('time (seconds)')
    plt.grid(True)
    if not color:
        color = 'red'
    plt.plot(np.linspace(0, message_len, len(data)), data, label=label, linewidth=2, color=color)   

def plot_step(data, label=None, color=None): 
    global message_len, message_signal_len
    plt.xticks(np.arange(0, message_len+1, 1))
    if max(data) - min(data) > 0.5:
        plt.yticks(np.arange(-10, 10, 0.25))
    if max(data) < 1 and color == 'green':
        plt.ylim(-0.1, 1.1)
    plt.xlabel('time (seconds)')
    plt.grid(True)
    if not color:
        color = 'blue'
    plt.step(np.linspace(0, message_len, message_signal_len), data, where='post', label=label, linewidth=3, color=color)   

def set_horizontal_line(y):
    plt.axhline(y, color='black', linestyle='--')

def set_vertical_line(x):
    plt.axvline(x, color='black', linestyle='--')

def add_message_and_noise(message, noise):
    return [int(message[i//10]) + noise[i] for i in range(len(noise))]

def add_image_to_plot(image_path, x, y, zoom=1):
    im = OffsetImage(plt.imread(image_path), zoom=zoom)
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)

def average_signal(signal):
    global sampling_factor, message_len
    ret = np.array([]).astype(np.float32)
    for i in range(message_len):
        ret = np.append(
            ret, 
            np.repeat(np.mean(signal[i*sampling_factor:(i+1)*sampling_factor]), sampling_factor)
        )
    # repeat the last value
    ret = np.append(ret, ret[-1])
    return ret

def show_setup_1(distance):
    add_image_to_plot("smartphone.png", distance, 0, zoom=0.15/np.sqrt(distance))
    add_image_to_plot("wireless-router.png", 0, 0, zoom=0.2/np.sqrt(distance))
    x = np.linspace(-1, distance+1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    Z =  1 / (2 + X**2 + Y**2)
    plt.imshow(Z, extent=extent, cmap=plt.cm.magma)
    plt.xlim(-1, distance+1)
    plt.xticks(np.arange(0, distance+1, 1))
    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.1, 1))
    plt.grid(True)

def set_average_line(signal):
    avg = signal.mean()
    set_horizontal_line(avg)

def show_setup_2(my_wifi_router_coordinates, my_phone_coordinates, neighboring_wifi_routers):
    min_x = min(my_wifi_router_coordinates[0], my_phone_coordinates[0], *[x[0] for x in neighboring_wifi_routers])
    max_x = max(my_wifi_router_coordinates[0], my_phone_coordinates[0], *[x[0] for x in neighboring_wifi_routers])
    min_y = min(my_wifi_router_coordinates[1], my_phone_coordinates[1], *[x[1] for x in neighboring_wifi_routers])
    max_y = max(my_wifi_router_coordinates[1], my_phone_coordinates[1], *[x[1] for x in neighboring_wifi_routers])
    extent = [min_x, max_x, min_y, max_y]
    diag_len = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)

    add_image_to_plot("smartphone.png", my_phone_coordinates[0], my_phone_coordinates[1], zoom=0.15/np.sqrt(diag_len))
    add_image_to_plot("wireless-router.png", my_wifi_router_coordinates[0], my_wifi_router_coordinates[1], zoom=0.2/np.sqrt(diag_len))
    for r in neighboring_wifi_routers:
        add_image_to_plot("wireless-router-evil.png", r[0], r[1], zoom=0.2/np.sqrt(diag_len))
    plt.xticks(np.arange(min_x, max_x+1, 1))
    plt.yticks(np.arange(min_y, max_y+1, 1))
    plt.xlim(min_x-1, max_x+1)
    plt.ylim(min_y-1, max_y+1)
    plt.grid(True)

def dist(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def plot_all_signals(message, my_wifi_router_coordinates, my_phone_coordinates, neighboring_wifi_routers):
    global message_len, sampling_factor
    message_signal = [int(message[i//sampling_factor]) for i in range(message_len * sampling_factor)]
    message_signal += [message_signal[-1]]
    received_signal = message_signal / (dist(my_wifi_router_coordinates, my_phone_coordinates)**2)
    colors = sns.husl_palette(len(neighboring_wifi_routers)+1, l=.5)
    plot_step(received_signal, label="My router's signal", color=colors[0])

    for i, r in enumerate(neighboring_wifi_routers):
        int_message = "".join((np.random.randint(2, size=message_len)).astype(str))
        int_signal = [int(int_message[i//sampling_factor]) for i in range(message_len * sampling_factor)]
        int_signal += [int_signal[-1]]
        int_signal = np.array(int_signal)
        int_signal = int_signal / (dist(my_phone_coordinates, r)**2)
        plot_step(int_signal, color=colors[i+1], label=f"Neighbour {i+1}'s signal")
        show_legend()
        # received_signal += int_signal

def get_received_signal(message, my_wifi_router_coordinates, my_phone_coordinates, neighboring_wifi_routers):
    global message_len, sampling_factor
    message_signal = [int(message[i//sampling_factor]) for i in range(message_len * sampling_factor)]
    message_signal += [message_signal[-1]]
    received_signal = message_signal / (dist(my_wifi_router_coordinates, my_phone_coordinates)**2)

    for i, r in enumerate(neighboring_wifi_routers):
        int_message = "".join((np.random.randint(2, size=message_len)).astype(str))
        int_signal = [int(int_message[i//sampling_factor]) for i in range(message_len * sampling_factor)]
        int_signal += [int_signal[-1]]
        int_signal = np.array(int_signal)
        int_signal = int_signal / (dist(my_phone_coordinates, r)**2)
        received_signal += int_signal
    return received_signal

def show_coverage(my_wifi_router_coordinates, my_phone_coordinates, neighboring_wifi_routers):
    min_x = min(my_wifi_router_coordinates[0], my_phone_coordinates[0], *[x[0] for x in neighboring_wifi_routers])
    max_x = max(my_wifi_router_coordinates[0], my_phone_coordinates[0], *[x[0] for x in neighboring_wifi_routers])
    min_y = min(my_wifi_router_coordinates[1], my_phone_coordinates[1], *[x[1] for x in neighboring_wifi_routers])
    max_y = max(my_wifi_router_coordinates[1], my_phone_coordinates[1], *[x[1] for x in neighboring_wifi_routers])
    # extent = [min_x-1, max_x+1, min_y-1, max_y+1]

    x = np.arange(min_x-1, max_x+1, 0.03)
    y = np.arange(min_y-1, max_y+1, 0.03)
    Z = np.zeros((len(y), len(x) ))
    extent = [min_x-1, max_x+1, min_y-1, max_y+1]
    # iterate over all combinations of x and y
    for i in range(len(x)):
        for j in range(len(y)):
            if (i, j) in tuple(neighboring_wifi_routers) or (i, j) == tuple(my_wifi_router_coordinates):
                Z[-j, i] = 1
            den = (dist(my_wifi_router_coordinates, [x[i], y[j]])**2)
            if np.abs(den) < 1:
                s = 1
            else:
                s = 1 / den

            na = 0
            for r in neighboring_wifi_routers:
                den = (dist(r, [x[i], y[j]])**2)
                if np.abs(den) < 1:
                    n = 100
                else:
                    n = 1 / den
                na += n
            r = np.random.normal(0, 0.02)
            # r = 0.05
            Z[-j, i] = s/(na + r)
    show_setup_2(my_wifi_router_coordinates, my_phone_coordinates, neighboring_wifi_routers)
    plt.imshow(Z > 0.7, extent=extent, cmap="gray", aspect='auto')

    plt.xlim(min_x-1, max_x+1)
    plt.xticks(np.arange(min_x-1, max_x+1, 1.0))
    plt.ylim(min_y-1, max_y+1)
    plt.yticks(np.arange(min_y-1, max_y+1, 1.0))
    plt.grid(True)