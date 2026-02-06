import threading
import queue
import numpy as np
import sounddevice as sd
import tkinter as tk
import time
import os
import random
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
import os

#globals
dac = 44100
CHANNELS = 1
audio_queue = queue.Queue(maxsize=5)
lock = threading.Lock()
stream = None
dt = 0.05  # aggiornamento globale

# comb filter globals
comb_buf = np.zeros(8192, dtype=np.float32)
comb_pos = 0.0
comb_con = 0.5 
comb_fb = 0.8   
comb_min = 10
comb_max = 50

#feedback
delay = 64   
fb_amt = 0.2
fb_buf_direct = []
fb_buf_acc = []

#params
rate = 44100
density = 100
level = 0.5
chance = 0.01
acc_decay = 0.85
fb_amt = 0.2
acc_size = 8192
blocksize = 1024
speed = 1.0
alpha = 0.5
cutoff = 0.1
smooth = 0.2
dist = 0.7

# keys
keys_down = {} 
pressed_keys = set()  
audio_params = {}

# acc/dir active
acc_active = False
direct_active = True
accumulator = np.zeros(acc_size, dtype=np.float32)
direct_memory = np.zeros(4096, dtype=np.float32)

# perc engine
perc_active = True
perc_mix = 0.4 # ex 0.5      
perc_gain = 0.95  

# funzioni base
def sigmoid(x, k=6):
    return 1 / (1 + np.exp(-k*(x - 0.5)))

def wrap(x, lo, hi):
    r = hi - lo
    return lo + ((x - lo) % r)

def bias(x, pow=2):
    return 1 - (1 - x)**pow

def lag(current, target, tau, dt):
    tau = max(tau, 1e-4)
    alpha = 1 - np.exp(-dt / tau)
    return current + (target - current) * alpha

#def curve(b, min=0.0, max=1.0, fl=1.0, s=3.0, fr=3.0):
    #x = np.sin(b * 2 * np.pi * fr) + np.cos(b * 2 * np.pi * (fr * 0.5))
    #x = np.tanh(x * s)
    #x = np.sign(x) * (np.abs(x) ** fl)
    #y = min + (x + 1) * 0.5 * (max - min)
    #return y

def linlin(x, inmin, inmax, outmin, outmax):
    if inmax - inmin == 0:
        return outmin
    t = (x - inmin) / (inmax - inmin)
    t = np.clip(t, 0.0, 1.0)
    return outmin + t * (outmax - outmin)

# lowpass filtering
def lowpass(x, factor=0.1):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = factor * x[i] + (1-factor) * y[i-1]
    return y

# hipass filtering
def hipass(x, alpha=0.02):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y[i-1] + x[i] - x[i-1])
    return y

# comb filtering
def comb(signal, buffer, read_pos, map_val=0.5, fb=0.7, min_delay=50, max_delay=500):

    signal = np.asarray(signal, dtype=np.float32).flatten()
    buf_len = len(buffer)
    out = np.zeros_like(signal, dtype=np.float32)

    delay = int(linlin(map_val, 0.0, 1.0, min_delay, max_delay))

    for i, s in enumerate(signal):
        read_idx = int((read_pos - delay) % buf_len)
        delayed = buffer[read_idx]

        out[i] = s + delayed * fb

        buffer[int(read_pos % buf_len)] = s + delayed * fb

        read_pos += 1

    return out, read_pos

# stutter freezer
def stutter(signal, chance=0.03, block_size=64):
    out = signal.copy()
    n = len(signal)

    for start in range(0, n, block_size):
        if np.random.rand() < chance:
            end = min(start + block_size, n)
            seg = out[start:end]

            if np.random.rand() < 0.5:
                # reverse 
                out[start:end] = seg[::-1]
            else:
                # freeze
                out[start:end] = seg[0]

    return out

# xorsig
def xorsig(
    sig,
    pattern,
    amt=1.0,
    bits=8
):     
    amt = np.clip(amt, 0.0, 1.0)

    if bits == 8:
        # pattern
        if not isinstance(pattern, int):
            pattern = int(np.clip(pattern, 0.0, 1.0) * 255)
        else:
            pattern = pattern & 0xFF

        # >uint8
        sig_i = np.clip((sig + 1.0) * 127.5, 0, 255).astype(np.uint8)

        # xor
        xor_i = np.bitwise_xor(sig_i, pattern)

        # >float
        xor_sig = (xor_i.astype(np.float32) / 127.5) - 1.0

    elif bits == 16:
        if not isinstance(pattern, int):
            pattern = int(np.clip(pattern, 0.0, 1.0) * 65535)
        else:
            pattern = pattern & 0xFFFF

        sig_i = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
        xor_i = np.bitwise_xor(sig_i, pattern)
        xor_sig = xor_i.astype(np.float32) / 32767

    else:
        raise ValueError("bits must be 8 or 16")
    # mix 
    return sig * (1.0 - amt) + xor_sig * amt

# xorcon
#def xorcon(val, ctrl=0.5, amt=1.0, bits=16):
    
    # clamp
    #val = np.clip(val, 0.0, 1.0)
    #ctrl  = np.clip(ctrl, 0.0, 1.0)
    #amt = np.clip(amt, 0.0, 1.0)

    #maxv = (1 << bits) - 1

    #vi = int(val * maxv)
    #pattern = int(ctrl * maxv)

    # xor
    #xi = vi ^ pattern

    # mix 
    #xi = int(vi * (1 - amt) + xi * amt)

    #return xi / maxv

# rand geometry
def rand_geometry():
    if perc_pool.raw is None:
        return
    L = len(perc_pool.raw)

    # loop size 
    perc_pool.loop_size = random.randint(max(32, L//128), max(128, L//4))
    
    # loop start 
    perc_pool.loop_start = random.randint(0, max(0, L - perc_pool.loop_size))
    
    # ptr random 
    perc_pool.ptr = perc_pool.loop_start + random.randint(0, perc_pool.loop_size-1)
    perc_pool.ptr = int(perc_pool.ptr)

    log(f"[RYTHM] geometry start={perc_pool.loop_start} size={perc_pool.loop_size} ptr={perc_pool.ptr}")

# rand morphology
def rand_morphology():
    perc_pool.density = random.uniform(0.01, 0.2)
    perc_pool.reverse = random.random() < 0.3
    perc_pool.jitter = random.uniform(0.0, 1.5)
    perc_pool.ptr = int(perc_pool.ptr)

    log(f"[PERC] morph density={perc_pool.density:.2f} rev={perc_pool.reverse}")

# class state
class State:
    # input 
    accumulator = 0.0
    direct = 0.0
    perc = 0.0

    # prev values
    _prev = {
        "acc": 0.0,
        "dir": 0.0,
        "perc": 0.0
    }

    # raw 
    _e = {
        "acc": 0.0,
        "dir": 0.0,
        "perc": 0.0
    }

    # lagged 
    _lagged = {
        "acc": 0.0,
        "dir": 0.0,
        "perc": 0.0,
        "sum": 0.0
    }

    # lag t
    _lag_time = {
        "acc": 0.01,
        "dir": 0.001,
        "perc": 0.01,
        "sum": 0.3
    }

    # gain 
    _energy_gain = {
        "acc": 5.0,
        "dir": 1.0,
        "perc": 3.0,
        "sum": 1.0
    }

    @classmethod
    def _partial_energy(cls, x, key):

        prev = cls._prev[key]
        dx = abs(x - prev)
        cls._prev[key] = x

        e = cls._e[key]
        e = max(dx * 10.0, e * 0.99)
        e = np.clip(e, 0.0, 1.0)
        cls._e[key] = e

        # sigmoid
        return sigmoid(e, 8)

    @classmethod
    def _apply_lag(cls, key, x):

        t = cls._lag_time[key]
        alpha = 1.0 - np.exp(-dt / max(t, 1e-4))

        y = cls._lagged[key]
        y += (x - y) * alpha
        cls._lagged[key] = y

        return y

    @classmethod
    def energy(cls):

        # raw 
        e_acc  = cls._partial_energy(cls.accumulator, "acc")
        e_dir  = cls._partial_energy(cls.direct, "dir")
        e_perc = cls._partial_energy(cls.perc, "perc")

        # lag
        e_acc  = cls._apply_lag("acc", e_acc)
        e_dir  = cls._apply_lag("dir", e_dir)
        e_perc = cls._apply_lag("perc", e_perc)

        # gain
        e_acc  *= cls._energy_gain["acc"]
        e_dir  *= cls._energy_gain["dir"]
        e_perc *= cls._energy_gain["perc"]

        # clip
        e_acc  = np.clip(e_acc, 0.0, 1.0)
        e_dir  = np.clip(e_dir, 0.0, 1.0)
        e_perc = np.clip(e_perc, 0.0, 1.0)

        # sum
        e_sum = np.clip(
            e_acc * 1.0 +
            e_dir * 0.5 +
            e_perc * 0.5,
            0.0, 1.0
        )

        # applying gain + lag
        e_sum = cls._apply_lag("sum", e_sum)
        e_sum *= cls._energy_gain["sum"]
        e_sum = np.clip(e_sum, 0.0, 1.0)

        return np.array([e_acc, e_dir, e_perc, e_sum], dtype=np.float32)

def debug_energy():
    while True:
        e = State.energy()  
        print(f"[DEBUG ENERGY] acc={e[0]:.3f}, dir={e[1]:.3f}, perc={e[2]:.3f}, sum={e[3]:.3f}")
        time.sleep(1.0)  # 1 sec

threading.Thread(target=debug_energy, daemon=True).start()

# key params range
param_ranges = {
    "blocksize": (8192, 16384),
    "rate": (8000, 88200),
    "level": (0.01, 1.0),
    "acc_decay": (0.1, 0.995),
    "fb_amt": (0.1, 0.5),
    "cutoff": (0.09, 0.4),
    "speed": (-6.0, 6.0),
    "alpha": (0.001, 1.0),
    "delay": (64, 1024),
    "dist": (0.5, 1.0),
    "smooth": (0.2, 0.5),
    "comb_con": (0.0, 1.0),       
    "comb_fb": (0.1, 0.6),      
    "comb_min": (5, 30),       
    "comb_max": (30, 80),
}

# mapping
key_param_map = {
    "b": "blocksize",
    "r": "rate",
    "l": "level",
    "y": "acc_decay",
    "f": "fb_amt",
    "c": "cutoff",
    "s": "speed",
    "a": "alpha",
    "d": "delay",
    "w": "smooth",
    "x": "dist",
    "u": "comb_con",   
    "i": "comb_fb",    
    #"o": "comb_min",    
    #"p": "comb_max"  
}

# random params
random_params = [
    "level",
    "acc_decay",
    "fb_amt",
    "cutoff",
    "speed",
    "alpha",
    "smooth",
    "dist",
]

with lock:
    audio_params = {p: globals()[p] for p in param_ranges}

# on key
def on_key(event):
    key = event.keysym.lower()

    with lock:
        keys_down[key] = True
        pressed_keys.add(key)

def on_key_release(event):
    key = event.keysym.lower()
    with lock:
        keys_down.pop(key, None)
        pressed_keys.discard(key)

# random all
def random_all():
    global accumulator, direct_memory, fb_buf_acc, fb_buf_direct

    limits = {
        "level": (0.01, 0.99),
        "acc_decay": (0.2, 0.95),
        "fb_amt": (0.1, 0.5),
        "cutoff": (0.1, 0.4),
        "speed": (-6.0, 6.0),
        "alpha": (0.001, 0.8),
        "smooth": (0.2, 0.5),
        "dist": (0.5, 1.0)
    }

    with lock:  
        for p in random_params:
            lo, hi = limits.get(p, param_ranges[p])
            val = random.uniform(lo, hi)
            audio_params[p] = val

        # buf reset 
        accumulator[:] = 0
        direct_memory[:] = 0
        fb_buf_acc.clear()
        fb_buf_direct.clear()

    # log 
    root.after(0, lambda: log("[RANDOM] buffer reset"))

# update params
def update_params_thread():
    int_params = {
        "delay", 
        #"comb_min", "comb_max",
        "rate", "acc_size", "blocksize"
    }

    lag_factor = 0.2
    rand_factor = 0.5  

    while True:
        with lock:
            current_keys = list(keys_down.keys())
            current_pressed = set(pressed_keys)

        for key in current_keys:
            if key not in key_param_map:
                continue

            param = key_param_map[key]
            lo, hi = param_ranges[param]

            # functions
            skip_lag = False
            if "1" in current_pressed:
                val_lo = (lo + hi) * rand_factor
                val_hi = hi
                skip_lag = True
            elif "2" in current_pressed:
                val_lo = lo
                val_hi = (lo + hi) * rand_factor
                skip_lag = True
            else:
                val_lo = lo
                val_hi = hi

            # safe range
            if param in int_params:
                val_lo = int(val_lo)
                val_hi = int(val_hi)
                val_lo, val_hi = min(val_lo, val_hi), max(val_lo, val_hi)
                new_val = random.randint(val_lo, val_hi)
            else:
                raw_val = random.uniform(val_lo, val_hi)
                if skip_lag:
                    new_val = raw_val  # no lag
                else:
                    cur = audio_params[param]
                    new_val = np.clip(cur + (raw_val - cur) * lag_factor, lo, hi)

            # assignment
            with lock:
                audio_params[param] = new_val

            # log
            root.after(
                0,
                lambda p=param, v=new_val: log(f"[PARAM] {p} = {v}")
            )

        time.sleep(dt)

# key functions
def function_keys_thread():
    while True:
        with lock:
            current_keys = dict(keys_down)  

        for key in current_keys:
            if key == "n":
                 pool_source.next_file()
                 log(f"[POOL] next file")
            elif key == "ù":
                 pool_source.stop()
                 log(f"[POOL] stop")
            elif key == "0":
                 random_all()
                 keys_down.pop("0", None)
                 pressed_keys.discard("0")
            elif key == "9":
                 perc_pool.next_file()
                 log("[PERC] next file")
            elif key == "7":
                 rand_geometry()
            elif key == "8":
                 rand_morphology()
 
        time.sleep(0.05) 

threading.Thread(target=update_params_thread, daemon=True).start()
threading.Thread(target=function_keys_thread, daemon=True).start()

# AUDIO STREAM
def recreate_stream(bs):
    global stream, blocksize, max_blocksize
    if stream:
        stream.stop()
        stream.close()
        time.sleep(0.05)
    blocksize = int(np.clip(bs, 128, acc_size))
    max_blocksize = 4 * blocksize
    stream = sd.OutputStream(
        samplerate=dac,
        blocksize=blocksize,
        channels=CHANNELS,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()

# pool 1
class PoolByteSource:
    def __init__(self):
        self.raw = None
        self.active = False
        self.ptr = 0
        self.current_file = None

    def load(self, path):
        if not os.path.isfile(path):
            print("file non valido")
            return
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            print("file vuoto")
            return
        norm_audio = (raw.astype(np.float32)-128)/128
        with lock:
            self.raw = norm_audio
            self.ptr = 0
            self.active = True
            self.current_file = path
        print(f">>> audio pool: {os.path.basename(path)}")

    def next_file(self):
        if not self.current_file:
            return
        directory = os.path.dirname(self.current_file)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        if not files: return
        next_path = os.path.join(directory, random.choice(files))
        self.load(next_path)

    def stop(self):
        with lock:
            self.active = False
            self.raw = None

    def next_byte(self):
        with lock:
            if not self.active or self.raw is None: return None
            b = int((self.raw[self.ptr % len(self.raw)]+1.0)*127.5)
            self.ptr += 1
            return b

    def next_chunk(self, size):
        with lock:
            if not self.active or self.raw is None: return None
            idxs = (np.arange(size) + self.ptr) % len(self.raw)
            chunk = self.raw[idxs]
            self.ptr += size
            max_c = 8192  
            if len(chunk) > max_c:
               chunk = chunk[:max_c]
            return chunk.astype(np.float32)

pool_source = PoolByteSource()

# pool 2
class PercussivePool:
    def __init__(self):
        self.raw = None
        self.ptr = 0
        self.active = False
        self.current_file = None

        self.loop_start = 0
        self.loop_size = 256

        # morphology
        self.density = 0.8
        self.reverse = False
        self.jitter = 0.0

    def load(self, path):
        if not os.path.isfile(path):
            return
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return
        self.raw = (raw.astype(np.float32) - 128) / 128
        self.ptr = 0
        self.ptr = int(self.ptr)
        self.active = True
        self.current_file = path
        print(f">>> perc pool: {os.path.basename(path)}")

    def next_file(self):
        if not self.current_file:
            return
        d = os.path.dirname(self.current_file)
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        if not files:
            return
        self.load(os.path.join(d, random.choice(files)))

    def stop(self):
        self.active = False
        self.raw = None

perc_pool = PercussivePool()

# non linear reader
def acc_read(accumulator, read_pos, frames, speed, smooth, dist):
    acc_len = len(accumulator)
    block = np.zeros(frames, dtype=np.float32)
    pos = read_pos
    prev = 0.0
    ampval = 0.5
    energy = State.energy()
    energy = energy[3]
    leak = linlin(energy, 0.0, 1.0, 0.008, 0.01)

    for i in range(frames):
        idx = int(pos) % acc_len
        d = dist
        # lettura + clipping
        val = (accumulator[idx] +
               accumulator[(idx-1) % acc_len] * d +
               accumulator[(idx+1) % acc_len] * d)
        val = 0.5*(val - prev*leak) + 0.5*prev
        val = np.tanh(val*ampval) 
        prev = val
        block[i] = val

        # avanzamento pos
        step = speed * (np.abs(val)*0.8 + 0.2)
        step = np.clip(step, 0.1, 3.0) #clip sugli step
        pos += step

    # smoothing
    smoothed = np.zeros_like(block)
    smoothed[0] = block[0]
    for i in range(1, frames):
        smoothed[i] = smooth * block[i] + (1 - smooth) * smoothed[i-1]

    read_pos = pos % acc_len
    return smoothed.astype(np.float32), read_pos

# control pool
class ControlPool:
    def __init__(self):
        self.raw = None
        self.ptr = 0
        self.active = False
        self.current_file = None

    def load(self, path):
        if not os.path.isfile(path):
            print("file non valido")
            return
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size==0: 
            print("file vuoto")
            return
        with lock:
            self.raw = raw
            self.ptr = 0
            self.active = True
            self.current_file = path
        print(f">>> control pool: {os.path.basename(path)}")

    def next_file(self):
        if not self.current_file:
            return
        directory = os.path.dirname(self.current_file)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        if not files: return
        next_path = os.path.join(directory, random.choice(files))
        self.load(next_path)

    def stop(self):
        with lock:
            self.active = False
            self.raw = None

    def step(self):
        with lock:
            if not self.active or self.raw is None: return None
            b = int(self.raw[self.ptr % len(self.raw)])
            self.ptr += 1
            return b

# UTILITY
def bytes_to_float(raw_bytes):
    return np.array(raw_bytes, dtype=np.float32)

# block resampling
def resample_block(block, target_len):
    if len(block)==0: return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0,1,len(block))
    x_new = np.linspace(0,1,target_len)
    return np.interp(x_new,x_old,block).astype(np.float32)

# DIRECT PROCESS
def process_direct(raw):
    global direct_memory, fb_buf_direct, comb_buf, comb_pos, State

    energy = State.energy()
    
    dir_factor = 0.9

    if not direct_active or raw is None:
        return None

    n = min(len(raw), len(direct_memory))
    raw = raw[:n]

    # snapshot dei parametri attuali
    with lock:
        p = audio_params.copy()

    fb_amt    = p["fb_amt"]
    delay     = int(p["delay"])
    cutoff    = p["cutoff"]
    comb_con  = p["comb_con"]
    comb_fb   = p["comb_fb"]

    # add raw input
    raw_factor = 0.95
    direct_memory[:n] *= dir_factor
    direct_memory[:n] += raw * raw_factor
    direct_copy = direct_memory[:n].copy()

    # pre-filtering
    hi_factor = 0.001
    lo_factor = 0.06 #ex 0.05
    f_direct = hipass(direct_copy, hi_factor)
    f_direct = lowpass(direct_copy, lo_factor)
    f_direct *= 0.999
    direct_memory[:n] -= f_direct
    direct_memory[:n] = np.clip(direct_memory[:n], -1.0, 1.0)

    # stutter
    chance = 0.02 + 0.05 * energy[3]
    block_size = int(32 + 96 * energy[1])
    direct_memory[:n] *= dir_factor
    direct_memory[:n] = stutter(direct_memory[:n], chance, block_size)

    # feedback
    direct_memory[:n] *= dir_factor
    if len(fb_buf_direct) > delay + 4:  # soglia
         fb_buf_direct = fb_buf_direct[-(delay + 4):]
    if len(fb_buf_direct) >= delay:
        delayed = fb_buf_direct.pop(0)
        min_len = min(len(delayed), n)
        direct_memory[-min_len:] += fb_amt * delayed[:min_len]
        direct_memory[-min_len:] = np.clip(direct_memory[-min_len:], -1.0, 1.0)

    # comb
    # min delay re-map
    min_range, max_range = param_ranges["comb_min"]
    comb_min = linlin(energy[0], 0.0, 1.0, min_range, max_range)
    comb_min = int(comb_min)
    # max delay re-map
    min_range, max_range = param_ranges["comb_max"]
    comb_max = linlin(energy[1], 0.0, 1.0, min_range, max_range)
    comb_max = int(comb_max)
    # pos etc
    comb_pos = comb_pos % len(comb_buf)
    map_val = comb_con * np.clip(energy[1], 0.5, 1.0)
    map_val = np.clip(map_val, 0.0, 1.0)
    comb_memory, comb_pos = comb(
        direct_memory,
        comb_buf,
        comb_pos,
        map_val,
        fb=comb_fb,
        min_delay=comb_min,
        max_delay=comb_max
    )

    # lowpass
    direct_memory[:n] = lowpass(direct_memory[:n], cutoff)
    comb_cutoff = linlin(energy[1], 0.0, 1.0, 0.01, 0.2)
    comb_memory[:n] = lowpass(comb_memory[:n], comb_cutoff)

    # smear
    smear = linlin(energy[3], 0.0, 1.0, 0.05, 0.2) # ex speed, 0.1
    direct_memory[1:n] = (
        (1 - smear) * direct_memory[1:n] +
        smear * direct_memory[:n-1]
    )

    # dc leak
    leak = linlin(energy[3], 0.0, 1.0, 0.008, 0.02) # ex 0.01
    dc_d = np.mean(direct_memory[:n])
    dc_c = np.mean(comb_memory[:n])
    direct_memory[:n] -= dc_d * leak
    comb_memory[:n] -= dc_c * leak

    # lag toward accumulator
    if acc_active:
        lagval = 0.8
        lagdir = lag(direct_memory[:n], accumulator[-n:], lagval, dt)
        lagdir = np.broadcast_to(lagdir, direct_memory[:n].shape)
        lagdir = np.interp(lagdir, [-1, 1], [0.1, 1.0]) # lagdir re-map
        direct_memory[:n] *= dir_factor # staging
        direct_memory[:n] *= lagdir

    # comb mix
    comb_mix = linlin(energy[1], 0.0, 1.0, 0.1, 0.3)
    mix = np.clip(comb_mix, 0.0, 1.0)
    direct_memory[:n] *= dir_factor
    direct_memory[:n] = direct_memory[:n]*(1.0 - mix) + comb_memory[:n]*mix
    direct_memory[:n] = np.clip(direct_memory[:n], -1.0, 1.0)

    # nonlinearity
    tanh_amp = 1.0
    direct_memory[:] = np.tanh(direct_memory * tanh_amp)

    # update state
    State.direct = np.mean(np.abs(direct_memory[:n]))

    # store feedback
    fb_buf_direct.append(direct_memory[:n].copy())

    return direct_memory[:n].copy()

# ACCUMULATOR PROCESS
def process_accumulator(raw):
    global accumulator, fb_buf_acc, State

    energy = State.energy()

    acc_factor = 0.5

    if not acc_active or raw is None:
        return None

    n = min(len(raw), acc_size)
    if n <= 0:
        return None

    # params snapshot 
    with lock:
        p = audio_params.copy()

    acc_decay = p["acc_decay"]
    fb_amt    = p["fb_amt"]
    delay     = int(p["delay"])

    # shift accumulator
    accumulator = np.roll(accumulator, -n) * acc_decay

    # add raw
    raw_factor = 0.99
    raw = raw[:n] * raw_factor
    accumulator *= acc_factor
    accumulator[-n:] += raw
    
    # pre-filtering
    hi_factor = 0.003  # highpass
    lo_factor = 0.05  # lowpass 
    acc_copy = accumulator[-n:].copy()
    f_acc = hipass(acc_copy, hi_factor)
    f_acc = lowpass(f_acc, lo_factor)
    f_acc *= 0.999
    accumulator[-n:] -= f_acc
    accumulator[-n:] = np.clip(accumulator[-n:], -1.0, 1.0)

    # feedback
    if len(fb_buf_acc) >= delay:
        delayed = fb_buf_acc.pop(0)
        min_len = min(len(delayed), n)
        accumulator[-min_len:] += fb_amt * delayed[:min_len]
        accumulator[-n:] *= acc_factor # staging

    # nonlinearity
    accumulator[:] = np.tanh(accumulator)

    # filtering
    cutoff = linlin(energy[0], 0.0, 1.0, 0.5, 0.1)
    accumulator[:] = lowpass(accumulator, cutoff)

    # clip
    accumulator[:] = np.clip(accumulator, -1.0, 1.0)

    # update state
    State.accumulator = np.mean(np.abs(accumulator[:]))

    # store feedback
    fb_buf_acc.append(accumulator[-n:].copy())

    return accumulator.copy()

# PERCUSSIVE PROCESS
def process_percussive(frames):
    if not perc_pool.active or perc_pool.raw is None:
        return None

    out = np.zeros(frames, dtype=np.float32)
    raw = perc_pool.raw
    L = len(raw)

    energy = State.energy()
    energy = energy[2]
    
    for i in range(frames):
        idx = (perc_pool.loop_start + int(perc_pool.ptr)) % L
        s = raw[idx]

        # transient shaping
        power = 0.1 + 0.3 * energy
        s = np.sign(s) * (abs(s) ** power) # ex 0.3
        s = np.tanh(s * 4.0)     # ex 3        

        # 
        leak_amount = 0.05
        prev = out[i-1] if i > 0 else 0.0
        s = s + prev * leak_amount

        out[i] = s

        # lineare
        perc_pool.ptr += 1

    # stutter
    chance = 0.02 + 0.05 * energy
    block_size = int(8 + 24 * energy)
    out = stutter(out, chance, block_size)

    # lowpass
    out = lowpass(out, factor=0.1)

    # clip
    out = np.clip(out, -1.0, 1.0)

    # update state
    State.perc = np.mean(np.abs(out))

    return out  # ex * 0.7  

# AUDIO CALLBACK 
acc_read_pos = 0.0   

def audio_callback(outdata, frames, t, status):
    global acc_read_pos

    out = np.zeros(frames, dtype=np.float32)
    filled = 0

    while filled < frames:
        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            break

        take = min(len(block), frames - filled)
        out[filled:filled + take] += block[:take]

        if take < len(block):
            try:
                audio_queue.put_nowait(block[take:])
            except queue.Full:
                pass

        filled += take

    if acc_active:
        with lock:
            p = audio_params.copy()

        acc_block, acc_read_pos = acc_read(
            accumulator,
            acc_read_pos,
            frames,
            p["speed"],
            p["smooth"],
            p["dist"]
        )

        out[:frames] += acc_block[:frames]

    if perc_active and perc_pool.active:
        perc = process_percussive(frames)
        if perc is not None:
            out[:frames] += perc_mix * perc[:frames]

    outdata[:, 0] = np.tanh(out) # riduzione

def audio_pool_thread():
    global pool_source, audio_queue, max_blocksize, speed
    while True:
        if pool_source.active:
            chunk = pool_source.next_chunk(1024)
            if chunk is not None:
                d = process_direct(chunk)
                a = process_accumulator(chunk)
                blocks = [b for b in [d, a] if b is not None]
                if blocks:
                    min_len = min(len(b) for b in blocks)
                    audio_block = sum(b[:min_len] for b in blocks)
                    audio_block = np.tanh(audio_block)
                    try:
                        audio_queue.put_nowait(audio_block[:max_blocksize])
                    except queue.Full:
                        pass
        time.sleep(0.01)  

threading.Thread(target=audio_pool_thread, daemon=True).start()

# GUI
root = TkinterDnD.Tk()
root.title("key2noise")
root.configure(bg="#FF00FF")
root.geometry("900x700")

# log box
log_box = tk.Text(root, bg="black", fg="#FF00FF", font=("Consolas", 20))
log_box.pack(expand=True, fill="both", padx=10, pady=10)
log_box.config(state="disabled") 

# funzione log
def log(msg):
    log_box.config(state="normal")
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    log_box.config(state="disabled")

# file load
def load_file():
    path = filedialog.askopenfilename(
        title="load",
        filetypes=[("All files", "*.*")]
    )
    if path:
        pool_source.load(path)
        log(f"[POOL] loaded: {os.path.basename(path)}")

# perc load
def load_perc():
    path = filedialog.askopenfilename(
        title="load perc",
        filetypes=[("All files", "*.*")]
    )
    if path:
        perc_pool.load(path)
        log(f"[PERC] loaded: {os.path.basename(path)}")

def stop_pool1():
    pool_source.stop()  # ferma il pool
    with audio_queue.mutex:
        audio_queue.queue.clear()  # svuota la coda audio
    with lock:
        accumulator[:] = 0
        direct_memory[:] = 0
        fb_buf_acc.clear()
        fb_buf_direct.clear()
    log("[POOL 1] stopped and buffers cleared")

# stop pool 2 
def stop_perc():
    perc_pool.stop()
    log("[POOL 2] stopped")

# pulsanti frame
frame_audio = tk.Frame(root, bg="#FF00FF", bd=2, relief="ridge")
frame_audio.pack(padx=10, pady=5, fill="x")

# pulsanti pool 1
btn_audio_load = tk.Button(frame_audio, text="LOAD 1", command=load_file, width=10, bg="white")
btn_audio_load.pack(side="left", padx=5, pady=5)

btn_audio_stop = tk.Button(frame_audio, text="STOP 1", command=stop_pool1, width=10, bg="white")
btn_audio_stop.pack(side="left", padx=5, pady=5)

btn_audio_next = tk.Button(frame_audio, text="NEXT 1", command=pool_source.next_file, width=10, bg="white")
btn_audio_next.pack(side="left", padx=5, pady=5)

# pulsanti pool 2 (percussive)
btn_perc_load = tk.Button(frame_audio, text="LOAD 2", command=load_perc, width=10, bg="white")
btn_perc_load.pack(side="left", padx=5, pady=5)

btn_perc_stop = tk.Button(frame_audio, text="STOP 2", command=stop_perc, width=10, bg="white")
btn_perc_stop.pack(side="left", padx=5, pady=5)

btn_perc_next = tk.Button(frame_audio, text="NEXT 2", command=perc_pool.next_file, width=10, bg="white")
btn_perc_next.pack(side="left", padx=5, pady=5)

# bind 
root.bind("<KeyPress>", on_key)
root.bind("<KeyRelease>", on_key_release)
root.focus_set()  

# polling per log dei parametri
def update_log():
    with lock:
        for key in keys_down:
            if key in key_param_map:
                param = key_param_map[key]
                val = audio_params.get(param, None)
                if val is not None:
                    log(
                        f"[KEY] {param} = {val:.4f} "
                        f"(key='{key}', "
                        f"1={'1' in pressed_keys}, "
                        f"2={'2' in pressed_keys})"
                    )
    root.after(200, update_log)

root.after(200, update_log)

# audio stream
recreate_stream(blocksize)

# avvia 
root.mainloop()
