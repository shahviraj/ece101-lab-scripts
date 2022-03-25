import math
from pydoc import plain
import numpy as np
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from binascii import hexlify
import hashlib

def sha_hash(x, n=8):
    x = int(x)
    x = x.to_bytes(math.ceil(x.bit_length() / 8), 'big')
    return int.from_bytes(hashlib.sha256(x).digest()[(-1 * n // 8):], byteorder='big')

def show_bits(stuff):
    bytes = None
    if isinstance(stuff, str):
        bytes = stuff.encode('utf-8')
    # elif isinstance(stuff, Image):
    #     bytes = stuff.tobytes()
    else:
        bytes = stuff
    max_bytes_to_show = 100
    if len(bytes) > max_bytes_to_show:
        print("{} bits, only showing first {}:".format(
            len(bytes) * 8, max_bytes_to_show * 8))
    else:
        print("{} bits:".format(len(bytes)*8))
    i = 0
    for byte in bytes:
        i += 1
        print(f'{byte:08b}', end='')
        if i > max_bytes_to_show:
            break
    print()


def show_bits_for_file(file):
    with open(file, 'rb') as f:
        bytes = f.read()
    show_bits(bytes)


def string_to_num(s):
    bytes = s.encode('utf-8')
    x = int.from_bytes(bytes, byteorder='big')
    return x


def hash_function(M, A, D):
    return lambda x: ((M * x + A) // D) % 256


def num_to_bits(x, n=8):
    return np.flip(np.array([x >> i & 1 for i in range(n)]))


def show_hash_distribution(hash_fn, n=8, s=10000):
    inputs = np.arange(0, s)
    hashes = [hash_fn(x) for x in inputs]
    plt.hist(hashes, bins=2**n)
    plt.title(f'Hash distribution')
    plt.xlabel('Hash value')
    plt.ylabel('Frequency')

    # bit distribution histogram
    # bits as numpy array
    bits = np.array([num_to_bits(hash_fn(x), n) for x in inputs])
    # number to array of bits
    sum_bits = np.sum(bits, axis=0)
    plt.figure()
    plt.bar(range(len(sum_bits)), sum_bits)
    plt.title(f'Bit distribution')
    plt.xlabel('Bit position')
    plt.ylabel('Frequency')
    plt.show()


def symmetic_encrypt(plaintext, key):
    if isinstance(key, str):
        key = key.encode('utf-8')
    if len(key) > 32:
        key = key[:32]
    elif len(key) < 32:
        key += b'\0' * (32 - len(key))
    key = key
    cipher = Cipher(algorithms.AES(key), modes.CBC(bytearray(16)))

    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')
    padded_plaintext = plaintext + b'\x00' * (16 - len(plaintext) % 16)
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_plaintext) + encryptor.finalize()
    return ct


def symmetric_decrypt(ciphertext, key):
    if isinstance(key, str):
        key = key.encode('utf-8')
    if len(key) > 32:
        key = key[:32]
    elif len(key) < 32:
        key += b'\0' * (32 - len(key))
    key = key
    cipher = Cipher(algorithms.AES(key), modes.CBC(bytearray(16)))

    decryptor = cipher.decryptor()
    pt = decryptor.update(ciphertext) + decryptor.finalize()
    pt = pt.decode('utf-8')
    return pt


def get_asymmetric_keys():
    private_key = RSA.generate(1024)
    public_key = private_key.publickey()
    return private_key, public_key

def show_asymmetric_keys(private_key, public_key):
    private_pem = private_key.export_key().decode()
    public_pem = public_key.export_key().decode()
    print(private_pem)
    print(public_pem)

def asymmetric_encrypt(plaintext, key):
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')
    cipher = PKCS1_OAEP.new(key=key)
    ct = cipher.encrypt(plaintext)
    return ct

def asymmetric_decrypt(ciphertext, key):
    cipher = PKCS1_OAEP.new(key=key)
    pt = cipher.decrypt(ciphertext)
    pt = pt.decode('utf-8')
    return pt