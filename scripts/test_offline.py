"""Run the offline test suite with the network blocked. Used by ship_check.py and CI.

    python scripts/test_offline.py

Any test that resolves or connects to a non-local host fails with "network blocked", so a
test can never call a model API or download data by accident. Loopback stays open (asyncio
and thread pools use it on Windows).
"""
import os
import socket
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = (None, "localhost", "127.0.0.1", "::1", "")


class NetworkBlocked(RuntimeError):
    pass


def block_network():
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def connect(self, address):
        if isinstance(address, tuple) and address[0] not in LOCAL:
            raise NetworkBlocked(f"network blocked in tests: connect {address}")
        return real_connect(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL:
            raise NetworkBlocked(f"network blocked in tests: resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.getaddrinfo = getaddrinfo   # socket.create_connection resolves through this


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    block_network()
    suite = unittest.defaultTestLoader.discover("tests", pattern="*_test*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
