"""End-to-end tests of the child-process RX/TX loops against a fake firmware."""

import threading
import unittest
from multiprocessing import Pipe, Value

import numpy as np

try:
    import urh  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("URH is not installed")

from urh.dev.native import ESPSDR as espsdr_module  # noqa: E402
from urh.dev.native.Device import Device  # noqa: E402
from urh.dev.native.ESPSDR import ESPSDR  # noqa: E402
from urh.dev.native.SendConfig import SendConfig  # noqa: E402
from urh.dev.native.espsdr_protocol import EspsdrLink  # noqa: E402

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_espsdr import FakeFirmware  # noqa: E402


def parameters(rate=40000000, bits=8):
    return {
        Device.Command.SET_FREQUENCY.name: 2442e6,
        Device.Command.SET_SAMPLE_RATE.name: rate,
        Device.Command.SET_BANDWIDTH.name: 20e6,
        Device.Command.SET_RF_GAIN.name: 20,
        "identifier": "FAKE",
        "rx_bits": bits,
        "tx_bits": bits,
        "capture_samples": 1024,
    }


class ChildLoopTests(unittest.TestCase):
    def setUp(self):
        self.firmwares = []
        self.original = espsdr_module.EspsdrLink

        def factory(port):
            firmware = FakeFirmware()
            self.firmwares.append(firmware)
            return EspsdrLink(ser=firmware)

        espsdr_module.EspsdrLink = factory

    def tearDown(self):
        espsdr_module.EspsdrLink = self.original

    def test_receive_burst_and_stop(self):
        data_rx, data_tx = Pipe(duplex=False)
        ctrl_rx, ctrl_tx = Pipe()
        errors = []

        thread = threading.Thread(
            target=ESPSDR.receive_sync,
            args=(data_tx, ctrl_tx, parameters()),
        )
        thread.start()

        payload = data_rx.recv_bytes()
        self.assertEqual(len(payload), 1024 * 2)
        samples = np.frombuffer(payload, dtype=np.int8).reshape(-1, 2)
        self.assertEqual(samples.shape, (1024, 2))

        ctrl_rx.send(Device.Command.STOP.name)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "receive loop did not stop")

    def test_receive_applies_tuning_commands(self):
        data_rx, data_tx = Pipe(duplex=False)
        ctrl_rx, ctrl_tx = Pipe()

        thread = threading.Thread(
            target=ESPSDR.receive_sync,
            args=(data_tx, ctrl_tx, parameters()),
        )
        thread.start()
        data_rx.recv_bytes()
        ctrl_rx.send((Device.Command.SET_FREQUENCY.name, 2450e6))
        data_rx.recv_bytes()
        ctrl_rx.send(Device.Command.STOP.name)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_receive_stream_and_stop(self):
        data_rx, data_tx = Pipe(duplex=False)
        ctrl_rx, ctrl_tx = Pipe()
        thread = threading.Thread(
            target=ESPSDR.receive_sync,
            args=(data_tx, ctrl_tx, parameters(rate=250000)),
        )
        thread.start()
        payload = data_rx.recv_bytes()
        self.assertEqual(len(payload), 64 * 2)
        ctrl_rx.send(Device.Command.STOP.name)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "stream loop did not stop")

    def test_transmit_repeats(self):
        samples = np.full((300, 2), 5, dtype=np.int8)
        buffer = ESPSDR.iq_to_bytes(samples)
        config = SendConfig(
            buffer,
            Value("L", 0),
            Value("L", 0),
            total_samples=2 * len(samples),
            sending_repeats=2,
            iq_to_bytes_method=ESPSDR.iq_to_bytes,
        )
        ctrl_rx, ctrl_tx = Pipe()
        ESPSDR.send_sequence(ctrl_tx, config, parameters(rate=80000000))
        self.assertEqual(len(self.firmwares[0].tx_frames), 2)
        self.assertEqual(config.current_sent_index.value, 2 * len(samples))


if __name__ == "__main__":
    unittest.main()
