"""Device-level tests. Skipped when URH is not importable."""

import os
import sys
import unittest

import numpy as np

try:
    import urh  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("URH is not installed")

URH_SRC = os.environ.get("URH_SRC")
if URH_SRC and URH_SRC not in sys.path:
    sys.path.insert(0, URH_SRC)

from urh.dev.native.ESPSDR import (  # noqa: E402
    ESPSDR,
    snap_bandwidth,
    snap_frequency,
    snap_sample_rate,
)


class SnapTests(unittest.TestCase):
    def test_sample_rate_snaps(self):
        self.assertEqual(snap_sample_rate(80e6), 80000000)
        self.assertEqual(snap_sample_rate(16e6), 16000000)
        self.assertEqual(snap_sample_rate(250e3), 250000)
        self.assertEqual(snap_sample_rate(1e6), 250000)
        self.assertEqual(snap_sample_rate(8e6), 16000000)
        self.assertEqual(snap_sample_rate(30e6), 40000000)

    def test_frequency_and_bandwidth(self):
        self.assertEqual(snap_frequency(2442.4e6), 2442)
        self.assertEqual(snap_frequency(90e6), 100)
        self.assertEqual(snap_bandwidth(20e6), 20)
        self.assertEqual(snap_bandwidth(5e6), 13)
        self.assertEqual(snap_bandwidth(100e6), 69)
        self.assertEqual(snap_bandwidth(0), 0)


class DeviceTests(unittest.TestCase):
    def make(self):
        return ESPSDR(center_freq=2442e6, sample_rate=1e6, bandwidth=20e6, gain=20)

    def test_parameters_snap(self):
        dev = self.make()
        self.assertEqual(dev.frequency, 2442)
        self.assertEqual(dev.sample_rate, 250000)
        self.assertEqual(dev.bandwidth, 20e6)
        self.assertEqual(dev.DATA_TYPE, np.int8)

    def test_properties_forward(self):
        dev = self.make()
        dev.frequency = 2443.7e6
        self.assertEqual(dev.frequency, 2444)
        dev.sample_rate = 80e6
        self.assertEqual(dev.sample_rate, 80000000)
        dev.bandwidth = 40e6
        self.assertEqual(dev.bandwidth, 40e6)

    def test_device_parameters(self):
        dev = self.make()
        dev.device_serial = "COM6"
        params = dev.device_parameters
        self.assertEqual(params["identifier"], "COM6")
        self.assertEqual(params["rx_bits"], 8)
        self.assertEqual(params["tx_bits"], 8)

    def test_iq_bytes_roundtrip(self):
        dev = self.make()
        samples = np.array([[64, -64], [-128, 127], [0, 10]], dtype=np.int8)
        buffer = dev.iq_to_bytes(samples)
        decoded = dev.bytes_to_iq(bytes(buffer))
        np.testing.assert_array_equal(decoded[:, 0], samples[:, 0])
        np.testing.assert_array_equal(decoded[:, 1], samples[:, 1])


    def test_receiver_settings_use_whole_mhz(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fake_espsdr import FakeFirmware
        from urh.dev.native.ESPSDR import _apply_receiver_settings
        from urh.dev.native.espsdr_protocol import EspsdrLink

        firmware = FakeFirmware()
        link = EspsdrLink(ser=firmware)
        params = {
            "SET_FREQUENCY": 2442,
            "SET_RF_GAIN": 20,
            "SET_BANDWIDTH": 20e6,
        }
        _apply_receiver_settings(link, params)
        self.assertIn("FREQ 2442", firmware.commands)
        self.assertIn("GAIN MANUAL 20", firmware.commands)
        self.assertIn("BANDWIDTH 20", firmware.commands)


if __name__ == "__main__":
    unittest.main()
