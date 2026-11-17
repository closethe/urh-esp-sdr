import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from urh.dev.native.espsdr_protocol import (  # noqa: E402
    BURST_RATES,
    STREAM_RATES,
    EspsdrError,
    EspsdrLink,
    decode_iq_payload,
    pack_iq8,
    pack_iq10,
    unpack_iq8,
    unpack_iq10,
    wire_size,
)
from fake_espsdr import FakeFirmware  # noqa: E402


def make_link():
    firmware = FakeFirmware()
    return firmware, EspsdrLink(ser=firmware)


class WireFormatTests(unittest.TestCase):
    def test_wire_sizes(self):
        self.assertEqual(wire_size(1, 8), 2)
        self.assertEqual(wire_size(2, 10), 5)
        self.assertEqual(wire_size(3, 10), 8)
        self.assertEqual(wire_size(16380, 10), 40950)

    def test_iq8_roundtrip_negates_q(self):
        samples = np.array([[10, 20], [-30, 40], [127, -128]], dtype=np.int16)
        payload = pack_iq8(samples)
        decoded = unpack_iq8(payload, len(samples))
        self.assertEqual(decoded.shape, (3, 2))
        np.testing.assert_array_equal(decoded[:, 0], samples[:, 0])
        expected_q = np.clip(-samples[:, 1], -128, 127)
        np.testing.assert_array_equal(decoded[:, 1].astype(np.int16), expected_q)

    def test_iq10_roundtrip(self):
        samples = np.array([[100, -100], [0, 1], [-512 / 4, 511 / 4]], dtype=np.float32)
        payload = pack_iq10(samples)
        self.assertEqual(len(payload), wire_size(len(samples), 10))
        decoded = unpack_iq10(payload, len(samples))
        self.assertEqual(decoded.shape, (3, 2))
        self.assertLessEqual(abs(int(decoded[0, 0]) - 100), 1)

    def test_tx_payload_decode_preserves_sign(self):
        samples = np.array([[64, -64], [-128, 127]], dtype=np.int8)
        for bits in (8, 10):
            payload = pack_iq8(samples) if bits == 8 else pack_iq10(samples)
            decoded = decode_iq_payload(payload, len(samples), bits)
            self.assertLessEqual(abs(int(decoded[0, 1]) + 64), 1)
            self.assertLessEqual(abs(int(decoded[1, 0]) + 128), 1)

    def test_rate_tables(self):
        self.assertEqual(BURST_RATES, (16000000, 40000000, 80000000))
        self.assertEqual(STREAM_RATES[0], 250000)
        self.assertEqual(STREAM_RATES[-1], 15625)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.firmware, self.link = make_link()
        self.link.open()

    def tearDown(self):
        self.link.close()

    def test_sync_and_identify(self):
        self.link.sync(0x1234)
        info = self.link.identify()
        self.assertEqual(info["info"], "S3SDR 6 burst 16380")
        self.assertIn("TX16", info["caps"])
        self.assertEqual(info["limits"]["rates"], [80000000, 40000000, 16000000])
        self.assertTrue(self.link.supports("REPLAY"))

    def test_settings(self):
        self.link.set_frequency(2442)
        self.link.set_gain(20)
        self.link.set_bandwidth(20)
        self.assertIn("MANUAL", self.link.query_gain())

    def test_capture_iq8(self):
        iq = self.link.capture(1024, 80000000, 8)
        self.assertEqual(iq.dtype, np.int8)
        self.assertEqual(iq.shape, (1024, 2))
        expected = self.firmware.last_capture
        np.testing.assert_array_equal(iq[:, 0], expected[:, 0])
        np.testing.assert_array_equal(iq[:, 1].astype(np.int16), -expected[:, 1])

    def test_capture_iq10(self):
        iq = self.link.capture(1024, 40000000, 10)
        expected = self.firmware.last_capture
        self.assertLessEqual(np.max(np.abs(iq[:, 0].astype(int) - expected[:, 0]), initial=0), 1)
        self.assertLessEqual(np.max(np.abs(iq[:, 1].astype(int) + expected[:, 1]), initial=0), 1)

    def test_tx_frame(self):
        samples = np.array([[10, 20], [-30, 40], [50, -60]], dtype=np.int8)
        payload = pack_iq8(samples)
        reply = self.link.tx(payload, len(samples), 80000000, 8)
        self.assertTrue(reply.startswith("SENT"))
        self.assertEqual(self.firmware.tx_frames[-1], payload)

    def test_replay_and_lease(self):
        payload = pack_iq8(np.full((256, 2), 7, dtype=np.int8))
        reply = self.link.replay(payload, 256, 80000000, 8)
        self.assertEqual(reply, "PLAYING")
        self.link.replay_keep()
        self.link.replay_stop()
        self.assertEqual(self.firmware.tx_replays, 1)

    def test_txrun(self):
        payload = pack_iq8(np.full((300, 2), 5, dtype=np.int8))
        seen = []
        self.link.tx_run(payload, 300, 80000000, 3, 8, on_frame=seen.append)
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(len(self.firmware.tx_frames), 3)

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(EspsdrError):
            self.link.capture(256, 20000000, 8)
        with self.assertRaises(EspsdrError):
            self.link.tx(b"\x00\x00", 1, 16000000, 8)

    def test_stream(self):
        start = self.link.start_stream(250000, 8)
        self.assertEqual(start["sample_rate"], 250000)
        frames = []
        while True:
            frame = self.link.read_stream_frame()
            if frame is None:
                break
            frames.append(frame)
        self.assertEqual(len(frames), self.firmware.iqs_frames)
        expected = self.firmware.last_capture
        self.assertEqual(frames[-1][2].shape, (64, 2))
        np.testing.assert_array_equal(frames[-1][2][:, 1].astype(np.int16), -expected[:, 1])

    def test_iqs_end_line(self):
        self.link.start_stream(15625, 8)
        while self.link.read_stream_frame() is not None:
            pass
        self.assertTrue(self.link.last_iqs_end.startswith("IQSEN"))


if __name__ == "__main__":
    unittest.main()
