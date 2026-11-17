"""A small ESP-SDR firmware simulator used by the protocol tests.

It implements exactly the commands the URH backend sends, including CRC
checking and the READY/RUN upload handshakes, so the host code can be
exercised without hardware.
"""

from __future__ import annotations

import json
import math
import struct
import zlib

import numpy as np

from urh.dev.native.espsdr_protocol import pack_iq8, pack_iq10, wire_size


class FakeFirmware:
    def __init__(self):
        self.out = bytearray()
        self.input = bytearray()
        self.state = "line"
        self.upload = None
        self.run = None
        self.phase = 0
        self.tx_frames = []
        self.tx_replays = 0
        self.commands = []
        self.last_capture = None
        self.iqs_frames = 3

    # -- serial interface --------------------------------------------------

    def write(self, data: bytes):
        self.input += data
        while self.input:
            if self.state == "upload":
                need = self.upload["size"] - len(self.upload["data"])
                chunk = self.input[:need]
                self.upload["data"] += chunk
                del self.input[: len(chunk)]
                if len(self.upload["data"]) == self.upload["size"]:
                    self._finish_upload()
                    continue
                return
            if self.state == "txrun_header":
                if len(self.input) < 4:
                    return
                header = bytes(self.input[:4])
                del self.input[:4]
                self.run["crc"] = struct.unpack("<I", header)[0]
                self._reply("READY")
                self.state = "txrun_payload"
                continue
            if self.state == "txrun_payload":
                need = self.run["size"] - len(self.run["data"])
                chunk = self.input[:need]
                self.run["data"] += chunk
                del self.input[: len(chunk)]
                if len(self.run["data"]) == self.run["size"]:
                    self._finish_run_frame()
                    continue
                return
            newline = self.input.find(b"\n")
            if newline < 0:
                return
            line = bytes(self.input[:newline]).decode("ascii", "replace")
            del self.input[: newline + 1]
            self._line(line)

    def read(self, size: int):
        if not self.out:
            return b""
        chunk = bytes(self.out[:size])
        del self.out[: len(chunk)]
        return chunk

    def readline(self):
        newline = self.out.find(b"\n")
        if newline < 0:
            return b""
        chunk = bytes(self.out[: newline + 1])
        del self.out[: newline + 1]
        return chunk

    def flush(self):
        pass

    def close(self):
        pass

    def reset_input_buffer(self):
        self.input.clear()

    # -- helpers -----------------------------------------------------------

    def _reply(self, text: str):
        self.out += text.encode("ascii") + b"\n"

    def _tone(self, n):
        samples = []
        for _ in range(n):
            i = int(round(100 * math.cos(self.phase)))
            q = int(round(100 * math.sin(self.phase)))
            # Hardware transmits (I, -Q); the host must negate Q back.
            samples.append((i, -q))
            self.phase += 0.1
        return np.array(samples, dtype=np.int16)

    def _line(self, line: str):
        self.commands.append(line)
        parts = line.split()
        if not parts:
            return
        cmd = parts[0]
        if cmd == "SYNC" and len(parts) == 2:
            self._reply("SYNC " + parts[1])
        elif cmd == "INFO":
            self._reply("S3SDR 6 burst 16380")
        elif cmd == "CAPS":
            self._reply(
                "CAPS VERSION GPIO UARTBAUD RXLIMITS SERIALLEASE TUNEEXT RX40 RX16 "
                "LPFANA GAIN HWAGC IQ8 RING SPEC TX16 TX20 TXRUN REPLAY CW"
            )
        elif cmd == "TRANSPORT?":
            self._reply("TRANSPORT USB 0")
        elif cmd == "LIMITS?":
            self._reply(
                "LIMITS "
                + json.dumps(
                    {
                        "gain": [0, 82, 1],
                        "bandwidth": [13, 69, 1, 0],
                        "rates": [80000000, 40000000, 16000000],
                        "bits": [8, 10],
                    }
                )
            )
        elif cmd == "FREQ":
            self._reply("OK")
        elif cmd in ("GAIN", "GAIN?"):
            if len(parts) > 1 and parts[1] == "MANUAL":
                self._reply("OK")
            elif len(parts) > 1 and parts[1] == "HARDWARE":
                self._reply("OK")
            else:
                self._reply("GAIN MANUAL 20 0 82 0")
        elif cmd == "BANDWIDTH":
            self._reply("OK")
        elif cmd in ("CAP16", "CAP20"):
            n, rate = int(parts[1]), int(parts[2])
            bits = 8 if cmd == "CAP16" else 10
            samples = self._tone(n)
            self.last_capture = samples.copy()
            payload = pack_iq8(samples) if bits == 8 else pack_iq10(samples)
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            self._reply("DATA %d %08x %d" % (n, crc, 200))
            self.out += payload
        elif cmd in ("TX16", "TX20", "REPLAY16", "REPLAY20"):
            n, rate, crc = int(parts[1]), int(parts[2]), int(parts[3], 16)
            bits = 8 if cmd.endswith("16") else 10
            self._reply("READY")
            self.upload = {
                "size": wire_size(n, bits),
                "crc": crc,
                "data": bytearray(),
                "kind": "replay" if cmd.startswith("REPLAY") else "tx",
            }
            self.state = "upload"
        elif cmd == "TXRUN":
            n, rate, repeats, fmt = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            bits = 8 if fmt == 16 else 10
            self._reply("RUN")
            self.run = {
                "left": repeats,
                "size": wire_size(n, bits),
                "bits": bits,
                "crc": None,
                "data": bytearray(),
            }
            self.state = "txrun_header"
        elif cmd == "REPLAY" and len(parts) == 2:
            self._reply("OK")
        elif cmd == "IQS":
            decim, bits = int(parts[2]), int(parts[3])
            rate = 16_000_000 // decim
            self._reply("IQS %d %d %d 0 0 2442" % (rate, decim, bits))
            for seq in range(self.iqs_frames):
                self._iqs_frame(seq, 64, bits)
            self._reply("IQSEND 0 0 64 12345")
        else:
            self._reply("ERR command")

    def _finish_upload(self):
        upload = self.upload
        self.state = "line"
        self.upload = None
        if zlib.crc32(upload["data"]) & 0xFFFFFFFF != upload["crc"]:
            self._reply("ERR crc")
            return
        self.tx_frames.append(bytes(upload["data"]))
        if upload["kind"] == "replay":
            self.tx_replays += 1
            self._reply("PLAYING")
        else:
            self._reply("SENT 1 100 0")

    def _finish_run_frame(self):
        run = self.run
        crc = run["crc"]
        payload = bytes(run["data"])
        run["data"] = bytearray()
        if zlib.crc32(payload) & 0xFFFFFFFF != crc:
            self._reply("ERR crc")
            self.state = "line"
            self.run = None
            return
        self.tx_frames.append(payload)
        self._reply("SENT 1 100 0")
        run["left"] -= 1
        if run["left"] <= 0:
            self.state = "line"
            self.run = None
            self._reply("END")
        else:
            self.state = "txrun_header"

    def _iqs_frame(self, seq: int, count: int, bits: int):
        samples = self._tone(count)
        self.last_capture = samples.copy()
        if bits == 8:
            payload = pack_iq8(samples)
        else:
            payload = pack_iq8(samples).astype(np.int16).tobytes()  # 16-bit test path
        header = (
            b"IQS1"
            + struct.pack("<IQ", seq, seq * count)
            + struct.pack("<HBBHBB", count, bits, 0, 16_000_000 // 250_000, 0, 0)
        )
        crc = zlib.crc32(header + payload) & 0xFFFFFFFF
        self.out += header + payload + struct.pack("<I", crc)
