"""Serial protocol client for the ESP-SDR ESP32-S3 firmware.

Covers the subset of docs/protocol.md and docs/tx.md that the URH backend
needs: SYNC, INFO, CAPS, LIMITS?, FREQ, GAIN, BANDWIDTH, CAP16/CAP20,
TX16/TX20, REPLAY16/REPLAY20, REPLAY KEEP/STOP, TXRUN and the IQS stream.

The module has no URH dependency so it can be used from the command line and
from tests. pyserial is imported lazily, only when a real port is opened.
"""

from __future__ import annotations

import struct
import time
import zlib
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

#: Largest snapshot the firmware can capture or transmit (14-bit word count).
MAX_CAPTURE_SAMPLES = 16380
MAX_TX_SAMPLES = 16380

#: CAP rate code by sample rate. The S3 receiver only accepts these three.
BURST_RATE_CODES = {80_000_000: 0, 40_000_000: 1, 16_000_000: 6}
BURST_RATES = tuple(sorted(BURST_RATE_CODES))

#: TX only runs at 40 or 80 MS/s.
TX_RATES = (40_000_000, 80_000_000)

#: IQS decimates the 16 MS/s ring; decimation must be a power of two.
STREAM_RING_RATE_HZ = 16_000_000
STREAM_DECIMATIONS = (64, 128, 256, 512, 1024)
STREAM_RATES = tuple(STREAM_RING_RATE_HZ // d for d in STREAM_DECIMATIONS)

DEFAULT_BAUD = 2_000_000

#: Minimum gap between a firmware reply and the next command byte. The S3
#: firmware is briefly out of its line reader right after replying; a command
#: written inside that window is truncated and answered with "ERR command".
COMMAND_SETTLE = 0.02

#: Payload uploads are split into chunks; the S3 USB-CDC endpoint loses bytes
#: from a single multi-kilobyte write, which the firmware reports as "ERR crc".
PAYLOAD_CHUNK = 256
PAYLOAD_CHUNK_GAP = 0.003

#: Quiet period before the first TX/REPLAY header after the settings commands.
TX_SETTLE = 0.3

# Rate code used by IQS: 6 selects the 16 MS/s ring on the S3.
_IQS_RING_CODE = 6


class EspsdrError(RuntimeError):
    """The firmware rejected a command or sent malformed data."""


class EspsdrPortNotFound(EspsdrError):
    """No serial port was given and no Espressif device was found."""


# ---------------------------------------------------------------------------
# Wire formats
# ---------------------------------------------------------------------------


def wire_size(samples: int, bits: int) -> int:
    """Payload bytes for ``samples`` samples at 8 or 10 bits per component."""
    if bits == 8:
        return samples * 2
    if bits == 10:
        return (samples * 20 + 7) // 8
    raise ValueError("bits must be 8 or 10, not %r" % (bits,))


def iq_to_int8(samples: np.ndarray) -> np.ndarray:
    """Clip float/int samples (shape (n, 2), int8 amplitude scale) to int8."""
    arr = np.asarray(samples)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("samples must have shape (n, 2)")
    return np.clip(np.rint(arr), -128, 127).astype(np.int8)


def pack_iq8(samples: np.ndarray) -> bytes:
    """IQ8 wire format: signed I then Q per sample, two bytes per sample."""
    return iq_to_int8(samples).tobytes(order="C")


def pack_iq10(samples: np.ndarray) -> bytes:
    """IQ10 wire format: two samples in five bytes, I bits 0-9, Q bits 10-19."""
    arr = np.asarray(samples)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("samples must have shape (n, 2)")
    # URH works on an int8 amplitude scale; the DAC takes 10-bit fields.
    # Multiply in float space: int8 * 4 overflows for values above 31.
    fields = np.clip(np.rint(arr.astype(np.float32) * 4), -512, 511).astype(np.int32) & 0x3FF
    n = fields.shape[0]
    pairs = (n + 1) // 2
    padded = np.zeros((pairs * 2, 2), dtype=np.int32)
    padded[:n] = fields
    words = padded[:, 0] | (padded[:, 1] << 10)
    a = words[0::2].astype(np.uint64)
    b = words[1::2].astype(np.uint64)
    out = np.zeros((pairs, 5), dtype=np.uint8)
    out[:, 0] = a & 0xFF
    out[:, 1] = (a >> 8) & 0xFF
    out[:, 2] = ((a >> 16) & 0x0F) | ((b & 0x0F) << 4)
    out[:, 3] = (b >> 4) & 0xFF
    out[:, 4] = (b >> 12) & 0xFF
    # An odd sample count ends after three bytes of the last pair.
    return out.tobytes(order="C")[: wire_size(len(fields), 10)]


def _sign_extend(value: np.ndarray, bits: int) -> np.ndarray:
    value = value.astype(np.int32)
    return (value ^ (1 << (bits - 1))) - (1 << (bits - 1))


def pack_iq(samples: np.ndarray, bits: int) -> bytes:
    return pack_iq8(samples) if bits == 8 else pack_iq10(samples)


def unpack_iq8(data: bytes, samples: int) -> np.ndarray:
    """Decode a CAP16 payload into int8 (n, 2), with Q negated.

    The firmware sends ``(I, -Q)``; this returns standard ``(I, Q)``.
    """
    raw = np.frombuffer(data[: samples * 2], dtype=np.int8).reshape(samples, 2)
    out = raw.astype(np.int16)
    out[:, 1] = -out[:, 1]
    return np.clip(out, -128, 127).astype(np.int8)


def unpack_iq10(data: bytes, samples: int) -> np.ndarray:
    """Decode a CAP20 payload into int8 (n, 2) at int8 amplitude scale."""
    pairs = (samples + 1) // 2
    raw = np.frombuffer(data, dtype=np.uint8)
    if len(raw) < pairs * 5:
        raw = np.concatenate((raw, np.zeros(pairs * 5 - len(raw), dtype=np.uint8)))
    b = raw[: pairs * 5].reshape(pairs, 5).astype(np.uint32)
    a = b[:, 0] | (b[:, 1] << 8) | ((b[:, 2] & 0x0F) << 16)
    c = (b[:, 2] >> 4) | (b[:, 3] << 4) | (b[:, 4] << 12)
    words = np.empty(pairs * 2, dtype=np.uint32)
    words[0::2] = a
    words[1::2] = c
    words = words[:samples]
    i = _sign_extend(words & 0x3FF, 10)
    q = _sign_extend((words >> 10) & 0x3FF, 10)
    out = np.stack((i, -q), axis=1)  # undo the firmware's (I, -Q)
    # 10-bit -> int8 amplitude scale, matching the 8-bit capture path.
    scaled = np.rint(out / 4.0)
    return np.clip(scaled, -128, 127).astype(np.int8)


def unpack_iq(data: bytes, samples: int, bits: int) -> np.ndarray:
    return unpack_iq8(data, samples) if bits == 8 else unpack_iq10(data, samples)


def decode_iq_payload(data: bytes, samples: int, bits: int) -> np.ndarray:
    """Decode an already packed transmit payload back to int8 ``(n, 2)``.

    Unlike :func:`unpack_iq` this does not negate Q: the transmit path takes
    standard ``(I, Q)`` exactly as packed by :func:`pack_iq`.
    """
    if bits == 8:
        raw = np.frombuffer(data[: samples * 2], dtype=np.int8).reshape(samples, 2)
        return raw.copy()
    pairs = (samples + 1) // 2
    raw = np.frombuffer(data, dtype=np.uint8)
    if len(raw) < pairs * 5:
        raw = np.concatenate((raw, np.zeros(pairs * 5 - len(raw), dtype=np.uint8)))
    b = raw[: pairs * 5].reshape(pairs, 5).astype(np.uint32)
    a = b[:, 0] | (b[:, 1] << 8) | ((b[:, 2] & 0x0F) << 16)
    c = (b[:, 2] >> 4) | (b[:, 3] << 4) | (b[:, 4] << 12)
    words = np.empty(pairs * 2, dtype=np.uint32)
    words[0::2] = a
    words[1::2] = c
    words = words[:samples]
    i = _sign_extend(words & 0x3FF, 10)
    q = _sign_extend((words >> 10) & 0x3FF, 10)
    out = np.stack((i, q), axis=1)
    return np.clip(np.rint(out / 4.0), -128, 127).astype(np.int8)


def _iqs_payload_to_iq(payload: bytes, count: int, bits: int) -> np.ndarray:
    """Decode one IQS frame payload to int8 (count, 2), Q negated."""
    if bits == 4:
        raw = np.frombuffer(payload[:count], dtype=np.uint8)
        i = _sign_extend(raw & 0x0F, 4).astype(np.float32)
        q = _sign_extend((raw >> 4) & 0x0F, 4).astype(np.float32)
        pair = np.stack((i, -q), axis=1) * 8.0
    elif bits == 8:
        raw = np.frombuffer(payload[: count * 2], dtype=np.int8).reshape(count, 2)
        pair = raw.astype(np.float32)
        pair[:, 1] = -pair[:, 1]
    elif bits == 16:
        raw = np.frombuffer(payload[: count * 4], dtype=np.int16).reshape(count, 2)
        pair = raw.astype(np.float32) / 256.0
        pair[:, 1] = -pair[:, 1]
    else:
        raise EspsdrError("unsupported IQS bit depth %r" % (bits,))
    return np.clip(np.rint(pair), -128, 127).astype(np.int8)


def autodetect_port() -> Optional[str]:
    """Return an Espressif USB serial port, or None."""
    from serial.tools import list_ports

    candidates = []
    for info in list_ports.comports():
        if getattr(info, "vid", None) == 0x303A:
            candidates.append(info.device)
    if candidates:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Serial link
# ---------------------------------------------------------------------------


class EspsdrLink:
    """Line-protocol client. Pass ``ser`` to inject a transport (tests)."""

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = 2.0,
        ser=None,
        ser_factory=None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = ser
        self.ser_factory = ser_factory
        self.info = ""
        self.caps: Tuple[str, ...] = ()
        self.limits: Dict[str, object] = {}
        self.transport = ""
        self.last_iqs_end = ""

    # -- lifecycle ---------------------------------------------------------

    def open(self):
        if self.ser is None:
            if self.ser_factory is not None:
                self.ser = self.ser_factory(self.port, self.baudrate)
            else:
                import serial

                port = self.port or autodetect_port()
                if not port:
                    raise EspsdrPortNotFound(
                        "no serial port given and no ESP32 (VID 0x303A) found"
                    )
                self.port = port
                self.ser = serial.Serial(
                    port,
                    self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout,
                )
        # Discard replies left over from an interrupted session (e.g. a
        # half-finished upload), otherwise they shift every later reply by one.
        reset = getattr(self.ser, "reset_input_buffer", None)
        if callable(reset):
            reset()
        return self.ser

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- low level ---------------------------------------------------------

    def _readline(self, timeout: Optional[float] = None) -> bytes:
        if self.ser is None:
            raise EspsdrError("port is not open")
        if timeout is not None:
            self.ser.timeout = timeout
        line = self.ser.readline()
        if not line:
            raise EspsdrError("timeout waiting for a firmware reply")
        if not line.endswith(b"\n"):
            raise EspsdrError("truncated firmware reply %r" % (line,))
        self._last_reply = time.monotonic()
        return line

    def _read_exact(self, size: int) -> bytes:
        deadline = time.monotonic() + max(2.0, size / self.baudrate * 4.0)
        chunks = []
        remaining = size
        while remaining:
            chunk = self.ser.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                continue
            if time.monotonic() > deadline:
                raise EspsdrError(
                    "timeout reading payload: %d of %d bytes" % (size - remaining, size)
                )
        return b"".join(chunks)

    def write(self, data: bytes):
        if self.ser is None:
            raise EspsdrError("port is not open")
        gap = COMMAND_SETTLE - (time.monotonic() - getattr(self, "_last_reply", 0.0))
        if gap > 0:
            time.sleep(gap)
        self.ser.write(data)
        if hasattr(self.ser, "flush"):
            self.ser.flush()

    def command(self, line: str, timeout: Optional[float] = None) -> str:
        self.write(line.encode("ascii") + b"\n")
        reply = self._readline(timeout).decode("ascii", "replace").strip()
        if reply.startswith("ERR"):
            raise EspsdrError(reply)
        return reply

    def command_raw(self, line: str, timeout: Optional[float] = None) -> str:
        """Like :meth:`command`, but returns ``ERR ...`` instead of raising."""
        self.write(line.encode("ascii") + b"\n")
        return self._readline(timeout).decode("ascii", "replace").strip()

    # -- session -----------------------------------------------------------

    def sync(self, nonce: Optional[int] = None, attempts: int = 20):
        if nonce is None:
            nonce = int(time.time() * 1000) & 0xFFFFFFFF
        expected = "SYNC %d" % nonce
        for _ in range(attempts):
            self.write(("SYNC %d\n" % nonce).encode("ascii"))
            try:
                line = self._readline(0.5).decode("ascii", "replace").strip()
            except EspsdrError:
                continue
            if line == expected:
                if self.ser is not None:
                    self.ser.timeout = self.timeout
                return expected
        raise EspsdrError("no SYNC reply from the firmware")

    def identify(self) -> Dict[str, object]:
        """Read INFO, CAPS, TRANSPORT? and LIMITS?; returns a dict."""
        self.info = self.command("INFO")
        caps_line = self.command("CAPS")
        self.caps = tuple(caps_line.split()[1:])
        try:
            self.transport = self.command("TRANSPORT?")
        except EspsdrError:
            self.transport = ""
        try:
            import json

            self.limits = json.loads(self.command("LIMITS?")[len("LIMITS "):])
        except (EspsdrError, ValueError):
            self.limits = {}
        return {
            "info": self.info,
            "caps": self.caps,
            "limits": self.limits,
            "transport": self.transport,
        }

    def supports(self, feature: str) -> bool:
        return feature.upper() in self.caps

    # -- receiver settings -------------------------------------------------

    def set_frequency(self, mhz: int):
        reply = self.command("FREQ %d" % int(mhz))
        if reply != "OK":
            raise EspsdrError("FREQ %d: %r" % (mhz, reply))

    def set_gain(self, index: int):
        reply = self.command("GAIN MANUAL %d" % int(index))
        if reply != "OK":
            raise EspsdrError("GAIN MANUAL %d: %r" % (index, reply))

    def set_gain_hardware(self):
        reply = self.command("GAIN HARDWARE")
        if reply != "OK":
            raise EspsdrError("GAIN HARDWARE: %r" % (reply,))

    def query_gain(self) -> str:
        return self.command("GAIN?")

    def set_bandwidth(self, mhz: int):
        reply = self.command("BANDWIDTH %d" % int(mhz))
        if reply != "OK":
            raise EspsdrError("BANDWIDTH %d: %r" % (mhz, reply))

    # -- snapshot capture --------------------------------------------------

    def capture(self, samples: int, rate: int, bits: int = 8) -> np.ndarray:
        """Capture one burst; returns int8 ``(samples, 2)`` standard I/Q."""
        if samples < 256 or samples > MAX_CAPTURE_SAMPLES:
            raise EspsdrError("capture size %d outside 256..%d" % (samples, MAX_CAPTURE_SAMPLES))
        try:
            code = BURST_RATE_CODES[int(rate)]
        except KeyError:
            raise EspsdrError("unsupported capture rate %r" % (rate,))
        head = "CAP16" if bits == 8 else "CAP20"
        reply = self.command("%s %d %d" % (head, samples, code))
        parts = reply.split()
        if parts[0] != "DATA" or len(parts) != 4:
            raise EspsdrError("unexpected capture reply %r" % (reply,))
        count = int(parts[1])
        crc = int(parts[2], 16)
        if count != samples:
            raise EspsdrError("asked for %d samples, got %d" % (samples, count))
        payload = self._read_exact(wire_size(count, bits))
        if zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise EspsdrError("capture CRC mismatch")
        return unpack_iq(payload, count, bits)

    # -- continuous IQS stream --------------------------------------------

    def start_stream(self, rate: int, bits: int = 8, shift: Optional[int] = None) -> Dict[str, object]:
        """Start an IQS run; returns the firmware's start reply fields."""
        if rate not in STREAM_RATES:
            raise EspsdrError("stream rate %r is not 16 MS/s / 2^n" % (rate,))
        decimation = STREAM_RING_RATE_HZ // rate
        shift_arg = ""
        if shift is not None:
            shift_arg = " %d" % int(shift)
        reply = self.command(
            "IQS 0 %d %d %d%s" % (decimation, bits, _IQS_RING_CODE, shift_arg)
        )
        parts = reply.split()
        if parts[0] != "IQS" or len(parts) < 6:
            raise EspsdrError("unexpected IQS reply %r" % (reply,))
        return {
            "sample_rate": int(parts[1]),
            "decimation": int(parts[2]),
            "bits": int(parts[3]),
            "shift": int(parts[4]),
            "mode": int(parts[5]),
            "mhz": parts[6] if len(parts) > 6 else "",
        }

    def read_stream_frame(self) -> Optional[Tuple[int, int, np.ndarray]]:
        """Read one IQS frame; returns (sequence, sample_index, iq) or None.

        Returns ``None`` when the firmware's IQSEND line closes the run.
        """
        first = self._read_exact(5)
        if first[:5] == b"IQSEN":
            rest = self.ser.readline().decode("ascii", "replace").strip()
            self.last_iqs_end = "IQSEN" + rest
            return None
        if first[:4] != b"IQS1":
            raise EspsdrError("bad IQS magic %r" % (first[:4],))
        header = first + self._read_exact(19)
        seq, index = struct.unpack("<IQ", header[4:16])
        count, bits, flags, decimation, gain, shift = struct.unpack("<HBBHBB", header[16:24])
        payload_size = count * bits * 2 // 8
        payload = self._read_exact(payload_size)
        crc = struct.unpack("<I", self._read_exact(4))[0]
        if zlib.crc32(header + payload) & 0xFFFFFFFF != crc:
            raise EspsdrError("IQS frame CRC mismatch")
        if count == 0:
            return seq, index, np.zeros((0, 2), dtype=np.int8)
        return seq, index, _iqs_payload_to_iq(payload, count, bits)

    def stop_stream(self, timeout: float = 1.0) -> str:
        """Ask the firmware to end the IQS run (any byte does that)."""
        self.write(b"\x00")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._readline(0.2).decode("ascii", "replace").strip()
            except EspsdrError:
                continue
            if line.startswith("IQSEND"):
                return line
        return ""

    def stream(self, rate: int, bits: int = 8) -> Iterator[Tuple[int, int, np.ndarray]]:
        self.start_stream(rate, bits)
        while True:
            frame = self.read_stream_frame()
            if frame is None:
                return
            yield frame

    # -- transmit ----------------------------------------------------------

    def upload_frame(
        self,
        prefix: str,
        payload: bytes,
        samples: int,
        rate: int,
        timeout: float = 5.0,
    ) -> str:
        """TX16/TX20/REPLAY16/REPLAY20 upload; returns the firmware's reply."""
        if int(rate) not in TX_RATES:
            raise EspsdrError("transmit rate must be 40 or 80 MS/s, not %r" % (rate,))
        if samples < 1 or samples > MAX_TX_SAMPLES:
            raise EspsdrError("transmit size %d outside 1..%d" % (samples, MAX_TX_SAMPLES))
        if samples * 10 > int(rate):
            raise EspsdrError("%d samples exceed one %d MS/s frame" % (samples, rate // 1_000_000))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        line = ("%s %d %d %08x\n" % (prefix, samples, rate, crc)).encode("ascii")
        reply = "ERR command"
        # Give the firmware time to leave the reply path before the header
        # arrives; without this pause the first header is answered with
        # "ERR command" even though the bytes are correct.
        time.sleep(TX_SETTLE)
        for delay in (0.0, 0.15, 0.4, 0.8):
            # A header written right after a reply can be swallowed by the S3
            # firmware, so retry with a growing gap; the payload is not sent
            # until the header has been acknowledged.
            time.sleep(delay)
            self.ser.write(line)
            reply = self._readline(timeout).decode("ascii", "replace").strip()
            if reply == "READY" or reply != "ERR command":
                break
        if reply != "READY":
            raise EspsdrError("expected READY, got %r" % (reply,))
        # The 2 Mbaud CDC link drops bytes from a single large burst; the
        # firmware then sees a short payload and answers "ERR crc".
        for offset in range(0, len(payload), PAYLOAD_CHUNK):
            self.ser.write(payload[offset : offset + PAYLOAD_CHUNK])
            time.sleep(PAYLOAD_CHUNK_GAP)
        reply = self._readline(timeout).decode("ascii", "replace").strip()
        if reply.startswith("ERR"):
            raise EspsdrError(reply)
        return reply

    def replay(self, payload: bytes, samples: int, rate: int, bits: int) -> str:
        prefix = "REPLAY16" if bits == 8 else "REPLAY20"
        return self.upload_frame(prefix, payload, samples, rate)

    def replay_keep(self):
        self.command("REPLAY KEEP")

    def replay_stop(self):
        try:
            self.command_raw("REPLAY STOP")
        except EspsdrError:
            pass

    def tx(self, payload: bytes, samples: int, rate: int, bits: int) -> str:
        prefix = "TX16" if bits == 8 else "TX20"
        return self.upload_frame(prefix, payload, samples, rate)

    def tx_run(
        self,
        payload: bytes,
        samples: int,
        rate: int,
        repeats: int,
        bits: int,
        on_frame=None,
    ) -> None:
        """TXRUN: repeat one frame ``repeats`` times, one upload per frame."""
        if repeats < 1 or repeats > 1000:
            raise EspsdrError("TXRUN repeats out of range: %r" % (repeats,))
        fmt = 16 if bits == 8 else 20
        self.write(("TXRUN %d %d %d %d\n" % (samples, rate, repeats, fmt)).encode("ascii"))
        reply = self._readline().decode("ascii", "replace").strip()
        if reply != "RUN":
            raise EspsdrError("expected RUN, got %r" % (reply,))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack("<I", crc)
        for i in range(repeats):
            # Per iteration the firmware reads the CRC header, answers READY,
            # then reads the payload and answers SENT (tx.c: transmit()).
            self.write(header)
            reply = self._readline(5.0).decode("ascii", "replace").strip()
            if reply != "READY":
                raise EspsdrError("TXRUN frame %d: expected READY, got %r" % (i, reply))
            self.write(payload)
            reply = self._readline(5.0).decode("ascii", "replace").strip()
            if not reply.startswith("SENT"):
                raise EspsdrError("TXRUN frame %d: %r" % (i, reply))
            if on_frame is not None:
                on_frame(i + 1)
        reply = self._readline(5.0).decode("ascii", "replace").strip()
        if reply != "END":
            raise EspsdrError("TXRUN did not end: %r" % (reply,))
