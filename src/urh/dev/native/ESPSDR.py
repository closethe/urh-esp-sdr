"""URH native backend for the ESP-SDR ESP32-S3 firmware.

Install this module next to ``urh/dev/native/Device.py`` (the bundled
``install_urh.py`` does that and wires the device into URH's backend list).

RX: snapshot bursts (CAP16/CAP20) at 16/40/80 MS/s, or the gapless IQS
    stream at 250-15.625 kS/s (native USB only).
TX: one-shot frames (TX16/TX20), a frame repeated N times (TXRUN), or a
    modem-local loop (REPLAY) for continuous transmission. TX needs 40 or
    80 MS/s.
"""

import os
import time
from collections import OrderedDict
from multiprocessing import Array

import numpy as np

from urh.dev.native.Device import Device
from urh.dev.native.espsdr_protocol import (
    BURST_RATE_CODES,
    BURST_RATES,
    DEFAULT_BAUD,
    MAX_CAPTURE_SAMPLES,
    MAX_TX_SAMPLES,
    STREAM_RATES,
    EspsdrError,
    EspsdrLink,
    decode_iq_payload,
    pack_iq,
    wire_size,
)

#: Capture size per snapshot. Larger bursts mean fewer USB transfers.
CAPTURE_SAMPLES = int(os.environ.get("URH_ESPSDR_CAPTURE_SAMPLES", "16380"))
#: 8 (CAP16) or 10 (CAP20) bits per component on the capture path.
RX_BITS = int(os.environ.get("URH_ESPSDR_RX_BITS", "8"))
#: 8 (TX16) or 10 (TX20) bits per component on the transmit path.
TX_BITS = int(os.environ.get("URH_ESPSDR_TX_BITS", "8"))
#: IQS bit depth; 8 keeps the stream inside a 2 Mbaud UART link.
STREAM_BITS = int(os.environ.get("URH_ESPSDR_STREAM_BITS", "8"))

if RX_BITS not in (8, 10):
    raise ValueError("URH_ESPSDR_RX_BITS must be 8 or 10")
if TX_BITS not in (8, 10):
    raise ValueError("URH_ESPSDR_TX_BITS must be 8 or 10")
if STREAM_BITS not in (8, 16):
    raise ValueError("URH_ESPSDR_STREAM_BITS must be 8 or 16")


def snap_sample_rate(value) -> int:
    """Nearest rate the firmware supports (16/40/80 MS/s or a 2^n decimation).

    Requests above 2 MS/s go to the snapshot path, below to the IQS stream:
    mixing the two would silently change the sample timeline.
    """
    rate = float(value)
    if rate >= 2_000_000:
        return min(BURST_RATES, key=lambda r: abs(r - rate))
    return min(STREAM_RATES, key=lambda r: abs(r - rate))


def snap_frequency(value) -> int:
    try:
        mhz = int(round(float(value) / 1e6))
    except (TypeError, ValueError):
        return 2442
    return max(100, min(6000, mhz))


def snap_bandwidth(value) -> int:
    """Returns the BANDWIDTH argument in MHz (0 = widest, else 13..69)."""
    try:
        mhz = int(round(float(value) / 1e6))
    except (TypeError, ValueError):
        return 0
    if mhz <= 0:
        return 0
    return max(13, min(69, mhz))


def _bw_from_mhz(mhz: int) -> int:
    return 1_000_000 * int(mhz) if mhz else 0


def _open_link(port: str, attempts: int = 3):
    """Open and SYNC, retrying a device that just re-enumerated on USB."""
    last_error = None
    for attempt in range(attempts):
        link = EspsdrLink(port)
        try:
            link.open()
            link.sync()
            return link
        except (EspsdrError, OSError) as exc:
            last_error = exc
            try:
                link.close()
            except Exception:
                pass
            if attempt + 1 < attempts:
                time.sleep(0.5)
    raise EspsdrError(
        "could not open %s: %s" % (port or "auto-detected port", last_error)
    )


class ESPSDR(Device):
    """ESP32-S3 SDR as a URH device.

    The device identifier is the serial port (``COM6``, ``/dev/ttyACM0``).
    Leave it empty to auto-detect the first Espressif USB device.
    """

    DATA_TYPE = np.int8
    SYNC_TX_CHUNK_SIZE = 0
    CONTINUOUS_TX_CHUNK_SIZE = 0

    @classmethod
    def get_device_list(cls):
        try:
            from serial.tools import list_ports

            return [p.device for p in list_ports.comports()]
        except ImportError:
            return []

    @property
    def has_multi_device_support(self):
        return True

    # -- constructor -------------------------------------------------------

    def __init__(
        self,
        center_freq,
        sample_rate,
        bandwidth,
        gain,
        if_gain=1,
        baseband_gain=1,
        resume_on_full_receive_buffer=False,
    ):
        super().__init__(
            center_freq=center_freq,
            sample_rate=sample_rate,
            bandwidth=bandwidth,
            gain=gain,
            if_gain=if_gain,
            baseband_gain=baseband_gain,
            resume_on_full_receive_buffer=resume_on_full_receive_buffer,
        )

        self._esp_frequency = snap_frequency(center_freq)
        self._esp_sample_rate = snap_sample_rate(sample_rate)
        self._esp_bandwidth = snap_bandwidth(bandwidth)
        self.success = 0
        self.error_codes = {0: "OK"}

        self.receive_process_function = ESPSDR.receive_sync
        self.send_process_function = ESPSDR.send_sequence

    # -- parameters --------------------------------------------------------

    @property
    def frequency(self) -> int:
        return self._esp_frequency

    @frequency.setter
    def frequency(self, value):
        mhz = snap_frequency(value)
        if mhz == self._esp_frequency:
            return
        if abs(float(value) / 1e6 - mhz) > 1e-6:
            self.device_messages.append(
                "ESP32-S3 SDR: frequency snapped to %d MHz (whole-MHz tuning)" % mhz
            )
        self._esp_frequency = mhz
        self.set_device_frequency(mhz * 1_000_000)

    @property
    def sample_rate(self) -> int:
        return self._esp_sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        rate = snap_sample_rate(value)
        if rate == self._esp_sample_rate:
            return
        if rate != int(value):
            self.device_messages.append(
                "ESP32-S3 SDR: sample rate snapped to %s" % _rate_text(rate)
            )
        self._esp_sample_rate = rate
        self.set_device_sample_rate(rate)

    @property
    def bandwidth(self) -> int:
        return _bw_from_mhz(self._esp_bandwidth)

    @bandwidth.setter
    def bandwidth(self, value):
        mhz = snap_bandwidth(value)
        if mhz == self._esp_bandwidth:
            return
        if mhz and abs(float(value) / 1e6 - mhz) > 1e-6:
            self.device_messages.append(
                "ESP32-S3 SDR: bandwidth snapped to %d MHz" % mhz
            )
        self._esp_bandwidth = mhz
        self.set_device_bandwidth(_bw_from_mhz(mhz))

    @property
    def device_parameters(self) -> OrderedDict:
        return OrderedDict(
            [
                (self.Command.SET_FREQUENCY.name, self.frequency),
                (self.Command.SET_SAMPLE_RATE.name, self.sample_rate),
                (self.Command.SET_BANDWIDTH.name, self.bandwidth),
                (self.Command.SET_RF_GAIN.name, self.gain),
                ("identifier", self.device_serial),
                ("rx_bits", RX_BITS),
                ("tx_bits", TX_BITS),
                ("capture_samples", CAPTURE_SAMPLES),
                ("baudrate", DEFAULT_BAUD),
            ]
        )

    @property
    def receive_process_arguments(self):
        return self.child_data_conn, self.child_ctrl_conn, self.device_parameters

    @property
    def send_process_arguments(self):
        return self.child_ctrl_conn, self.send_config, self.device_parameters

    # -- IQ conversion -----------------------------------------------------

    @staticmethod
    def bytes_to_iq(buffer) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.int8).reshape((-1, 2), order="C").copy()

    @staticmethod
    def iq_to_bytes(samples: np.ndarray):
        if isinstance(samples, np.ndarray) and samples.dtype == np.complex64:
            samples = samples.view(np.float32).reshape(-1, 2)
        payload = pack_iq(np.asarray(samples), TX_BITS)
        arr = Array("B", len(payload), lock=False)
        arr[:] = np.frombuffer(payload, dtype=np.uint8)
        return arr

    # -- receive process ---------------------------------------------------

    @staticmethod
    def receive_sync(data_connection, ctrl_connection, params):
        port = params.get("identifier") or ""
        bits = int(params.get("rx_bits", 8))
        capture_samples = int(params.get("capture_samples", MAX_CAPTURE_SAMPLES))
        capture_samples = max(256, min(capture_samples, MAX_CAPTURE_SAMPLES))

        link = None
        try:
            link = _open_link(port)
            ident = link.identify()
            ctrl_connection.send("%s:0" % ident["info"])
            if ident["transport"]:
                ctrl_connection.send("%s:0" % ident["transport"])
            _apply_receiver_settings(link, params)
            user_gain = link.query_gain()
            ctrl_connection.send("%s:0" % user_gain)

            while True:
                rate = int(params[Device.Command.SET_SAMPLE_RATE.name])
                if rate in BURST_RATE_CODES:
                    action = _burst_loop(
                        link, data_connection, ctrl_connection, params, rate, bits, capture_samples
                    )
                else:
                    action = _stream_loop(
                        link, data_connection, ctrl_connection, params, rate
                    )
                if action == "stop":
                    break
                # A rate/gain change needs a clean receiver state.
                link.close()
                link = _open_link(port)
                _apply_receiver_settings(link, params)
        except (EspsdrError, OSError) as exc:
            try:
                ctrl_connection.send("ESP32-S3 SDR RX: %s:1" % exc)
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            try:
                link.close()
            except Exception:
                pass
            try:
                ctrl_connection.send("close:0")
            except (BrokenPipeError, EOFError, OSError):
                pass
            data_connection.close()
            ctrl_connection.close()

    # -- transmit process --------------------------------------------------

    @staticmethod
    def send_sequence(ctrl_connection, send_config, params):
        port = params.get("identifier") or ""
        bits = int(params.get("tx_bits", 8))
        requested_rate = int(params[Device.Command.SET_SAMPLE_RATE.name])
        rate = 80_000_000 if requested_rate >= 60_000_000 else 40_000_000

        raw = np.frombuffer(send_config.send_buffer, dtype=np.uint8).tobytes()
        if bits == 8:
            samples_total = len(raw) // 2
        else:
            samples_total = (len(raw) * 2) // 5
        samples = decode_iq_payload(raw, samples_total, bits)

        repeats = int(send_config.sending_repeats)  # 0 = forever
        start_sample = int(send_config.current_sent_index.value) // 2
        start_sample = min(start_sample, samples_total)
        if start_sample >= samples_total:
            start_sample = 0  # a finished run restarts from the beginning
        if start_sample:
            samples = samples[start_sample:]

        link = None
        sent_frames = 0
        try:
            link = _open_link(port)
            ident = link.identify()
            ctrl_connection.send("%s:0" % ident["info"])
            if requested_rate != rate:
                ctrl_connection.send(
                    "ESP32-S3 SDR TX: sample rate forced to %d MHz (TX supports 40/80):0"
                    % (rate // 1_000_000)
                )
            _apply_receiver_settings(link, params)

            max_frame = min(MAX_TX_SAMPLES, rate // 10)
            chunks = [
                samples[i : i + max_frame]
                for i in range(0, len(samples), max_frame)
            ]

            def stop_requested() -> bool:
                action = _poll_tx_commands(ctrl_connection, link, params)
                return action == "stop"

            if not chunks:
                ctrl_connection.send("ESP32-S3 SDR TX: empty buffer:1")
                return

            payload0 = pack_iq(chunks[0], bits)
            n0 = len(chunks[0])
            if repeats == 0 and len(chunks) == 1:
                # Modem-local loop: gap-free repetition of a single frame.
                reply = link.replay(payload0, n0, rate, bits)
                ctrl_connection.send("%s:0" % reply)
                if not reply.startswith("PLAYING"):
                    raise EspsdrError(reply)
                last_keep = time.monotonic()
                while not stop_requested():
                    if time.monotonic() - last_keep >= 1.0:
                        try:
                            link.replay_keep()
                            last_keep = time.monotonic()
                        except EspsdrError as exc:
                            ctrl_connection.send("ESP32-S3 SDR TX keepalive: %s:1" % exc)
                            break
                    time.sleep(0.1)
                link.replay_stop()
                return

            if repeats > 1 and repeats <= 1000 and len(chunks) == 1:
                def on_frame(_i):
                    send_config.current_sending_repeat.value = _i
                    send_config.current_sent_index.value = samples_total * 2

                link.tx_run(payload0, n0, rate, repeats, bits, on_frame=on_frame)
                send_config.current_sending_repeat.value = repeats
                send_config.current_sent_index.value = samples_total * 2
                ctrl_connection.send("TX finished %d repeats:0" % repeats)
                return

            total_passes = 0
            while repeats == 0 or total_passes < repeats:
                for i, chunk in enumerate(chunks):
                    if stop_requested():
                        return
                    reply = link.tx(pack_iq(chunk, bits), len(chunk), rate, bits)
                    if not reply.startswith("SENT"):
                        raise EspsdrError(reply)
                    sent_frames += 1
                    done = sent_frames * max_frame
                    send_config.current_sent_index.value = min(done * 2, samples_total * 2)
                total_passes += 1
                send_config.current_sending_repeat.value = total_passes
                if repeats == 0 and len(chunks) > 1:
                    send_config.current_sent_index.value = 0

            send_config.current_sent_index.value = samples_total * 2
            ctrl_connection.send("TX finished %d passes:0" % max(total_passes, 1))
        except (EspsdrError, OSError) as exc:
            try:
                ctrl_connection.send("ESP32-S3 SDR TX: %s:1" % exc)
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            try:
                link.replay_stop()
            except Exception:
                pass
            try:
                link.close()
            except Exception:
                pass
            try:
                ctrl_connection.send("close:0")
            except (BrokenPipeError, EOFError, OSError):
                pass


# ---------------------------------------------------------------------------
# Helpers running inside the child processes
# ---------------------------------------------------------------------------


def _rate_text(rate: int) -> str:
    if rate >= 1_000_000:
        return "%g MS/s" % (rate / 1e6)
    return "%g kS/s" % (rate / 1e3)


def _apply_receiver_settings(link: EspsdrLink, params):
    # device_parameters already carries the snapped whole-MHz frequency; the
    # bandwidth is in Hz and has to be snapped to MHz here.
    freq = params[Device.Command.SET_FREQUENCY.name]
    link.set_frequency(int(round(float(freq))))
    gain = params[Device.Command.SET_RF_GAIN.name]
    link.set_gain(int(gain))
    link.set_bandwidth(snap_bandwidth(params[Device.Command.SET_BANDWIDTH.name]))


def _poll_rx_commands(ctrl_connection, link, params):
    """Apply queued control messages; returns "stop", "restart" or None."""
    from urh.dev.native.Device import Device as Dev

    action = None
    while ctrl_connection.poll():
        command = ctrl_connection.recv()
        if command == Dev.Command.STOP.name:
            return "stop"
        try:
            tag, value = command
        except (TypeError, ValueError):
            continue
        try:
            if tag == Dev.Command.SET_FREQUENCY.name:
                params[tag] = value
                link.set_frequency(snap_frequency(value))
            elif tag == Dev.Command.SET_RF_GAIN.name:
                params[tag] = value
                link.set_gain(int(value))
            elif tag == Dev.Command.SET_BANDWIDTH.name:
                params[tag] = value
                link.set_bandwidth(snap_bandwidth(value))
            elif tag == Dev.Command.SET_SAMPLE_RATE.name:
                params[tag] = value
                action = "restart"
        except EspsdrError as exc:
            ctrl_connection.send("ESP32-S3 SDR: %s:1" % exc)
    return action


def _poll_tx_commands(ctrl_connection, link, params):
    from urh.dev.native.Device import Device as Dev

    while ctrl_connection.poll():
        command = ctrl_connection.recv()
        if command == Dev.Command.STOP.name:
            return "stop"
        try:
            tag, value = command
        except (TypeError, ValueError):
            continue
        try:
            if tag == Dev.Command.SET_FREQUENCY.name:
                link.set_frequency(snap_frequency(value))
            elif tag == Dev.Command.SET_RF_GAIN.name:
                link.set_gain(int(value))
        except EspsdrError as exc:
            ctrl_connection.send("ESP32-S3 SDR: %s:1" % exc)
    return None


def _burst_loop(link, data_connection, ctrl_connection, params, rate, bits, capture_samples):
    while True:
        action = _poll_rx_commands(ctrl_connection, link, params)
        if action:
            return action
        try:
            iq = link.capture(capture_samples, rate, bits)
        except EspsdrError as exc:
            ctrl_connection.send("ESP32-S3 SDR capture: %s:1" % exc)
            time.sleep(0.2)
            continue
        data_connection.send_bytes(np.ascontiguousarray(iq).tobytes())


def _stream_loop(link, data_connection, ctrl_connection, params, rate):
    stream_rate = min(STREAM_RATES, key=lambda r: abs(r - rate))
    if stream_rate != rate:
        ctrl_connection.send(
            "ESP32-S3 SDR: stream rate snapped to %d kS/s:0" % (stream_rate // 1_000)
        )
        params[Device.Command.SET_SAMPLE_RATE.name] = stream_rate
    bits = STREAM_BITS
    link.start_stream(stream_rate, bits=bits)
    ctrl_connection.send("IQS %d kS/s %d bit:0" % (stream_rate // 1_000, bits))
    while True:
        action = _poll_rx_commands(ctrl_connection, link, params)
        if action:
            link.stop_stream()
            return action
        frame = link.read_stream_frame()
        if frame is None:
            return "restart"
        _seq, _index, iq = frame
        if len(iq):
            data_connection.send_bytes(np.ascontiguousarray(iq).tobytes())
