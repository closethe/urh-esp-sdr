# ESP32-S3 SDR backend

This fork registers `ESP32-S3 SDR` as a native URH device. The radio is an
ESP32-S3 running the [esp32s3-sdr](https://github.com/closethe/esp32s3-sdr)
firmware, reached over the chip's native USB Serial/JTAG port at 2 Mbaud, 8N1.

The backend lives in three files:

| File | Role |
|---|---|
| `src/urh/dev/native/espsdr_protocol.py` | line protocol, CRC32 payload uploads, snapshot capture, IQS stream framing, IQ8/IQ10 codecs |
| `src/urh/dev/native/ESPSDR.py` | URH `Device` subclass: RX/TX child processes, rate snapping, IQ conversion |
| `src/urh/dev/{BackendHandler,config,VirtualDevice}.py` | registration, device range, construction |

## Firmware protocol in one page

Commands are newline-terminated ASCII; replies are one line. Uploads are
`header -> READY -> binary payload -> SENT`, where the header carries the CRC32
(little-endian zlib) of the payload the board is about to receive.

| Command | Meaning |
|---|---|
| `SYNC <nonce>` | session start, echoes the nonce |
| `INFO`, `CAPS`, `TRANSPORT?`, `LIMITS?` | identity and capability discovery |
| `FREQ <mhz>`, `GAIN MANUAL <index>`, `BANDWIDTH <index>` | front-end settings (whole MHz only) |
| `CAP16 <n> <rate>`, `CAP20 <n> <rate>` | snapshot burst, `n` samples, rate `16/40/80 MS/s` |
| `TX16 <n> <rate> <crc>`, `TX20 …` | one frame, `n <= rate/10` and `n <= 16380` |
| `TXRUN <n> <rate> <repeats> <fmt>` | `RUN`, then per frame `CRC(4B) -> READY -> payload -> SENT`, finally `END` |
| `REPLAY16 <n> <rate> <crc>` / `REPLAY STOP` / `REPLAY KEEP` | looping playback with a 5 s lease |
| `CW START` / `CW?` / `CW STOP` | unmodulated carrier |
| `IQS <rate> <bits>` | continuous 16 MS/s ring stream, decimation 64…1024 |

`n` always counts **samples**; the payload is `wire_size(n, bits)` bytes (2n for
IQ8, 5n/2 for IQ10). Declaring `n` and sending fewer bytes parks the firmware in
its 5 s upload window, during which every later command is swallowed as payload.

## Mapping URH settings to the firmware

| URH | Firmware | Notes |
|---|---|---|
| Frequency | `FREQ <mhz>` | integers only; the backend snaps to whole MHz |
| Sample rate ≥ 2 MS/s | `CAP16`/`CAP20` | snaps to 80/40/16 MS/s |
| Sample rate < 2 MS/s | `IQS` | snaps to 250k/125k/62.5k/31.25k/15.625k |
| Bandwidth | `BANDWIDTH <index>` | index range 13…69 |
| Gain | `GAIN MANUAL <index>` | index range 0…82 |
| TX repeats = 1 | `TX16`/`TX20` | one frame |
| TX repeats 2…1000 | `TXRUN` | device-side repeat loop |

IQ8 is the default for both directions (`URH_ESPSDR_RX_BITS`,
`URH_ESPSDR_TX_BITS`, `URH_ESPSDR_STREAM_BITS` select 10/16-bit variants).
The S3 transmits `(I, -Q)`; the backend negates Q so URH sees a standard
constellation.

## Bench notes that shaped the backend

These are firmware/host timing properties, not optional tuning:

- **Half duplex.** RX and TX share the front end. Transmitting invalidates the
  current receive stream and rebuilds RX state afterwards, so a single board
  cannot receive while it transmits. Use two boards for simultaneous TX/RX.
- **Stale replies.** `EspsdrLink.open()` discards leftover bytes, otherwise one
  unanswered `ERR ...` line shifts every later reply by one.
- **Header settle.** A TX/REPLAY header written immediately after a settings
  reply can be truncated by the firmware and answered with `ERR command`; the
  backend waits `TX_SETTLE` (0.3 s) and retries with a growing gap.
- **Payload chunking.** A single multi-kilobyte write loses bytes in the USB
  CDC endpoint, producing `ERR crc`. Payloads are written in 256-byte blocks
  with a 3 ms gap.
- **Post-TX quiet period.** After a frame the firmware is briefly busy; wait
  several seconds (port closed) before opening the next session.

## Verified on hardware

- RX, `COM7`, 80 MS/s snapshot: 3 s run delivered 835 380 samples (~278 kS/s
  including gaps); 250 kS/s `IQS` run delivered 897 536 samples (~224 kS/s).
- TX, `COM6`, 1024-sample IQ8 frame at 80 MS/s:
  `ESPSDR-TX finished 1 passes`, `current_sent_sample = 1024`,
  `sending_finished = True`.

Not characterised: absolute power, harmonics, EVM, and symbol timing across
snapshot gaps (URH assumes contiguous samples — prefer stream mode for decoding).

## Tests

```sh
python -m pytest tests/esp_sdr -q
```

`tests/esp_sdr/fake_espsdr.py` emulates the firmware handshakes, CRCs, lease
handling and IQS framing, so the suite needs neither a board nor a serial port.
