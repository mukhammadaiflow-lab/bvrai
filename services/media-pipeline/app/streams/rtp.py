"""RTP (Real-time Transport Protocol) stream handling."""

import asyncio
import struct
import time
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import socket
import structlog


logger = structlog.get_logger()


@dataclass
class RTPHeader:
    """RTP packet header."""
    version: int = 2
    padding: bool = False
    extension: bool = False
    cc: int = 0  # CSRC count
    marker: bool = False
    payload_type: int = 0
    sequence_number: int = 0
    timestamp: int = 0
    ssrc: int = 0


@dataclass
class RTPPacket:
    """RTP packet."""
    header: RTPHeader
    payload: bytes
    received_at: float = 0.0


class RTPStream:
    """
    RTP stream handler for real-time audio.

    Supports:
    - RTP packet parsing
    - Sequence number tracking
    - Timestamp handling
    - SSRC management
    """

    def __init__(
        self,
        session_id: str,
        local_port: int = 0,
        remote_addr: Optional[Tuple[str, int]] = None,
    ):
        self.session_id = session_id
        self.local_port = local_port
        self.remote_addr = remote_addr

        # Socket
        self._socket: Optional[socket.socket] = None

        # Sequence tracking
        self._local_sequence = 0
        self._local_timestamp = 0
        self._local_ssrc = self._generate_ssrc()
        self._remote_ssrc: Optional[int] = None

        # Callbacks
        self._on_packet_received: Optional[Callable] = None

        # Statistics
        self._packets_sent = 0
        self._packets_received = 0
        self._bytes_sent = 0
        self._bytes_received = 0
        self._packets_lost = 0
        self._last_sequence: Optional[int] = None

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False

    def _generate_ssrc(self) -> int:
        """Generate random SSRC."""
        import random
        return random.randint(0, 0xFFFFFFFF)

    async def start(self) -> int:
        """
        Start RTP stream.

        Returns:
            Local port number
        """
        # Create UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)

        # Bind to port (0 = random available port)
        self._socket.bind(("0.0.0.0", self.local_port))
        self.local_port = self._socket.getsockname()[1]

        self._running = True

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        logger.info(
            "rtp_stream_started",
            session_id=self.session_id,
            port=self.local_port,
        )

        return self.local_port

    async def stop(self) -> None:
        """Stop RTP stream."""
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._socket:
            self._socket.close()
            self._socket = None

        logger.info(
            "rtp_stream_stopped",
            session_id=self.session_id,
            stats=self.get_statistics(),
        )

    async def send_audio(
        self,
        audio_data: bytes,
        payload_type: int = 0,
        marker: bool = False,
    ) -> bool:
        """
        Send audio as RTP packet.

        Args:
            audio_data: Audio payload
            payload_type: RTP payload type
            marker: Marker bit

        Returns:
            True if sent successfully
        """
        if not self._socket or not self.remote_addr:
            return False

        # Build RTP packet
        header = RTPHeader(
            payload_type=payload_type,
            sequence_number=self._local_sequence,
            timestamp=self._local_timestamp,
            ssrc=self._local_ssrc,
            marker=marker,
        )

        packet_data = self._build_packet(header, audio_data)

        try:
            loop = asyncio.get_event_loop()
            await loop.sock_sendto(self._socket, packet_data, self.remote_addr)

            self._local_sequence = (self._local_sequence + 1) & 0xFFFF
            self._local_timestamp += len(audio_data) // 2  # Assuming 16-bit samples
            self._packets_sent += 1
            self._bytes_sent += len(packet_data)

            return True

        except Exception as e:
            logger.error(
                "rtp_send_error",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    def set_remote_address(self, addr: Tuple[str, int]) -> None:
        """Set remote address for sending."""
        self.remote_addr = addr

    def set_callback(self, on_packet_received: Callable) -> None:
        """Set packet received callback."""
        self._on_packet_received = on_packet_received

    async def _receive_loop(self) -> None:
        """Main receive loop."""
        loop = asyncio.get_event_loop()

        while self._running and self._socket:
            try:
                data, addr = await loop.sock_recvfrom(self._socket, 2048)

                if len(data) >= 12:  # Minimum RTP header size
                    packet = self._parse_packet(data)

                    if packet:
                        self._track_packet(packet)

                        if self._on_packet_received:
                            await self._on_packet_received(packet)

                        self._packets_received += 1
                        self._bytes_received += len(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.error(
                        "rtp_receive_error",
                        session_id=self.session_id,
                        error=str(e),
                    )

    def _build_packet(self, header: RTPHeader, payload: bytes) -> bytes:
        """Build RTP packet bytes."""
        # First byte: V=2, P, X, CC
        byte0 = (header.version << 6) | (header.padding << 5) | (header.extension << 4) | header.cc

        # Second byte: M, PT
        byte1 = (header.marker << 7) | header.payload_type

        # Pack header
        header_bytes = struct.pack(
            ">BBHII",
            byte0,
            byte1,
            header.sequence_number,
            header.timestamp,
            header.ssrc,
        )

        return header_bytes + payload

    def _parse_packet(self, data: bytes) -> Optional[RTPPacket]:
        """Parse RTP packet from bytes."""
        if len(data) < 12:
            return None

        try:
            byte0, byte1, seq, ts, ssrc = struct.unpack(">BBHII", data[:12])

            header = RTPHeader(
                version=(byte0 >> 6) & 0x3,
                padding=bool((byte0 >> 5) & 0x1),
                extension=bool((byte0 >> 4) & 0x1),
                cc=byte0 & 0xF,
                marker=bool((byte1 >> 7) & 0x1),
                payload_type=byte1 & 0x7F,
                sequence_number=seq,
                timestamp=ts,
                ssrc=ssrc,
            )

            # Skip CSRC entries
            header_size = 12 + (header.cc * 4)

            # Handle extension header
            if header.extension and len(data) >= header_size + 4:
                ext_len = struct.unpack(">H", data[header_size + 2:header_size + 4])[0]
                header_size += 4 + (ext_len * 4)

            payload = data[header_size:]

            return RTPPacket(
                header=header,
                payload=payload,
                received_at=time.time(),
            )

        except Exception as e:
            logger.error("rtp_parse_error", error=str(e))
            return None

    def _track_packet(self, packet: RTPPacket) -> None:
        """Track packet for statistics."""
        # Track SSRC
        if self._remote_ssrc is None:
            self._remote_ssrc = packet.header.ssrc

        # Track packet loss
        if self._last_sequence is not None:
            expected = (self._last_sequence + 1) & 0xFFFF
            if packet.header.sequence_number != expected:
                # Calculate lost packets (handling wraparound)
                if packet.header.sequence_number > expected:
                    lost = packet.header.sequence_number - expected
                else:
                    lost = (0xFFFF - expected) + packet.header.sequence_number + 1
                self._packets_lost += lost

        self._last_sequence = packet.header.sequence_number

    def get_statistics(self) -> dict:
        """Get stream statistics."""
        total_expected = self._packets_received + self._packets_lost
        loss_rate = self._packets_lost / max(1, total_expected)

        return {
            "session_id": self.session_id,
            "local_port": self.local_port,
            "packets_sent": self._packets_sent,
            "packets_received": self._packets_received,
            "packets_lost": self._packets_lost,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "loss_rate": round(loss_rate, 4),
            "local_ssrc": self._local_ssrc,
            "remote_ssrc": self._remote_ssrc,
        }
