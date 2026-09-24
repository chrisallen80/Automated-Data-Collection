"""
Robust TCP Client for Merlin Medipix detector.
"""

from __future__ import annotations
import socket
import threading
import time
import re
from typing import Optional, Tuple, Any
import numpy as np
from .base import BaseCamera


class MerlinClient(BaseCamera):
    """
    TCP client connecting to the Merlin detector command channel (port 6341)
    and optional data channel (port 6342).
    """

    def __init__(self, host: str = "127.0.0.1", cmd_port: int = 6341, data_port: int = 6342):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port

        self._cmd_sock: Optional[socket.socket] = None
        self._data_sock: Optional[socket.socket] = None
        self._connected = False
        self._is_acquiring = False

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._data_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        if host is not None:
            self.host = host
        if port is not None:
            self.cmd_port = port

        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.settimeout(5.0)
            self._cmd_sock.connect((self.host, self.cmd_port))
            self._connected = True
            return True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Could not connect to Merlin at {self.host}:{self.cmd_port} -> {e}")

    def disconnect(self) -> None:
        self.stop_acquisition()
        self._connected = False
        if self._cmd_sock:
            try:
                self._cmd_sock.close()
            except Exception:
                pass
            self._cmd_sock = None

    def is_connected(self) -> bool:
        return self._connected

    def send_command(self, cmd_type: str, cmd_body: str) -> str:
        """Format and transmit MPX command packet."""
        if not self._connected or self._cmd_sock is None:
            raise RuntimeError("Merlin client is not connected.")

        # Command syntax: MPX,000<len>,<TYPE>,<BODY>
        full_inner = f",{cmd_type},{cmd_body}"
        length_str = f"{len(full_inner):03d}"
        packet = f"MPX,000{length_str}{full_inner}"

        with self._lock:
            self._cmd_sock.sendall(packet.encode("utf-8"))
            try:
                data = self._cmd_sock.recv(1024).decode("utf-8", "ignore")
                return data
            except socket.timeout:
                return ""

    def set_value(self, var_name: str, value: Any) -> bool:
        """Set a Merlin parameter via MPX,SET."""
        response = self.send_command("SET", f"{var_name},{value}")
        return len(response) > 0

    def get_value(self, var_name: str) -> str:
        """Query a Merlin parameter via MPX,GET."""
        response = self.send_command("GET", var_name)
        parts = response.split(",")
        if len(parts) >= 5:
            return parts[4]
        return ""

    def configure(
        self,
        num_frames: int,
        bit_depth: int = 12,
        dwell_s: float = 0.001,
        save_dir: str = "",
        filename: str = "scan_data",
        continuous_rw: bool = True
    ) -> bool:
        if not self._connected:
            return False

        self.set_value("NUMFRAMESTOACQUIRE", num_frames)
        self.set_value("NUMBERFRAMESPERTRIGGER", num_frames)
        self.set_value("COUNTERDEPTH", bit_depth)
        self.set_value("CONTINUOUSRW", 1 if continuous_rw else 0)
        self.set_value("ACQUISITIONTIME", dwell_s)

        if save_dir:
            self.set_value("FILEDIRECTORY", save_dir)
            self.set_value("FILENAME", filename)
            self.set_value("FILEENABLE", 1)
        else:
            self.set_value("FILEENABLE", 0)

        # Trigger mode
        self.set_value("TRIGGERSTART", 1)
        self.set_value("TRIGGERSTOP", 1)
        return True

    def start_acquisition(self) -> bool:
        if not self._connected:
            return False
        # Send MPX START command
        pkt = "MPX,21,CMD,STARTACQUISITION"
        with self._lock:
            self._cmd_sock.sendall(pkt.encode("utf-8"))
        self._is_acquiring = True
        return True

    def stop_acquisition(self) -> bool:
        if not self._connected:
            return False
        pkt = "MPX,20,CMD,STOPACQUISITION"
        try:
            with self._lock:
                self._cmd_sock.sendall(pkt.encode("utf-8"))
        except Exception:
            pass
        self._is_acquiring = False
        return True

    def is_acquiring(self) -> bool:
        status = self.get_value("DETECTORSTATUS")
        if status:
            try:
                # 0: idle, 1: acquire/busy
                return int(status) != 0
            except ValueError:
                pass
        return self._is_acquiring

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
