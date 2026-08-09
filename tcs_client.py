from __future__ import annotations

import math
import telnetlib
import threading
import time
from typing import Mapping

import numpy as np

from scara_diffusion_policy.constants import (
    DEFAULT_JOINT_TARGET,
    JOINT_LIMITS,
    JOINT_NAMES,
    STATE_DIM,
)


class TCSCommandError(RuntimeError):
    def __init__(self, command: str, reply: str) -> None:
        super().__init__(f"TCS error on {command!r}: {reply}")
        self.command = command
        self.reply = reply


class JointLimitError(ValueError):
    pass


def complete_joint_target(target: Mapping[str, float]) -> dict[str, float]:
    complete = dict(DEFAULT_JOINT_TARGET)
    complete.update(target)
    return {joint: float(complete[joint]) for joint in JOINT_NAMES}


def require_joint_target_within_limits(target: Mapping[str, float]) -> dict[str, float]:
    complete = complete_joint_target(target)
    violations = []
    for joint in JOINT_NAMES:
        value = float(complete[joint])
        lower, upper = JOINT_LIMITS[joint]
        if not math.isfinite(value):
            violations.append(f"{joint} is not finite")
        elif value < lower or value > upper:
            violations.append(f"{joint}={value:.6g} outside [{lower:.6g}, {upper:.6g}]")
    if violations:
        raise JointLimitError("; ".join(violations))
    return complete


def joint_dict_to_vector(target: Mapping[str, float]) -> np.ndarray:
    complete = complete_joint_target(target)
    vector = np.zeros(STATE_DIM, dtype=np.float32)
    for index, joint in enumerate(JOINT_NAMES):
        vector[index] = float(complete[joint])
    return vector


def vector_to_joint_dict(vector: np.ndarray | list[float]) -> dict[str, float]:
    if len(vector) < len(JOINT_NAMES):
        raise ValueError(f"expected at least {len(JOINT_NAMES)} joint values, got {vector!r}")
    return {joint: float(vector[index]) for index, joint in enumerate(JOINT_NAMES)}


def format_joint_target(target: Mapping[str, float]) -> str:
    complete = complete_joint_target(target)
    return " ".join(f"{joint}={float(complete[joint]):.4f}" for joint in JOINT_NAMES)


class TCSBaseClient:
    def __init__(
        self,
        host: str = "192.168.0.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.verbose = bool(verbose)
        self.lock = threading.Lock()
        self.connection = telnetlib.Telnet(host, port, timeout)

    def command(self, command: str) -> str:
        if self.verbose:
            print(f"TCS command: {command}")
        with self.lock:
            self.connection.write((command + "\n").encode("ascii"))
            line = self.connection.read_until(b"\n").decode("ascii").strip()
        if not line:
            raise TCSCommandError(command, "empty reply")
        parts = line.split(" ", 1)
        if parts[0].startswith("-"):
            raise TCSCommandError(command, line)
        return parts[1] if len(parts) > 1 else ""

    def command_sleep(self, command: str, delay: float = 0.15) -> str:
        reply = self.command(command)
        time.sleep(max(0.0, float(delay)))
        return reply

    def close(self) -> None:
        self.connection.close()


class TCSReadClient(TCSBaseClient):
    def __init__(
        self,
        host: str = "192.168.0.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)

    def get_wherej(self) -> list[float]:
        return [float(value) for value in self.command("wherej").split()]

    def get_joint_state(self) -> dict[str, float]:
        values = self.get_wherej()
        if len(values) < STATE_DIM:
            raise ValueError(f"expected at least {STATE_DIM} values from wherej, got {values!r}")
        return {
            "J1": values[0] * 0.001,
            "J2": values[1] * math.pi / 180.0,
            "J3": values[2] * math.pi / 180.0,
            "J4": values[3] * math.pi / 180.0,
        }


class TCSMotionClient(TCSReadClient):
    def __init__(
        self,
        host: str = "192.168.0.10",
        port: int = 10100,
        timeout: float = 5.0,
        verbose: bool = False,
        profile: int = 2,
        mspeed: int = 100,
        profile_speed: int = 50,
        profile_accel: int = 50,
        profile_ramp: float = 0.08,
        profile_straight: int = 0,
        configure_motion: bool = True,
        set_tool: bool = True,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout, verbose=verbose)
        self.profile = int(profile)
        self.axis_count = 4
        try:
            self.startup()
            if configure_motion:
                self.configure_motion(
                    mspeed=mspeed,
                    profile_speed=profile_speed,
                    profile_accel=profile_accel,
                    profile_ramp=profile_ramp,
                    profile_straight=profile_straight,
                    set_tool=set_tool,
                )
        except BaseException:
            self.close()
            raise

    def startup(self) -> None:
        startup_delay = 0.15
        self.command_sleep("nop", startup_delay)
        self.command_sleep("mode 0", startup_delay)
        self.command_sleep("SelectRobot 1", startup_delay)
        self.command_sleep("hp 1 30", startup_delay)
        if self.command_sleep("hp", startup_delay) != "1":
            raise TCSCommandError("hp", "high power is not enabled")
        self.command_sleep("attach 1", startup_delay)
        if self.command_sleep("attach", startup_delay) != "1":
            raise TCSCommandError("attach", "robot 1 is not attached")
        self.axis_count = len(self.get_wherej())
        self.command_sleep("state", startup_delay)
        self.command_sleep("nop", startup_delay)

    def configure_motion(
        self,
        mspeed: int = 20,
        profile_speed: int = 35,
        profile_accel: int = 35,
        profile_ramp: float = 0.08,
        profile_straight: int = 0,
        set_tool: bool = True,
    ) -> None:
        self.command(f"mspeed {int(mspeed)}")
        self.command(
            f"profile {self.profile} {int(profile_speed)} {int(profile_speed)} "
            f"{int(profile_accel)} {int(profile_accel)} "
            f"{float(profile_ramp)} {float(profile_ramp)} -1 {int(profile_straight)}"
        )
        if set_tool:
            self.command("tool 0 0 0 0 0 0")

    def movej(self, joints: Mapping[str, float]) -> None:
        joints = require_joint_target_within_limits(joints)
        values = [
            joints["J1"] * 1000.0,
            joints["J2"] * 180.0 / math.pi,
            joints["J3"] * 180.0 / math.pi,
            joints["J4"] * 180.0 / math.pi,
        ]
        self.command(f"MoveJ {self.profile} " + " ".join(f"{value:.6g}" for value in values))

    def halt(self) -> None:
        self.command("halt")
