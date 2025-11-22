#!/usr/bin/env python3
"""Discord bot that posts Raspberry Pi health graphs every 30 seconds and hourly."""

from __future__ import annotations

import asyncio
import math
import os
import re
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Iterable, Optional

import discord
from discord.ext import tasks
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import psutil

SAMPLE_INTERVAL_SECONDS = 5
SHORT_WINDOW = timedelta(seconds=30)
LONG_WINDOW = timedelta(hours=1)
PING_TARGET = "1.1.1.1"
PING_COUNT = 3
PING_TIMEOUT_SECONDS = 2
JST = timezone(timedelta(hours=9))

_LAST_NET_STATS: Optional[tuple[int, int, datetime]] = None


@dataclass
class MetricSample:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    cpu_temp_c: Optional[float]
    gpu_temp_c: Optional[float]
    net_sent_bps: Optional[float]
    net_recv_bps: Optional[float]
    ping_ms: Optional[float]
    packet_loss_percent: Optional[float]
    throttled: Optional[bool]


def read_cpu_temperature() -> Optional[float]:
    """Attempt to read CPU temperature in Celsius."""
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, NotImplementedError):
        temps = {}

    candidates = ("cpu-thermal", "soc_thermal", "cpu_thermal", "cpu_thermal0", "temp")
    for key in candidates:
        entries = temps.get(key)
        if entries:
            try:
                return float(entries[0].current)
            except (ValueError, AttributeError):
                pass

    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            raw = thermal_path.read_text().strip()
            return float(raw) / 1000
        except (OSError, ValueError):
            return None

    return None


def read_gpu_temperature() -> Optional[float]:
    """Use vcgencmd if available to read GPU temperature."""
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    data = output.strip()
    if data.startswith("temp=") and data.endswith("'C"):
        try:
            return float(data[5:-2])
        except ValueError:
            return None
    return None


def read_network_throughput() -> tuple[Optional[float], Optional[float]]:
    """Return bytes/sec sent and received since last reading."""
    global _LAST_NET_STATS
    counters = psutil.net_io_counters()
    now = datetime.now(timezone.utc)
    if _LAST_NET_STATS is None:
        _LAST_NET_STATS = (counters.bytes_sent, counters.bytes_recv, now)
        return None, None

    last_sent, last_recv, last_time = _LAST_NET_STATS
    _LAST_NET_STATS = (counters.bytes_sent, counters.bytes_recv, now)
    elapsed = (now - last_time).total_seconds()
    if elapsed <= 0:
        return None, None

    sent_bps = max(0.0, (counters.bytes_sent - last_sent) / elapsed)
    recv_bps = max(0.0, (counters.bytes_recv - last_recv) / elapsed)
    return sent_bps, recv_bps


def measure_network_quality(
    target: str = PING_TARGET, count: int = PING_COUNT, timeout: int = PING_TIMEOUT_SECONDS
) -> tuple[Optional[float], Optional[float]]:
    """Ping target host to obtain average latency (ms) and packet loss percentage."""
    cmd = ["ping", "-n", "-c", str(count), "-w", str(timeout), target]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None

    ping_ms: Optional[float] = None
    packet_loss: Optional[float] = None
    for line in output.splitlines():
        if "packet loss" in line:
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)% packet loss", line)
            if match:
                packet_loss = float(match.group(1))
        if "round-trip" in line or "rtt" in line:
            # Linux/mac summary lines both include values split by slash.
            parts = line.split("=")
            if len(parts) < 2:
                continue
            stats = parts[-1].strip().split("/")
            if len(stats) >= 2:
                try:
                    ping_ms = float(stats[1])
                except ValueError:
                    ping_ms = None

    return ping_ms, packet_loss


def read_throttled_state() -> Optional[bool]:
    """Detect whether the Pi is currently thermal throttling via vcgencmd."""
    try:
        output = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    text = output.strip()
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    try:
        value = int(text, 16 if text.startswith("0x") else 10)
    except ValueError:
        return None
    # Bit 2 (0x4) indicates the system is currently throttled due to temperature.
    return bool(value & 0x4)


def collect_sample() -> MetricSample:
    """Read the current Raspberry Pi metrics."""
    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory().percent
    cpu_temp = read_cpu_temperature()
    gpu_temp = read_gpu_temperature()
    net_sent_bps, net_recv_bps = read_network_throughput()
    ping_ms, packet_loss = measure_network_quality()
    throttled = read_throttled_state()
    return MetricSample(
        timestamp=datetime.now(timezone.utc),
        cpu_percent=cpu,
        memory_percent=memory,
        cpu_temp_c=cpu_temp,
        gpu_temp_c=gpu_temp,
        net_sent_bps=net_sent_bps,
        net_recv_bps=net_recv_bps,
        ping_ms=ping_ms,
        packet_loss_percent=packet_loss,
        throttled=throttled,
    )


def load_env_file(path: str = ".env") -> None:
    """Load a .env file (KEY=VALUE per line) into os.environ if present."""
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text().splitlines()
    except OSError:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
            try:
                value = shlex.split(value, posix=True)[0]
            except ValueError:
                value = value.strip(value[0])
        os.environ.setdefault(key, value)


def _series_from_samples(samples: Iterable[MetricSample], attr: str) -> list[float]:
    series: list[float] = []
    for sample in samples:
        value = getattr(sample, attr)
        series.append(float("nan") if value is None else value)
    return series


def _format_optional(value: Optional[float], suffix: str = "", precision: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _format_bytes_per_second(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def render_graph(samples: list[MetricSample], label: str, output_dir: Path) -> Path:
    """Render graph for the given samples and return the PNG path."""
    timestamps = [sample.timestamp for sample in samples]
    cpu_series = [sample.cpu_percent for sample in samples]
    memory_series = [sample.memory_percent for sample in samples]
    cpu_temp_series = _series_from_samples(samples, "cpu_temp_c")
    gpu_temp_series = _series_from_samples(samples, "gpu_temp_c")
    net_sent_series = _series_from_samples(samples, "net_sent_bps")
    net_recv_series = _series_from_samples(samples, "net_recv_bps")
    ping_series = _series_from_samples(samples, "ping_ms")
    packet_loss_series = _series_from_samples(samples, "packet_loss_percent")
    throttle_series = [
        float("nan") if sample.throttled is None else (100.0 if sample.throttled else 0.0)
        for sample in samples
    ]
    net_sent_kbps = [
        value / 1024 if not math.isnan(value) else value for value in net_sent_series
    ]
    net_recv_kbps = [
        value / 1024 if not math.isnan(value) else value for value in net_recv_series
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Raspberry Pi Status ({label})", fontsize=14)

    axes[0].plot(timestamps, cpu_series, label="CPU %", color="#ff7f0e")
    axes[0].plot(timestamps, memory_series, label="Memory %", color="#1f77b4")
    axes[0].set_ylabel("Usage %")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(timestamps, cpu_temp_series, label="CPU Temp °C", color="#d62728")
    axes[1].plot(timestamps, gpu_temp_series, label="GPU Temp °C", color="#2ca02c")
    axes[1].set_ylabel("Temp °C")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(timestamps, net_sent_kbps, label="TX KB/s", color="#9467bd")
    axes[2].plot(timestamps, net_recv_kbps, label="RX KB/s", color="#8c564b")
    axes[2].set_ylabel("Throughput (KB/s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(timestamps, ping_series, label="Ping ms", color="#17becf")
    axes[3].set_ylabel("Ping (ms)")
    axes[3].grid(alpha=0.3)
    ax4b = axes[3].twinx()
    loss_line = ax4b.plot(timestamps, packet_loss_series, label="Packet Loss %", color="#bcbd22")
    throttle_line = ax4b.step(
        timestamps, throttle_series, label="Thermal Throttle (100%=ON)", where="post", color="#7f7f7f"
    )
    ax4b.set_ylabel("Loss / Throttle %")
    ax4b.set_ylim(0, 100)
    lines = axes[3].get_lines() + loss_line + throttle_line
    labels = [line.get_label() for line in lines]
    axes[3].legend(lines, labels, loc="upper right")

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[3].set_xlabel("Time (UTC)")
    fig.autofmt_xdate()

    filename = f"raspi_status_{label.replace(' ', '_').replace('/', '-')}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return file_path


class StatusBot(discord.Bot):
    def __init__(self, channel_id: int, guild_id: int) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.channel_id = channel_id
        self.guild_id = guild_id
        self.channel: Optional[discord.TextChannel] = None
        max_samples = int(LONG_WINDOW.total_seconds() / SAMPLE_INTERVAL_SECONDS) + 10
        self.samples: Deque[MetricSample] = deque(maxlen=max_samples)
        self.graph_dir = Path("graphs")
        self.last_hourly_post: Optional[datetime] = None
        self._slash_registered = False
        self._startup_done = False
        psutil.cpu_percent(interval=None)  # Prime psutil so first reading is meaningful.

    async def setup_hook(self) -> None:
        await self._initialize_bot()

    async def _initialize_bot(self) -> None:
        if self._startup_done:
            return
        self.collect_metrics.start()
        self.scheduled_report.start()
        if not self._slash_registered:
            await self._clear_commands()
            self._register_slash_commands()
            self._slash_registered = True
        print("Syncing slash commands...")
        try:
            synced = await self.sync_commands(guild_ids=[self.guild_id])
        except discord.HTTPException as exc:
            print(f"Failed to sync slash commands: {exc}")
        else:
            synced = synced or []
            names = ", ".join(f"/{command.name}" for command in synced)
            print(f"Slash command sync finished. Synced commands: {names or 'none'}")
        self._startup_done = True

    async def on_ready(self) -> None:
        await self._initialize_bot()
        if self.channel is None:
            self.channel = self.get_channel(self.channel_id)  # type: ignore[assignment]
        if self.channel is None:
            print("Unable to resolve target channel. Check CHANNEL_ID.")
        else:
            print(f"Posting Raspberry Pi stats to #{self.channel.name} ({self.channel.id}).")

    async def _wait_for_channel(self) -> None:
        await self.wait_until_ready()
        while self.channel is None:
            channel = self.get_channel(self.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(self.channel_id)
                except discord.HTTPException:
                    channel = None
            self.channel = channel  # type: ignore[assignment]
            if self.channel is None:
                print(f"Channel {self.channel_id} not cached yet. Retrying in 10s.")
                await asyncio.sleep(10)

    @tasks.loop(seconds=SAMPLE_INTERVAL_SECONDS)
    async def collect_metrics(self) -> None:
        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(None, collect_sample)
        self.samples.append(sample)

    @collect_metrics.before_loop
    async def before_collect(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(minutes=1)
    async def scheduled_report(self) -> None:
        jst_now = datetime.now(timezone.utc).astimezone(JST)
        current_hour = jst_now.replace(minute=0, second=0, microsecond=0)
        if jst_now.minute != 0:
            return
        if self.last_hourly_post == current_hour:
            return
        await self._post_combined_report()
        self.last_hourly_post = current_hour

    @scheduled_report.before_loop
    async def before_scheduled_report(self) -> None:
        await self._wait_for_channel()

    async def _post_combined_report(self) -> None:
        if not self.channel:
            return
        windows = [
            (SHORT_WINDOW, "last 30 seconds"),
            (LONG_WINDOW, "last hour"),
        ]
        attachments: list[Path] = []
        details = []
        loop = asyncio.get_running_loop()

        try:
            for window, label in windows:
                cutoff = datetime.now(timezone.utc) - window
                relevant = [sample for sample in self.samples if sample.timestamp >= cutoff]
                if len(relevant) < 2:
                    continue
                file_path = await loop.run_in_executor(
                    None, render_graph, relevant, label, self.graph_dir
                )
                attachments.append(file_path)
                details.append(f"{label}: {len(relevant)} samples")

            if not attachments:
                print("Not enough samples to produce scheduled report.")
                return

            files = [discord.File(path) for path in attachments]
            content = "JST 00分まとめレポート\n" + "\n".join(details)
            await self.channel.send(content=content, files=files)
        finally:
            for path in attachments:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _register_slash_commands(self) -> None:
        obj = discord.Object(id=self.guild_id)
        decorator = self.slash_command(
            name="raspi_status",
            description="最新のラズパイ状況を表示",
            guild_ids=[self.guild_id],
        )

        @decorator
        async def raspi_status(ctx: discord.ApplicationContext) -> None:
            await self._handle_status_command(ctx)  # type: ignore[arg-type]

        print(f"Registered /raspi_status for guild {self.guild_id}.")

    async def _handle_status_command(self, ctx: discord.ApplicationContext) -> None:
        await ctx.defer()
        if self.samples:
            sample = self.samples[-1]
        else:
            loop = asyncio.get_running_loop()
            sample = await loop.run_in_executor(None, collect_sample)
        embed = self._build_status_embed(sample)
        await ctx.followup.send(embed=embed)

    def _build_status_embed(self, sample: MetricSample) -> discord.Embed:
        ts_utc = sample.timestamp
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        ts_jst = ts_utc.astimezone(JST)
        embed = discord.Embed(
            title="Raspberry Pi 現在の状態",
            description=f"サンプル取得: {ts_jst:%Y-%m-%d %H:%M:%S %Z}",
            timestamp=ts_utc,
            color=discord.Color.green(),
        )
        embed.add_field(
            name="CPU / メモリ",
            value=f"CPU: {sample.cpu_percent:.1f}%\nRAM: {sample.memory_percent:.1f}%",
            inline=True,
        )
        embed.add_field(
            name="温度",
            value=f"CPU: {_format_optional(sample.cpu_temp_c, '°C')}\nGPU: {_format_optional(sample.gpu_temp_c, '°C')}",
            inline=True,
        )
        embed.add_field(
            name="ネットワーク",
            value=f"送信: {_format_bytes_per_second(sample.net_sent_bps)}\n受信: {_format_bytes_per_second(sample.net_recv_bps)}",
            inline=True,
        )
        embed.add_field(
            name="ネットワーク品質",
            value=f"Ping: {_format_optional(sample.ping_ms, ' ms')}\nLoss: {_format_optional(sample.packet_loss_percent, '%')}",
            inline=True,
        )
        throttle_state = "不明" if sample.throttled is None else ("発生中" if sample.throttled else "問題なし")
        embed.add_field(name="サーマルスロットリング", value=throttle_state, inline=True)
        embed.set_footer(text=f"Ping宛先: {PING_TARGET}")
        return embed

    async def _clear_commands(self) -> None:
        http_client = getattr(self, "http", None)
        bulk_clear = getattr(http_client, "bulk_overwrite_guild_application_commands", None)
        if callable(bulk_clear):
            try:
                await bulk_clear(self.application_id, self.guild_id, [])
                print(f"Cleared existing slash commands in guild {self.guild_id}.")
            except discord.HTTPException as exc:
                print(f"Failed to clear slash commands: {exc}")
        else:
            print("HTTP client cannot bulk clear commands; skipping clear step.")


def main() -> None:
    load_env_file()
    token = os.environ.get("DISCORD_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    guild_id = os.environ.get("DISCORD_GUILD_ID")
    if not token or not channel_id or not guild_id:
        raise SystemExit("DISCORD_TOKEN, DISCORD_CHANNEL_ID, and DISCORD_GUILD_ID must be set.")
    bot = StatusBot(
        channel_id=int(channel_id),
        guild_id=int(guild_id),
    )
    bot.run(token)


if __name__ == "__main__":
    main()
