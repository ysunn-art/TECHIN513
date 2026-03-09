"""
PsychoMouse Data Collection Script
===================================
Collects high-frequency mouse movement data for fatigue detection research.

Usage:
    python collect_data.py --participant <ID> [options]

Examples:
    python collect_data.py --participant P01
    python collect_data.py --participant P01 --duration 180 --rate 1000
    python collect_data.py --participant P01 --fatigue-poll 30

Requirements:
    pip install pynput
    
Output:
    data/<participant_id>/<session_timestamp>/
        ├── mouse_events.csv       # Raw mouse data
        ├── fatigue_reports.csv    # Subjective fatigue ratings
        └── session_meta.json      # Session metadata
"""

import os
import sys
import csv
import json
import time
import argparse
import threading
import platform
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    from pynput import mouse
except ImportError:
    print("=" * 60)
    print("ERROR: pynput is not installed.")
    print("Please run:  pip install pynput")
    print("=" * 60)
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

DEFAULT_POLLING_RATE = 1000   # Target polling rate (Hz)
DEFAULT_DURATION_MIN = 180    # Default session duration (minutes)
DEFAULT_FATIGUE_POLL = 30     # Ask fatigue rating every N minutes
BUFFER_FLUSH_SIZE = 5000      # Flush to disk every N events
CSV_HEADER = ["timestamp_ms", "x", "y", "event_type", "button", "dx", "dy"]

# Karolinska Sleepiness Scale (KSS) - simplified
KSS_SCALE = """
┌─────────────────────────────────────────────────┐
│         Karolinska Sleepiness Scale (KSS)       │
├─────────────────────────────────────────────────┤
│  1 = Extremely alert                            │
│  2 = Very alert                                 │
│  3 = Alert                                      │
│  4 = Rather alert                               │
│  5 = Neither alert nor sleepy                   │
│  6 = Some signs of sleepiness                   │
│  7 = Sleepy, but no effort to stay awake        │
│  8 = Sleepy, some effort to stay awake          │
│  9 = Very sleepy, great effort, fighting sleep  │
└─────────────────────────────────────────────────┘
"""


# ============================================================
# Data Collector
# ============================================================

class MouseDataCollector:
    """High-frequency mouse event collector with buffered disk writes."""

    def __init__(self, participant_id, duration_min, fatigue_poll_min, output_dir="data"):
        self.participant_id = participant_id
        self.duration_sec = duration_min * 60
        self.fatigue_poll_sec = fatigue_poll_min * 60

        # Session setup
        self.session_start = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / participant_id / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Data buffer (thread-safe via deque)
        self.buffer = deque()
        self.event_count = 0
        self.flush_count = 0

        # CSV writer
        self.csv_path = self.session_dir / "mouse_events.csv"
        self.csv_file = None
        self.csv_writer = None

        # Fatigue reports
        self.fatigue_path = self.session_dir / "fatigue_reports.csv"
        self.fatigue_reports = []

        # Control flags
        self.running = False
        self.paused = False
        self.lock = threading.Lock()

        # Stats
        self.move_count = 0
        self.click_count = 0
        self.scroll_count = 0

    def _init_csv(self):
        """Initialize CSV file with header."""
        self.csv_file = open(self.csv_path, "w", newline="", buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(CSV_HEADER)

    def _timestamp_ms(self):
        """Get current timestamp in milliseconds (high resolution)."""
        return int(time.perf_counter() * 1000)

    def _record_event(self, x, y, event_type, button="", dx=0, dy=0):
        """Record a mouse event to the buffer."""
        if self.paused or not self.running:
            return
        ts = self._timestamp_ms()
        self.buffer.append((ts, x, y, event_type, button, dx, dy))
        self.event_count += 1

        # Flush buffer periodically
        if len(self.buffer) >= BUFFER_FLUSH_SIZE:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered events to CSV file."""
        with self.lock:
            events = []
            while self.buffer:
                events.append(self.buffer.popleft())
            if events and self.csv_writer:
                self.csv_writer.writerows(events)
                self.csv_file.flush()
                self.flush_count += len(events)

    # ---- pynput callbacks ----

    def _on_move(self, x, y):
        """Called on every mouse move event."""
        self._record_event(x, y, "move")
        self.move_count += 1

    def _on_click(self, x, y, button, pressed):
        """Called on mouse button press/release."""
        event_type = "click_down" if pressed else "click_up"
        btn_name = button.name if hasattr(button, "name") else str(button)
        self._record_event(x, y, event_type, button=btn_name)
        if pressed:
            self.click_count += 1

    def _on_scroll(self, x, y, dx, dy):
        """Called on mouse scroll."""
        self._record_event(x, y, "scroll", dx=dx, dy=dy)
        self.scroll_count += 1

    # ---- Fatigue polling ----

    def _fatigue_poll_loop(self):
        """Periodically ask the user for a fatigue rating."""
        next_poll = self.fatigue_poll_sec
        while self.running:
            elapsed = time.time() - self.session_start
            if elapsed >= next_poll:
                self._ask_fatigue_rating(elapsed)
                next_poll += self.fatigue_poll_sec
            time.sleep(5)  # Check every 5 seconds

    def _ask_fatigue_rating(self, elapsed_sec):
        """Prompt user for KSS fatigue rating (non-blocking in separate thread)."""
        self.paused = True  # Pause data collection during input
        elapsed_min = int(elapsed_sec / 60)

        print("\n" + "=" * 60)
        print(f"  ⏰ FATIGUE CHECK  (Session time: {elapsed_min} min)")
        print(KSS_SCALE)

        while True:
            try:
                rating = input("  Enter your fatigue rating (1-9): ").strip()
                rating = int(rating)
                if 1 <= rating <= 9:
                    break
                print("  Please enter a number between 1 and 9.")
            except (ValueError, EOFError):
                print("  Please enter a valid number.")

        report = {
            "elapsed_min": elapsed_min,
            "timestamp": datetime.now().isoformat(),
            "kss_rating": rating,
        }
        self.fatigue_reports.append(report)
        print(f"  ✓ Recorded: KSS = {rating} at {elapsed_min} min")
        print("  Resuming data collection...")
        print("=" * 60 + "\n")

        self.paused = False

    # ---- Session management ----

    def _save_fatigue_reports(self):
        """Save all fatigue reports to CSV."""
        with open(self.fatigue_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["elapsed_min", "timestamp", "kss_rating"])
            writer.writeheader()
            writer.writerows(self.fatigue_reports)

    def _save_metadata(self):
        """Save session metadata to JSON."""
        duration_actual = time.time() - self.session_start
        effective_rate = self.event_count / duration_actual if duration_actual > 0 else 0

        meta = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "session_start": datetime.fromtimestamp(self.session_start).isoformat(),
            "session_end": datetime.now().isoformat(),
            "duration_seconds": round(duration_actual, 1),
            "duration_minutes": round(duration_actual / 60, 1),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "target_polling_rate_hz": DEFAULT_POLLING_RATE,
            "effective_event_rate_hz": round(effective_rate, 1),
            "total_events": self.event_count,
            "move_events": self.move_count,
            "click_events": self.click_count,
            "scroll_events": self.scroll_count,
            "fatigue_reports_count": len(self.fatigue_reports),
            "mouse_info": "Update manually: mouse model, DPI, polling rate setting",
        }

        meta_path = self.session_dir / "session_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def _status_display_loop(self):
        """Print live status every 30 seconds."""
        while self.running:
            time.sleep(30)
            if not self.running:
                break
            elapsed = time.time() - self.session_start
            elapsed_min = int(elapsed / 60)
            remaining = max(0, self.duration_sec - elapsed)
            remaining_min = int(remaining / 60)
            rate = self.event_count / elapsed if elapsed > 0 else 0
            print(
                f"  📊 [{elapsed_min}m elapsed | {remaining_min}m left] "
                f"Events: {self.event_count:,} | "
                f"Rate: {rate:.0f} Hz | "
                f"Moves: {self.move_count:,} | "
                f"Clicks: {self.click_count:,}"
            )

    def start(self):
        """Start the data collection session."""
        print("\n" + "=" * 60)
        print("  🖱️  PsychoMouse Data Collector")
        print("=" * 60)
        print(f"  Participant:    {self.participant_id}")
        print(f"  Session ID:     {self.session_id}")
        print(f"  Duration:       {self.duration_sec // 60} minutes")
        print(f"  Fatigue poll:   every {self.fatigue_poll_sec // 60} minutes")
        print(f"  Output dir:     {self.session_dir}")
        print(f"  Platform:       {platform.system()}")
        print("=" * 60)
        print("  Press Ctrl+C to stop early.")
        print("  Data collection starts in 3 seconds...\n")

        time.sleep(3)

        # Initialize
        self._init_csv()
        self.session_start = time.time()
        self.running = True

        # Initial fatigue rating (baseline)
        print("  📋 Before we start, please provide your baseline fatigue level:")
        self._ask_fatigue_rating(0)

        # Start listener
        listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        listener.start()

        # Start background threads
        fatigue_thread = threading.Thread(target=self._fatigue_poll_loop, daemon=True)
        fatigue_thread.start()

        status_thread = threading.Thread(target=self._status_display_loop, daemon=True)
        status_thread.start()

        print("  ✅ Recording started! Use your computer normally.\n")

        # Main loop - wait for duration or Ctrl+C
        try:
            while self.running:
                elapsed = time.time() - self.session_start
                if elapsed >= self.duration_sec:
                    print("\n  ⏰ Session duration reached!")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  ⚠️  Session stopped by user (Ctrl+C)")

        # Cleanup
        self.running = False
        listener.stop()
        self._flush_buffer()

        # Final fatigue rating
        print("\n  📋 Final fatigue rating:")
        self._ask_fatigue_rating(time.time() - self.session_start)

        # Save everything
        if self.csv_file:
            self.csv_file.close()
        self._save_fatigue_reports()
        meta = self._save_metadata()

        # Summary
        print("\n" + "=" * 60)
        print("  📁 SESSION COMPLETE")
        print("=" * 60)
        print(f"  Duration:          {meta['duration_minutes']} min")
        print(f"  Total events:      {meta['total_events']:,}")
        print(f"  Effective rate:    {meta['effective_event_rate_hz']} Hz")
        print(f"  Move events:       {meta['move_events']:,}")
        print(f"  Click events:      {meta['click_events']:,}")
        print(f"  Scroll events:     {meta['scroll_events']:,}")
        print(f"  Fatigue reports:   {meta['fatigue_reports_count']}")
        print(f"  Output directory:  {self.session_dir}")
        print("=" * 60)
        print("\n  Files saved:")
        print(f"    📄 {self.csv_path}")
        print(f"    📄 {self.fatigue_path}")
        print(f"    📄 {self.session_dir / 'session_meta.json'}")
        print()


# ============================================================
# Label Assignment Utility
# ============================================================

def assign_labels(session_dir, alert_minutes=30, fatigued_minutes=120):
    """
    Post-process: assign 'Alert' / 'Fatigued' / 'Transition' labels
    to the raw mouse event data based on elapsed time.

    Creates a new file: mouse_events_labeled.csv
    """
    session_dir = Path(session_dir)
    input_path = session_dir / "mouse_events.csv"
    output_path = session_dir / "mouse_events_labeled.csv"
    meta_path = session_dir / "session_meta.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    # Load metadata to get session start reference
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Labeling session: {session_dir}")
    print(f"  Alert:      first {alert_minutes} min")
    print(f"  Transition: {alert_minutes}-{fatigued_minutes} min")
    print(f"  Fatigued:   after {fatigued_minutes} min")

    alert_ms = alert_minutes * 60 * 1000
    fatigued_ms = fatigued_minutes * 60 * 1000

    count = {"Alert": 0, "Transition": 0, "Fatigued": 0}
    first_ts = None

    with open(input_path, "r") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header + ["label"])

        for row in reader:
            ts = int(row[0])
            if first_ts is None:
                first_ts = ts

            elapsed = ts - first_ts

            if elapsed < alert_ms:
                label = "Alert"
            elif elapsed < fatigued_ms:
                label = "Transition"
            else:
                label = "Fatigued"

            count[label] += 1
            writer.writerow(row + [label])

    total = sum(count.values())
    print(f"  Total events: {total:,}")
    for label, n in count.items():
        pct = (n / total * 100) if total > 0 else 0
        print(f"    {label}: {n:,} ({pct:.1f}%)")
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PsychoMouse: Mouse data collection for fatigue detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_data.py --participant P01
  python collect_data.py --participant P01 --duration 180
  python collect_data.py --participant P01 --duration 60 --fatigue-poll 15
  python collect_data.py --label data/P01/20260206_143000
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Default: collect data (no subcommand needed)
    parser.add_argument("--participant", "-p", type=str,
                        help="Participant ID (e.g., P01, P02)")
    parser.add_argument("--duration", "-d", type=int, default=DEFAULT_DURATION_MIN,
                        help=f"Session duration in minutes (default: {DEFAULT_DURATION_MIN})")
    parser.add_argument("--fatigue-poll", "-f", type=int, default=DEFAULT_FATIGUE_POLL,
                        help=f"Fatigue rating interval in minutes (default: {DEFAULT_FATIGUE_POLL})")
    parser.add_argument("--output", "-o", type=str, default="data",
                        help="Output directory (default: data/)")

    # Subcommand: label
    label_parser = subparsers.add_parser("label", help="Assign time-based labels to collected data")
    label_parser.add_argument("session_dir", type=str, help="Path to session directory")
    label_parser.add_argument("--alert-min", type=int, default=30,
                              help="Minutes considered 'Alert' (default: 30)")
    label_parser.add_argument("--fatigued-min", type=int, default=120,
                              help="Minutes after which 'Fatigued' starts (default: 120)")

    args = parser.parse_args()

    if args.command == "label":
        assign_labels(args.session_dir, args.alert_min, args.fatigued_min)
    elif args.participant:
        collector = MouseDataCollector(
            participant_id=args.participant,
            duration_min=args.duration,
            fatigue_poll_min=args.fatigue_poll,
            output_dir=args.output,
        )
        collector.start()
    else:
        parser.print_help()
        print("\n  ❌ Error: --participant is required for data collection.")
        print("  Example: python collect_data.py --participant P01\n")


if __name__ == "__main__":
    main()
