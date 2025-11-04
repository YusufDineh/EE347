#!/usr/bin/env python3
"""
Simplified MechArm forward-run script.

Reads previously-captured positions from a CSV, sends coordinates to the robot
using pymycobot (or simulates them in dry-run mode), and records the measured
poses and servo angles to an output CSV.

Usage examples:
  python mecharm_frd_group_5.py --input mecharm_control_group_5.csv --output recorded_run.csv --dry-run
  python mecharm_frd_group_5.py --input mecharm_control_group_5.csv --output recorded_run.csv --speed 30 --sleep 5

The script is intentionally small and focused: it does not perform any data
processing beyond reading the requested coordinates and writing recorded data.
"""
import time
import csv
import argparse
import logging
from typing import List, Tuple

try:
    from pymycobot.mycobot import MyCobot
    from pymycobot import PI_PORT, PI_BAUD
    PYMCOBOT_AVAILABLE = True
except Exception:
    # If the library or hardware isn't available we will allow dry-run mode.
    PYMCOBOT_AVAILABLE = False


def read_positions_from_csv(path: str) -> List[Tuple[int, List[float]]]:
    """Read positions from CSV and return list of (index, coords).

    The function attempts to read columns named X,Y,Z,RX,RY,RZ. If these are not
    present it falls back to reading by column indices (matching the old
    export format: index, J1..J6, X,Y,Z,RX,RY,RZ).
    """
    positions = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        # If header contains X then use dict access
        if all(k in reader.fieldnames for k in ("X", "Y", "Z")):
            for row in reader:
                idx = int(row.get(reader.fieldnames[0], 0))
                coords = [
                    float(row.get("X", 0.0)),
                    float(row.get("Y", 0.0)),
                    float(row.get("Z", 0.0)),
                    float(row.get("RX", 0.0)),
                    float(row.get("RY", 0.0)),
                    float(row.get("RZ", 0.0)),
                ]
                positions.append((idx, coords))
        else:
            # Fallback: read rows as lists and extract columns 7..12 (0-based indexes)
            f.seek(0)
            simple_reader = csv.reader(f)
            headers = next(simple_reader, None)
            for row in simple_reader:
                if not row:
                    continue
                try:
                    idx = int(row[0])
                except Exception:
                    idx = 0
                # safe extraction of coords: attempt indices 7..12
                coords = []
                for i in range(7, 13):
                    try:
                        coords.append(float(row[i]))
                    except Exception:
                        coords.append(0.0)
                positions.append((idx, coords))

    return positions


def send_coords_and_record(positions: List[Tuple[int, List[float]]], output_csv: str,
                           speed: int = 30, sleep_time: float = 5.0,
                           dry_run: bool = False, interactive: bool = False):
    """Send coordinates to the robot (or simulate) and record measured data.

    Each output CSV row contains:
      Index, Req_X..Req_RZ, Meas_X..Meas_RZ, J1..J6
    """
    logging.info("Starting run: %d positions, dry_run=%s", len(positions), dry_run)

    mc = None
    if not dry_run:
        if not PYMCOBOT_AVAILABLE:
            raise RuntimeError("pymycobot not available; run with --dry-run or install library")
        mc = MyCobot(PI_PORT, PI_BAUD)
        # ensure the robot is powered on
        try:
            mc.power_on()
        except Exception:
            logging.warning("Failed to call power_on(); continuing assuming robot is ready")

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Index",
            "Req_X", "Req_Y", "Req_Z", "Req_RX", "Req_RY", "Req_RZ",
            "Meas_X", "Meas_Y", "Meas_Z", "Meas_RX", "Meas_RY", "Meas_RZ",
            "J1", "J2", "J3", "J4", "J5", "J6",
        ]
        writer.writerow(header)

        for idx, req_coords in positions:
            if interactive:
                user_input = input(f"Press Enter to move to position {idx} or 0 to cancel: ")
                if user_input.strip() == "0":
                    logging.info("User cancelled run at index %s", idx)
                    break

            logging.info("Moving to index %s -> requested coords %s", idx, req_coords)

            if not dry_run:
                try:
                    mc.send_coords(req_coords, speed)
                except Exception as e:
                    logging.warning("send_coords failed for idx %s: %s", idx, e)
                time.sleep(sleep_time)

                # read back measured pose and angles
                try:
                    meas_coords = mc.get_coords()
                except Exception:
                    meas_coords = [0.0] * 6
                try:
                    angles = mc.get_angles()
                except Exception:
                    angles = [0.0] * 6
            else:
                # In dry-run just echo the requested coords and zero angles
                meas_coords = req_coords[:6]
                angles = [0.0] * 6

            # Ensure lists are length 6
            meas_coords = list(meas_coords) + [0.0] * (6 - len(meas_coords))
            angles = list(angles) + [0.0] * (6 - len(angles))

            row = [idx] + [float(x) for x in req_coords[:6]] + [float(x) for x in meas_coords[:6]] + [float(x) for x in angles[:6]]
            writer.writerow(row)
            logging.info("Recorded row for index %s", idx)

    logging.info("Run finished; results written to %s", output_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Simple forward-run script for MechArm")
    p.add_argument("--input", "-i", required=True, help="Input CSV with recorded positions")
    p.add_argument("--output", "-o", required=True, help="Output CSV to write measured poses and angles")
    p.add_argument("--speed", type=int, default=30, help="Servo speed to use when sending commands")
    p.add_argument("--sleep", type=float, default=5.0, help="Seconds to wait after each move")
    p.add_argument("--dry-run", action="store_true", help="Do not connect to the robot; simulate instead")
    p.add_argument("--interactive", action="store_true", help="Prompt before each position")
    p.add_argument("--no-confirm", action="store_true", help="Skip confirmation before run (use carefully)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    if args.dry_run:
        logging.info("Running in dry-run mode; robot will not be contacted")

    if not args.no_confirm:
        resp = input(f"About to run {args.input} -> {args.output}. Continue? [y/N]: ")
        if resp.lower() != 'y':
            logging.info("User cancelled before start")
            return

    positions = read_positions_from_csv(args.input)
    if not positions:
        logging.error("No positions read from input file: %s", args.input)
        return

    send_coords_and_record(positions, args.output, speed=args.speed, sleep_time=args.sleep,
                           dry_run=args.dry_run, interactive=args.interactive)


if __name__ == '__main__':
    main()
