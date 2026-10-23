"""Run frontend experiments; the research agent manages the ledger and Git history."""

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EDITABLE = "meta_learning/frontend.py"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def current_commit():
    if os.environ.get("AUTORESEARCH_COMMIT"):
        return os.environ["AUTORESEARCH_COMMIT"]
    if shutil.which("git"):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return "unknown"


def controlled_files(root):
    root = Path(root)
    files = [path for path in root.iterdir() if path.suffix in {".py", ".sh", ".ps1", ".json"}]
    files += [root / name for name in (".gitattributes", "Dockerfile", "requirements.txt", "program.md", "AGENTS.md")]
    for folder in ("meta_learning", "models", "configs", "scripts"):
        files += [path for path in (root / folder).rglob("*")
                  if path.is_file() and path.suffix in {".py", ".json", ".sh", ".ps1"}]
    return sorted(path.relative_to(root).as_posix() for path in files
                  if path.relative_to(root).as_posix() not in {EDITABLE, "protocol.lock.json"})


def verify_protocol(root=ROOT):
    """Reject changed controls and additional executable candidate modules."""
    root = Path(root)
    lock = read_json(root / "protocol.lock.json")
    if set(lock) != {"version", "editable", "objective", "fixed_sha256"} or lock["version"] != 3:
        raise ValueError("Invalid frontend research protocol")
    if lock["editable"] != [EDITABLE] or lock["objective"] != "minimum_test_mouse_mean_log_loss":
        raise ValueError("Unsupported research scope or objective")
    actual = {name: sha256(root / name) for name in controlled_files(root)}
    if actual != lock["fixed_sha256"]:
        changed = sorted(name for name in set(actual) | set(lock["fixed_sha256"])
                         if actual.get(name) != lock["fixed_sha256"].get(name))
        raise ValueError(f"Fixed research controls changed: {changed}")
    return lock


def snapshot(destination, root=ROOT):
    root, destination = Path(root), Path(destination)
    lock = verify_protocol(root)
    names = sorted([*lock["fixed_sha256"], EDITABLE, "protocol.lock.json", "README.md"])
    hashes = {name: sha256(root / name) for name in names}
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    if hashes != {name: sha256(destination / name) for name in names} or hashes != {
            name: sha256(root / name) for name in names}:
        raise ValueError("Sources changed while snapshotting")
    write_json(destination.parent / "snapshot.json", hashes)
    return hashes


def verify_snapshot(experiment):
    experiment = Path(experiment)
    hashes = read_json(experiment / "snapshot.json")
    # The manifest must cover the full current snapshot inventory, including frontend.
    lock = verify_protocol(experiment / "source")
    names = {*lock["fixed_sha256"], EDITABLE, "protocol.lock.json", "README.md"}
    if set(hashes) != names or any(sha256(experiment / "source" / name) != hashes[name] for name in names):
        raise ValueError("Experiment source snapshot changed")
    return hashes


@contextlib.contextmanager
def session_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".research.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another research operation owns this session") from error
        yield


def worker(command, cwd, log, timeout):
    """Keep worker output and terminate its process group on interruption/timeout."""
    with Path(log).open("a") as output:
        process = subprocess.Popen(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Worker exited {code}; inspect {log}")


def session_identity(args):
    from meta_learning.common import resolve_device, runtime_identity
    from meta_learning.config import Config
    from prepare import load_dataset

    lock = verify_protocol()
    dataset = load_dataset(args.prepared_dir, args.data_dir)
    device = resolve_device(args.device)
    return dict(protocol_sha256=digest(lock), source_inventory_sha256=digest(dataset.signature),
                config=Config.load(ROOT / "configs/meta.json").to_dict(), runtime=runtime_identity(device),
                timeout_seconds=args.timeout_seconds)


def ensure_session(args):
    identity = session_identity(args)
    path = args.run_dir / "session.json"
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError("Use a new session: protocol, data, config, runtime, or watchdog differs")
    else:
        if (args.run_dir / "experiments").exists():
            raise ValueError("Experiments exist without session metadata")
        write_json(path, dict(created_utc=timestamp(), identity=identity))
    return identity


def experiment(args):
    identity = ensure_session(args)
    if args.resume is not None:
        directory = experiment_dir(args.run_dir, args.resume)
        hashes = verify_snapshot(directory)
        attempt = read_json(directory / "experiment.json")
        if attempt["session_identity"] != identity:
            raise ValueError("Resume session identity mismatch")
        if (directory / "result.json").exists() and read_json(directory / "result.json")["status"] != "crash":
            raise ValueError("Experiment is complete; its result is already recorded")
        if not (directory / "run.json").exists():
            raise ValueError("No training run to resume; create a new experiment")
    else:
        if not args.description or not args.description.strip():
            raise ValueError("A new experiment requires --description")
        directory = args.run_dir / "experiments" / timestamp()
        directory.mkdir(parents=True)
        hashes = snapshot(directory / "source")
        attempt = dict(experiment=directory.name, description=args.description.strip(),
                       commit=current_commit(),
                       frontend_sha256=hashes[EDITABLE], session_identity=identity, created_utc=timestamp())
        write_json(directory / "experiment.json", attempt)
    source = directory / "source"
    started = time.monotonic()
    result = {key: attempt[key] for key in ("experiment", "description", "commit", "frontend_sha256")}
    result.update(attempt=timestamp(), resumed=args.resume is not None)
    attempt_path = directory / "attempts" / f"{result['attempt']}.json"
    write_json(attempt_path, dict(result, status="running"))
    write_json(directory / "research_status.json", dict(result, status="running"))
    try:
        worker([sys.executable, "-m", "meta_learning.frontend_contract", str(identity["runtime"]["device"])],
               source, directory / "contract.log", min(args.timeout_seconds, 180))
        command = [sys.executable, "train.py", "--config", "configs/meta.json", "--data-dir", str(args.data_dir),
                   "--prepared-dir", str(args.prepared_dir), "--cache-dir", str(args.cache_dir),
                   "--run-dir", str(directory), "--device", str(identity["runtime"]["device"])]
        if args.resume is not None:
            command.append("--resume")
        remaining = args.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Candidate watchdog exhausted during contract checks")
        worker(command, source, directory / "train.log", remaining)
        verify_snapshot(directory)
        status = read_json(directory / "status.json")
        selection = read_json(directory / "selection.json")
        if status["status"] != "complete":
            raise ValueError("Training did not complete its fixed budget")
        expected_role = "query" if identity["config"]["selection_validation_days"] is None else "valid"
        if selection["role"] != expected_role:
            raise ValueError("Unexpected checkpoint selection role")
        selected_test = selection["test"]["aggregate"]
        if not math.isfinite(selected_test["loss"]):
            raise ValueError("Expected finite development-test loss")
        result.update(selection_role=selection["role"], selection_loss=selection["aggregate"]["loss"],
                      selection_accuracy=selection["aggregate"]["accuracy"],
                      test_loss=selected_test["loss"], test_accuracy=selected_test["accuracy"],
                      peak_vram_mb=status["peak_vram_mb"], selected_step=selection["step"],
                      status="complete")
    except BaseException as error:
        result.update(status="crash", error=type(error).__name__, message=str(error))
        result["wall_seconds"] = time.monotonic() - started
        write_json(attempt_path, result)
        write_json(directory / "result.json", result)
        write_json(directory / "research_status.json", result)
        raise
    result["wall_seconds"] = time.monotonic() - started
    write_json(attempt_path, result)
    write_json(directory / "result.json", result)
    write_json(directory / "research_status.json", result)
    for name in ("test_loss", "test_accuracy", "selection_loss", "peak_vram_mb", "experiment"):
        print(f"{name}: {result[name]}", flush=True)
    return result


def experiment_dir(run_dir, identifier):
    if not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Expected one experiment identifier")
    directory = run_dir / "experiments" / identifier
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Missing experiment directory")
    return directory


def replay(args):
    directory = experiment_dir(args.run_dir, args.experiment)
    verify_snapshot(directory)
    source = directory / "source"
    from meta_learning.common import resolve_device, runtime_identity
    from prepare import load_dataset

    identity = read_json(args.run_dir / "session.json")["identity"]
    if (digest(load_dataset(args.prepared_dir, args.data_dir).signature) != identity["source_inventory_sha256"] or
            runtime_identity(resolve_device(args.device)) != identity["runtime"]):
        raise ValueError("Replay requires the session's sources and runtime")
    worker([sys.executable, "evaluate.py", "--split", args.split, "--data-dir", str(args.data_dir),
            "--prepared-dir", str(args.prepared_dir), "--cache-dir", str(args.cache_dir),
            "--run-dir", str(directory), "--device", str(identity["runtime"]["device"])],
           source, directory / f"replay-{args.split}.log", args.timeout_seconds)
    report = read_json(directory / "evaluation" / args.split / "result.json")
    print(json.dumps(report), flush=True)
    return report


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Research interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("check", help="Verify protocol and run finite synthetic frontend checks")
    for action in ("experiment", "replay"):
        child = commands.add_parser(action)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--data-dir", type=Path, default=Path("/data/MouseLFP"))
        child.add_argument("--prepared-dir", type=Path, default=Path("prepared"))
        child.add_argument("--cache-dir", type=Path, default=Path("cache"))
        child.add_argument("--device", default="auto")
        child.add_argument("--timeout-seconds", type=int, default=12000,
                           help="Candidate watchdog; failure preserves artifacts without shortening the training budget")
        if action == "experiment":
            child.add_argument("--description")
            child.add_argument("--resume", metavar="EXPERIMENT_ID")
        if action == "replay":
            child.add_argument("--experiment", required=True, metavar="EXPERIMENT_ID")
            child.add_argument("--split", choices=("query", "valid", "test"), required=True)
    args = parser.parse_args()
    verify_protocol()
    if args.action == "check":
        worker([sys.executable, "-m", "meta_learning.frontend_contract"], ROOT, "/dev/stdout", 180)
        return
    if hasattr(args, "timeout_seconds") and args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    for name in ("run_dir", "data_dir", "prepared_dir", "cache_dir"):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    with session_lock(args.run_dir):
        if args.action == "experiment":
            experiment(args)
        else:
            replay(args)


if __name__ == "__main__":
    main()
