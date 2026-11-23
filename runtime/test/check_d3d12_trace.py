"""Check the trace on the CPU command-list fixture, or on d3d12_com_test with --queue."""

import argparse
import os
import re
import signal
import subprocess


RUNNER = []  # the command that runs a Windows executable here: empty on Windows, ["wine"] for a cross build
# A first start of Wine in a fresh prefix can take tens of seconds, hence the longer limit through a runner.
TIMEOUT_S = 30
RUNNER_TIMEOUT_S = 120


def stop(process):
    """Kill the fixture and, on POSIX, everything it started: Wine's children outlive the launcher it runs."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def run_fixture(exe, *args):
    command = [*RUNNER, exe, *args]
    timeout = RUNNER_TIMEOUT_S if RUNNER else TIMEOUT_S
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace", start_new_session=os.name == "posix")
    except OSError as error:  # no Wine on PATH, or a fixture that was not built
        raise SystemExit(f"cannot start {command[0]}: {error.strerror or error}")
    with process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop(process)
            raise SystemExit(f"{exe} did not finish within {timeout} s")
        except BaseException:
            stop(process)
            raise
    if process.returncode:
        raise SystemExit(f"{exe} exited with {process.returncode}\n{stdout}{stderr}")
    return [line for line in stderr.splitlines() if line.startswith("[d3d12-trace]")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    parser.add_argument("--queue", action="store_true", help="use the Vulkan-backed d3d12_com_test fixture")
    parser.add_argument("--runner", action="append", default=[],
                        help="run the fixture through this command (repeat for its arguments), e.g. wine")
    options = parser.parse_args()
    RUNNER[:] = options.runner

    if run_fixture(options.exe):
        raise SystemExit("D3D12 trace appeared with tracing disabled")

    trace = run_fixture(options.exe, "--trace-d3d12")
    entries = []
    for number, line in enumerate(trace, 1):
        if "synthetic-marker-argument" in line:
            raise SystemExit("D3D12 trace leaked the synthetic marker payload")
        match = re.fullmatch(r"\[d3d12-trace\] #(\d+) ([A-Za-z_][\w:]* (?:called|ok|failed|unsupported|ignored))", line)
        if not match or int(match[1]) != number:
            raise SystemExit(f"Invalid trace sequence at entry {number}\n{line}")
        entries.append(match[2])

    if options.queue:
        expected = [
            "ID3D12CommandQueue::ExecuteCommandLists failed",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList::Close failed",
            "ID3D12CommandQueue::Wait ok",
            "ID3D12CommandQueue::ExecuteCommandLists ok",
            "ID3D12CommandQueue::Signal ok",
        ]
    else:
        expected = [
            "ID3D12GraphicsCommandList::SetPipelineState called",
            "ID3D12GraphicsCommandList::SetGraphicsRootSignature called",
            "ID3D12GraphicsCommandList::SetMarker called",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList::Close ok",
            "ID3D12GraphicsCommandList::Close failed",
            "ID3D12GraphicsCommandList::Reset ok",
            "ID3D12GraphicsCommandList::SetDescriptorHeaps called",
            "ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable called",
            "ID3D12GraphicsCommandList::SetComputeRootSignature called",
            "ID3D12GraphicsCommandList::Dispatch called",
            "ID3D12GraphicsCommandList::DrawInstanced called",
            "ID3D12GraphicsCommandList::DrawIndexedInstanced called",
            "ID3D12GraphicsCommandList::Close failed",
        ]

    observed = iter(entries)
    for entry in expected:
        if entry not in observed:
            raise SystemExit(f"Missing or out-of-order trace entry {entry!r}\n" + "\n".join(trace))
    print(f"D3D12 trace verified with {len(entries)} ordered entries")


if __name__ == "__main__":
    main()
