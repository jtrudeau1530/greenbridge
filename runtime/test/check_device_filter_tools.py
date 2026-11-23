"""LEONIDA_VK_DEVICE in vk_descriptor_bench and vk_sandbox (TEST-1141).

With a filter that matches no device, a tool must not choose another GPU. It names the filter and fails the run, as
the other harnesses do (runtime/test/vk_device_pick.h, docs/testing.md "Choosing the GPU"). Before, both tools
ignored the filter and ran on the best-ranked GPU.

    python -I check_device_filter_tools.py <vk_descriptor_bench> <vk_sandbox> <work dir>

Exit 0 when both tools refuse the filter, 1 when one does not, and 77 (skip) on a machine with no Vulkan device,
where no tool can choose one. Standard library only.
"""
import os
import struct
import subprocess
import sys

FILTER = "leonida-no-such-gpu"
UNSUPPORTED = 77


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=120)
    return p.returncode, p.stdout + p.stderr


def main(argv):
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    bench, sandbox, work = argv[1:]
    os.makedirs(work, exist_ok=True)

    # The device list ignores the filter, so it shows whether there is any device to choose from.
    plain = {k: v for k, v in os.environ.items() if k != "LEONIDA_VK_DEVICE"}
    code, out = run([bench, "--list-devices"], env=plain)
    if code != 0 or not any(line.startswith("[") for line in out.splitlines()):
        print(f"SKIP: no Vulkan device to choose from ({bench} --list-devices exit {code})")
        return UNSUPPORTED

    # The smallest scene vk_sandbox reads: magic, version 1, no textures, meshes or draws, and a centre.
    scene = os.path.join(work, "empty_scene.bin")
    with open(scene, "wb") as f:
        f.write(b"LSB1" + struct.pack("<4I", 1, 0, 0, 0) + struct.pack("<3f", 0.0, 0.0, 0.0))
    missing = os.path.join(work, "never_read.spv")   # the device is chosen before the shaders are read

    env = dict(plain, LEONIDA_VK_DEVICE=FILTER)
    failures = []
    for name, cmd in (("vk_descriptor_bench", [bench]),
                      ("vk_sandbox", [sandbox, scene, missing, missing, "--out", os.path.join(work, "out.ppm")])):
        code, out = run(cmd, env=env)
        named = f"LEONIDA_VK_DEVICE={FILTER}" in out
        if code == 0 or not named or "device:" in out:
            failures.append(f"{name}: exit {code}, {'names' if named else 'does not name'} the filter:\n{out.strip()}")
        else:
            print(f"ok: {name} refuses LEONIDA_VK_DEVICE={FILTER} (exit {code})")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
