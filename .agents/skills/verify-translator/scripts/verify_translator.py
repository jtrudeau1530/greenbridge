"""Executable verification harness for the verify-translator skill (standard library only).

    python -I .agents/skills/verify-translator/scripts/verify_translator.py doctor --evidence <dir>
    python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature valu-ops --evidence <dir>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

FEATURES = {
    "valu-ops": ("test_ps_semantics", "ps_ops"),
    "memory-bounds": ("test_buffer_oob_select", "cs_buf_tail"),
    "control-flow-wave64": ("test_saveexec_b64", "cs_w64x_saveexec_b64"),
}


def doctor(repo_root):
    checks = {
        "repo_root": repo_root,
        "rdna2spirv_exists": os.path.isfile(os.path.join(repo_root, "tools", "rdna2spirv.py")),
        "semantic_support_exists": os.path.isfile(os.path.join(repo_root, "tests", "semantic", "support.py")),
        "spirv_as": shutil.which("spirv-as"),
        "spirv_val": shutil.which("spirv-val"),
        "llvm_mc": shutil.which("llvm-mc"),
        "python": sys.version.split()[0],
    }
    checks["ready_for_cpu_spirv"] = bool(
        checks["rdna2spirv_exists"]
        and checks["semantic_support_exists"]
        and checks["spirv_as"]
        and checks["spirv_val"]
    )
    return checks


def drive_feature(repo_root, feature, evidence_dir):
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; choose from {sorted(FEATURES)}")
    test_mod, fixture_name = FEATURES[feature]
    os.makedirs(evidence_dir, exist_ok=True)
    sem_dir = os.path.join(repo_root, "tests", "semantic")
    tools_dir = os.path.join(repo_root, "tools")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([sem_dir, tools_dir]))

    t0 = time.perf_counter()
    p_test = subprocess.run(
        [sys.executable, "-m", "unittest", test_mod, "-v"],
        cwd=sem_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    spv_val_ok = None
    spv_val_err = ""
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        with tempfile.TemporaryDirectory() as tmp:
            spvasm = os.path.join(tmp, fixture_name + ".spvasm")
            spvbin = os.path.join(tmp, fixture_name + ".spv")
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); import support\n"
                "spv = support.translate_fixture(sys.argv[2])\n"
                "open(sys.argv[3], 'w', encoding='utf-8', newline='\\n').write(spv)\n"
            )
            p_tr = subprocess.run(
                [sys.executable, "-c", code, sem_dir, fixture_name, spvasm],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if p_tr.returncode == 0:
                p_as = subprocess.run(
                    ["spirv-as", "--target-env", "vulkan1.2", spvasm, "-o", spvbin],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if p_as.returncode == 0:
                    p_val = subprocess.run(
                        ["spirv-val", "--target-env", "vulkan1.2", spvbin],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    spv_val_ok = p_val.returncode == 0
                    spv_val_err = (p_val.stdout + p_val.stderr).strip()
                else:
                    spv_val_ok = False
                    spv_val_err = (p_as.stdout + p_as.stderr).strip()
            else:
                spv_val_ok = False
                spv_val_err = (p_tr.stdout + p_tr.stderr).strip()

    dt = round((time.perf_counter() - t0) * 1000.0, 1)
    ev = {
        "feature": feature,
        "test_module": test_mod,
        "fixture": fixture_name,
        "unittest_rc": p_test.returncode,
        "spirv_val_ok": spv_val_ok,
        "spirv_val_err": spv_val_err,
        "ok": p_test.returncode == 0 and spv_val_ok in (True, None),
        "ms": dt,
        "unittest_tail": (p_test.stdout + p_test.stderr).strip().splitlines()[-10:],
    }
    out_file = os.path.join(evidence_dir, f"{feature}_{'pass' if ev['ok'] else 'fail'}.json")
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ev, f, indent=2)
        f.write("\n")
    return ev, out_file


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify tools/rdna2spirv.py end to end.")
    ap.add_argument("cmd", choices=["doctor", "drive"])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Repository root")
    ap.add_argument("--feature", default="valu-ops", choices=sorted(FEATURES), help="Feature to verify")
    ap.add_argument("--evidence", required=True, help="Directory for JSON proof artifacts")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if args.cmd == "doctor":
        res = doctor(root)
        print(json.dumps(res, indent=2))
        return 0 if res["rdna2spirv_exists"] and res["semantic_support_exists"] else 1

    ev, out_path = drive_feature(root, args.feature, os.path.abspath(args.evidence))
    print(f"verify-translator [{args.feature}]: {'PASS' if ev['ok'] else 'FAIL'} ({ev['ms']} ms) -> {out_path}")
    return 0 if ev["ok"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
