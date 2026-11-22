"""Use readable PS5 inputs to check, inspect or stage an application on a PC."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps5_app0 import check_app0
from ps5_audio import export_audio
from ps5_content import inventory, replay_shaders, verify_pipelines
from ps5_reference import compare_app0
from ps5_runtime import stage_runtime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='Validate a readable app0 and optionally compare PlayGo paths')
    check.add_argument('source', type=Path, help='Readable app0 directory')
    check.add_argument('--reference-sce-sys', type=Path, help='Exported param.json and three PlayGo tables')
    check.add_argument('--path-catalog', type=Path, help='JSON catalog of known names for paths in the reference')
    assets = commands.add_parser('assets', help='Inventory readable media and archive headers')
    assets.add_argument('source', type=Path, help='Directory to inventory')
    audio = commands.add_parser('audio', help='Export playable ATRAC9 and AWC audio')
    audio.add_argument('source', type=Path, help='Directory containing readable audio files')
    audio.add_argument('--out', type=Path, required=True, help='New audio output directory')
    audio.add_argument('--awc-decoder', type=Path, help='vgmstream-cli path, defaults to searching PATH')
    shaders = commands.add_parser('shaders', help='Replay shader requests or validate SPIR-V from compute captures')
    shaders.add_argument('source', type=Path, help='Directory containing .req or .aps5cp inputs')
    shaders.add_argument('--out', type=Path, required=True, help='New shader output directory')
    shaders.add_argument('--anyps5-build', type=Path, help='AnyPS5 build directory, required for .req inputs')
    shaders.add_argument('--target', default='vulkan1.3', help='spirv-val target environment')
    pipelines = commands.add_parser('pipelines', help='Verify Vulkan compute pipeline creation on the capture GPU')
    pipelines.add_argument('source', type=Path, help='Directory containing .aps5cp inputs')
    pipelines.add_argument('--out', type=Path, required=True, help='New pipeline report directory')
    pipelines.add_argument('--anyps5-build', type=Path, required=True, help='AnyPS5 build with SPIRV-Tools')
    pipelines.add_argument('--gpu', help='GPU name substring to select')
    runtime = commands.add_parser('runtime', help='Stage a Windows executable and audit its imports')
    runtime.add_argument('source', type=Path, help='Readable app0 with an ELF eboot.bin')
    runtime.add_argument('--out', type=Path, required=True, help='New Windows staging directory')
    runtime.add_argument('--anyps5-build', type=Path, required=True, help='AnyPS5 native Windows build')
    runtime.add_argument('--anyps5-source', type=Path, help='Matching AnyPS5 checkout, defaults to build parent')
    runtime.add_argument('--module-dir', action='append', default=[], help='Additional module directory relative to app0')
    args = parser.parse_args()
    if args.command == 'check' and args.path_catalog and not args.reference_sce_sys:
        check.error('--path-catalog requires --reference-sce-sys')
    return args


def run(args, source, report):
    if args.command == 'check':
        report['title_id'] = check_app0(str(source))
        if args.reference_sce_sys is not None:
            reference = args.reference_sce_sys.resolve(strict=True)
            comparison = report['reference_comparison'] = compare_app0(source, reference, args.path_catalog)
            if 'error' in comparison:
                report['error'] = comparison['error']
                return 3
    elif args.command == 'assets':
        report.update(inventory(source))
    elif args.command == 'audio':
        return export_audio(source, args.out, report, args.awc_decoder, Path(__file__).with_name('awc_opus.py'))
    elif args.command == 'shaders':
        return replay_shaders(source, args.out, args.anyps5_build, args.target, report)
    elif args.command == 'pipelines':
        return verify_pipelines(source, args.out, args.anyps5_build, args.gpu, report)
    elif args.command == 'runtime':
        return stage_runtime(source, args.out, args.anyps5_build, args.anyps5_source, args.module_dir, report)
    return 0


def main():
    args = parse_args()
    report = {'command': args.command, 'source': str(args.source)}
    try:
        source = args.source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError('Input must be a directory')
        report['source'] = str(source)
        code = run(args, source, report)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
        code = 2
    report['exit_code'] = code
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
