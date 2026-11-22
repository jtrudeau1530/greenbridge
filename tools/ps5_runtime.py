"""Prepare native Windows output from a readable PS5 app0 and an AnyPS5 build."""

import os
from pathlib import Path
import subprocess
import sys

from ps5_app0 import check_app0, executable_header
from ps5_common import compiler_directory, copy_file, digest, native_pe
from ps5_content import begin_output, publish_output, source_files


def module_directories(source, extras):
    automatic = [Path(name) for name in ('sce_module', 'sce_modules', 'prx') if (source / name).is_dir()]
    if Path('sce_module') in automatic and Path('sce_modules') in automatic:
        raise ValueError('Both sce_module and sce_modules exist. Supply a coherent app0 tree.')
    selected = list(automatic)
    custom = []
    for name in extras:
        relative = Path(name)
        if relative.anchor or '..' in relative.parts or not (source / relative).is_dir():
            raise ValueError(f'Module directory must exist inside app0: {name}')
        if relative.as_posix().casefold() in {path.as_posix().casefold() for path in selected}:
            raise ValueError(f'Duplicate module directory: {name}')
        selected.append(relative)
        custom.append(relative)
    return selected, custom


def check_modules(source, files, selected):
    modules = []
    for path in files:
        if path.name.lower().endswith('.guest.prx'):
            raise ValueError(f'Input app0 contains a stale Windows guest module: {path.relative_to(source)}')
        if path.parent.relative_to(source) not in selected or path == source / 'eboot.bin':
            continue
        with path.open('rb') as stream:
            magic = stream.read(4)
        if magic == b'\x7fELF' or path.suffix.lower() in ('.prx', '.sprx'):
            executable_header(path, module=True)
            modules.append(path.relative_to(source))
    return modules


def verify_converted_modules(module_paths, stage, conversion_input, conversion_log):
    converted, missing = [], []
    for relative in module_paths:
        output = stage / 'app0' / relative.parent / (relative.name + '.guest.prx')
        if output.is_file():
            native_pe(output)
            converted.append(relative)
        else:
            missing.append(relative)
    if not missing:
        return len(converted), []

    messages = set(conversion_log.read_bytes().splitlines())
    skipped = []
    for relative in missing:
        matches = [
            {'path': relative.as_posix(), 'kept_module': kept.as_posix(), 'reason': reason}
            for kept in converted
            if relative.parent == kept.parent and relative.name.startswith(kept.name)
            for reason in ('same SONAME', 'same module names')
            if f'Skipping backup copy: {relative.as_posix()} ({reason} as {kept.as_posix()})'.encode('utf-8') in messages
        ]
        if len(matches) != 1:
            raise ValueError(f'A selected module was not converted: {relative}')
        skipped.append(matches[0])
    for relative in missing:
        (conversion_input / relative).unlink()
    return len(converted), skipped


def stage_libraries(libraries, compiler, native):
    native.mkdir()
    candidates = sorted(path for path in libraries.iterdir() if path.is_file() and path.suffix.lower() in ('.prx', '.dll'))
    candidates.extend(compiler / name for name in ('libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll'))
    records = []
    for file in candidates:
        native_pe(file)
        target = native / file.name
        if target.exists():
            if digest(target) != digest(file):
                raise ValueError(f'Conflicting native runtime files: {file.name}')
            continue
        records.append(dict(copy_file(file, target), relative_path=target.name))
    if not (native / 'libkernel.prx').is_file():
        raise ValueError('Native system-library bundle is missing libkernel.prx')
    return records


def stage_runtime(source, output, build, repository, extra_modules, report):
    if os.name != 'nt':
        raise ValueError('PS5 runtime staging targets Windows')
    files = sorted(path for path, _ in source_files(source))
    report['title_id'] = check_app0(str(source))
    selected, custom = module_directories(source, extra_modules)
    modules = check_modules(source, files, selected)
    build = build.resolve(strict=True)
    repository = repository.resolve(strict=True) if repository else build.parent
    relinker = build / 'core/relinker/relinker.exe'
    libraries = build / 'core/libs/libs'
    audit = repository / 'tools/import_audit.py'
    if not relinker.is_file() or not libraries.is_dir() or not audit.is_file():
        raise ValueError('Supply a matching AnyPS5 Windows build and source checkout with tools/import_audit.py')
    compiler = compiler_directory(build)
    report['relinker_sha256'] = digest(relinker)
    report['import_audit_tool_sha256'] = digest(audit)

    resolved_output = output.resolve()
    if any(root == resolved_output or root in resolved_output.parents for root in (repository, build)):
        raise ValueError('Output must be outside the AnyPS5 source checkout and build')
    output, stage = begin_output(source, output, report)
    conversion_input = stage / 'conversion-input'
    logs = stage / 'preparation-logs'
    logs.mkdir()
    environment = os.environ.copy()
    environment['PATH'] = os.pathsep.join([str(compiler), environment.get('PATH', '')])

    def run(label, command):
        with (logs / (label + '.stdout.log')).open('xb') as out, (logs / (label + '.stderr.log')).open('xb') as err:
            result = subprocess.run(list(map(str, command)), cwd=stage, stdout=out, stderr=err,
                                    timeout=600, env=environment, check=False)
        if result.returncode:
            raise ValueError(f'{label} failed with exit code {result.returncode}. See preparation-logs.')

    for directory in selected:
        (conversion_input / directory).mkdir(parents=True, exist_ok=True)
    inputs = {
        relative: copy_file(source / relative, conversion_input / relative)
        for relative in [Path('eboot.bin'), *modules]
    }
    report['conversion_inputs'] = [dict(info, relative_path=path.as_posix()) for path, info in inputs.items()]
    flags = ['--windows', '--registry', '--windows-diagnostics', '--no-patches', '--sce-module-path', conversion_input]
    for directory in custom:
        flags.extend(['--module-dir', directory])
    positional = ['--', conversion_input / 'eboot.bin', stage / 'app.exe']
    run('dry-run', [relinker, *flags, '--dry-run', *positional])
    run('conversion', [relinker, *flags, *positional])
    native_pe(stage / 'app.exe')
    converted, skipped = verify_converted_modules(modules, stage, conversion_input, logs / 'conversion.stdout.log')
    report.update(converted_module_count=converted, skipped_backup_modules=skipped)
    for path in files:
        relative = path.relative_to(source)
        staged = copy_file(path, stage / 'app0' / relative)
        if relative in inputs and staged['sha256'] != inputs[relative]['sha256']:
            raise ValueError(f'Executable changed between conversion and staging: {relative}')
    report['app0_files_staged'] = len(files)
    native = stage / 'libs'
    report['native_runtime_files'] = stage_libraries(libraries, compiler, native)
    registries = [stage / 'app.registry.json', *sorted(stage.glob('app.*.registry.json'))]
    audit_command = [sys.executable, audit, *registries, '--libs', native]
    for directory in selected:
        audit_command.extend(['--modules', conversion_input / directory])
    audit_command.extend(['--strict-stubs', '--strict-providers', '--strict-partial', '--json', stage / 'import-audit.json'])
    run('import-audit', audit_command)
    report['strict_import_audit_passed'] = True
    report['executable'] = 'app.exe'
    publish_output(output, stage, report)
    return 0
