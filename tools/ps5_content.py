"""Inventory readable assets and replay AnyPS5 shader captures."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import uuid
import zlib

from ps5_common import compiler_directory, digest


ATRAC9 = bytes.fromhex('d242e147ba368d4d88fc61654f8c836c')
PNG = b'\x89PNG\r\n\x1a\n'
SPIRV = b'\x03\x02\x23\x07'
ASSET_SUFFIXES = {'.at9', '.awc', '.bk2', '.dds', '.glb', '.png', '.rpf', '.spv', '.ttpf', '.wav'}


def source_files(root):
    pending = [root]
    while pending:
        for entry in sorted(pending.pop().iterdir()):
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f'Input contains a link or reparse point: {entry}')
            if stat.S_ISDIR(info.st_mode):
                pending.append(entry)
            elif stat.S_ISREG(info.st_mode):
                yield entry, info.st_size
            else:
                raise ValueError(f'Input contains a non-regular entry: {entry}')


def inspect_wave(stream, size):
    stream.seek(4)
    declared, = struct.unpack('<I', stream.read(4))
    if declared + 8 != size:
        raise ValueError('RIFF size differs from the file size')
    fields = {}
    stream.seek(12)
    for _ in range(1024):
        offset = stream.tell()
        if offset == size:
            break
        tag, length = struct.unpack('<4sI', stream.read(8))
        end = offset + 8 + length + length % 2
        if end > size:
            raise ValueError('RIFF chunk extends past the file')
        if tag in (b'fmt ', b'data', b'fact', b'smpl'):
            if tag in fields:
                raise ValueError(f'Duplicate RIFF chunk {tag!r}')
            fields[tag] = length if tag == b'data' else stream.read(min(length, 60))
        stream.seek(end)
    else:
        raise ValueError('RIFF chunk count exceeds the supported limit')
    fmt = fields.get(b'fmt ', b'')
    if len(fmt) < 16 or b'data' not in fields:
        raise ValueError('Missing WAVE format or data chunk')
    encoding, channels, rate = struct.unpack_from('<HHI', fmt)
    if not channels or not rate:
        raise ValueError('Invalid WAVE sample rate or channel count')
    details = {'format_tag': encoding, 'channels': channels, 'sample_rate': rate, 'data_bytes': fields[b'data']}
    format_name = 'ATRAC9' if encoding == 0xfffe and fmt[24:40] == ATRAC9 else 'WAVE'
    fact = fields.get(b'fact', b'')
    if len(fact) >= 4:
        samples, = struct.unpack_from('<I', fact)
        details.update(declared_samples=samples, duration_seconds=samples / rate)
    if format_name == 'ATRAC9' and len(fact) == 12:
        _, base_skip, skip = struct.unpack('<3I', fact)
        block_align, = struct.unpack_from('<H', fmt, 12)
        samples_per_block, = struct.unpack_from('<H', fmt, 18)
        if base_skip == skip and block_align and samples_per_block and fields[b'data'] % block_align == 0:
            frames = fields[b'data'] // block_align * samples_per_block
            if samples > 0 and skip + samples <= frames:
                details.update(encoder_delay_samples=skip, stored_sample_frames=frames)
    sample_chunk = fields.get(b'smpl', b'')
    if len(sample_chunk) == 60 and struct.unpack_from('<I', sample_chunk, 28)[0] == 1:
        _, loop_type, start, end, _, repeats = struct.unpack_from('<6I', sample_chunk, 36)
        details['loop'] = {'type': loop_type, 'start_sample': start, 'end_sample_inclusive': end, 'play_count': repeats}
    return format_name, details


def inspect_rpf9(head, size):
    if len(head) < 0x40:
        raise ValueError('Truncated RPF9 header')
    header_size, table_split, trailer_size = struct.unpack_from('<3I', head, 4)
    if not 0x40 <= table_split <= header_size <= size or (table_split - 0x40) % 16 or (header_size - table_split) % 16:
        raise ValueError('RPF9 record table bounds or alignment are invalid')
    if trailer_size > size - header_size:
        raise ValueError('RPF9 declared trailer exceeds the remaining file')
    version_word, platform_word, file_count = struct.unpack_from('<HHI', head, 0x20)
    return {'header_bytes': header_size, 'first_table_records': (table_split - 0x40) // 16,
            'second_table_records': (header_size - table_split) // 16, 'declared_trailer_bytes': trailer_size,
            'version_word': version_word, 'platform_word': platform_word, 'declared_files': file_count}


def read_records(stream, layout, count):
    record = struct.Struct(layout)
    remaining = count
    while remaining:
        batch = min(remaining, 4096)
        data = stream.read(batch * record.size)
        if len(data) != batch * record.size:
            raise ValueError('Truncated archive record table')
        yield from record.iter_unpack(data)
        remaining -= batch


def inspect_awc_opus_sfx(stream, stream_offset, tags_offset, entries):
    tracks = []
    ids = set()
    for index in range(entries):
        stream.seek(stream_offset + 4 * index)
        word, = struct.unpack('<I', stream.read(4))
        stream_id, count = word & 0x1fffffff, word >> 29
        if not stream_id or stream_id in ids:
            return []
        stream.seek(tags_offset)
        format_tags, data_sizes = [], []
        for tag, in read_records(stream, '<Q', count):
            tag_type, size, offset = tag >> 56, (tag >> 28) & 0xfffffff, tag & 0xfffffff
            if tag_type == 0x76:
                format_tags.append((offset, size))
            elif tag_type == 0x55:
                data_sizes.append(size)
        tags_offset += 8 * count
        if len(format_tags) != 1 or len(data_sizes) != 1 or data_sizes[0] % 2048:
            return []
        offset, size = format_tags[0]
        if size < 32:
            return []
        stream.seek(offset)
        fields = stream.read(32)
        codec, channels = fields[28:30]
        if codec != 0x0b or channels not in (1, 2):
            return []
        samples, = struct.unpack_from('<I', fields, 0)
        frames, = struct.unpack_from('<I', fields, 12)
        frames = frames or samples // channels
        if not frames:
            return []
        ids.add(stream_id)
        tracks.append({'stream_id': f'{stream_id:08X}', 'sample_frames': frames, 'channels': channels})
    return tracks


def inspect_awc(stream, head, size):
    endian = '<' if head.startswith(b'ADAT') else '>'
    flags, entries, payload_offset = struct.unpack_from(endian + 'III', head, 4)
    stream_offset = 16 + (2 * entries if flags & 0x10000 else 0)
    if endian == '<':
        stream_offset += (4 if flags & 0x400000 else 0) + (4 * (entries + 1) if flags & 0x200000 else 0)
    tags_offset = stream_offset + 4 * entries
    if flags & 0xff00ffff != 0xff000001 or not entries or not tags_offset <= payload_offset <= size:
        raise ValueError('AWC flags or stream table bounds are invalid')
    stream.seek(stream_offset)
    entries_table = read_records(stream, endian + 'I', entries)
    first, = next(entries_table)
    tag_count = (first >> 29) + sum(word >> 29 for word, in entries_table)
    if not tag_count or tags_offset + 8 * tag_count > payload_offset:
        raise ValueError('AWC tag table exceeds the declared header')
    kinds = Counter()
    for word, in read_records(stream, endian + 'Q', tag_count):
        tag_size, offset = (word >> 28) & 0xfffffff, word & 0xfffffff
        if not tag_size or offset < payload_offset or offset + tag_size > size:
            raise ValueError('AWC tag payload bounds are invalid')
        kinds[f'{word >> 56:02x}'] += 1
    shape = {'byte_order': 'little' if endian == '<' else 'big', 'streams': entries, 'tag_counts': dict(kinds),
             'subsong_count': entries if first & 0x1fffffff else 1, 'first_payload_offset': payload_offset,
             'encryption_flags': flags & 0x000e0000, 'encrypted_audio': bool(flags & 0x00080000)}
    if (endian == '<' and first & 0x1fffffff and not shape['encrypted_audio']
            and kinds['76'] == kinds['55'] == entries):
        tracks = inspect_awc_opus_sfx(stream, stream_offset, tags_offset, entries)
        if tracks:
            shape['opus_sfx_streams'] = tracks
    return shape


def inspect_ttpf(stream, head, size):
    blocks, low_size, high_size, section = struct.unpack_from('<4I', head, 4)
    section *= 256
    unpacked_size = low_size | ((high_size & 0xff) << 32)
    if not blocks or not unpacked_size or not 24 + 8 * blocks <= section <= size - 16:
        raise ValueError('RDS5 block table or ETPF offset is out of range')
    stream.seek(24)
    stored_end, unpacked_end, codec = section, 0, 0
    block_codecs = Counter()
    for word, in read_records(stream, '<Q', blocks):
        next_stored = (word & 0x3ffffff) * 256
        next_unpacked = ((word >> 30) & 0x3fffff) * 4096
        if not stored_end < next_stored <= size or not unpacked_end < next_unpacked:
            raise ValueError('RDS5 block endpoints are out of order or out of range')
        block_codecs[codec] += 1
        stored_end, unpacked_end, codec = next_stored, next_unpacked, (word >> 26) & 0xf
    stream.seek(section)
    magic, _, textures, section_size_word = struct.unpack('<4sIII', stream.read(16))
    if magic != b'ETPF' or section + 16 + 16 * textures > size:
        raise ValueError('Missing ETPF header or truncated texture table')
    return {'declared_blocks': blocks, 'unpacked_bytes': unpacked_size, 'etpf_offset': section,
            'declared_textures': textures, 'etpf_size_word': section_size_word, 'block_codec_counts': dict(block_codecs),
            'stored_end': stored_end, 'unpacked_end': unpacked_end}


def inspect_header(stream, size):
    head = stream.read(148)
    if head.startswith(PNG):
        if len(head) < 33 or head[8:16] != b'\0\0\0\rIHDR' or zlib.crc32(head[12:29]) != int.from_bytes(head[29:33], 'big'):
            raise ValueError('Invalid PNG IHDR')
        width, height, depth, color = struct.unpack_from('>IIBB', head, 16)
        if not width or not height:
            raise ValueError('Invalid PNG dimensions')
        return 'PNG', {'width': width, 'height': height, 'bit_depth': depth, 'color_type': color}
    if head.startswith(b'DDS '):
        if len(head) < 128 or struct.unpack_from('<I', head, 4)[0] != 124 or struct.unpack_from('<I', head, 76)[0] != 32:
            raise ValueError('Invalid DDS header')
        height, width = struct.unpack_from('<II', head, 12)
        if not width or not height:
            raise ValueError('Invalid DDS dimensions')
        details = {'width': width, 'height': height, 'mip_levels': struct.unpack_from('<I', head, 28)[0] or 1}
        if head[84:88] == b'DX10':
            dxgi, resource_dimension, _, array_size, _ = struct.unpack_from('<5I', head, 128)
            details.update(dxgi_format=dxgi, resource_dimension=resource_dimension, array_size=array_size,
                           payload_bytes=size - 148)
        return 'DDS', details
    if head.startswith(b'RIFF') and head[8:12] == b'WAVE':
        return inspect_wave(stream, size)
    if head.startswith(SPIRV):
        if size < 20 or size % 4:
            raise ValueError('Truncated or unaligned SPIR-V')
        version, generator, bound, schema = struct.unpack_from('<4I', head, 4)
        if schema != 0 or not bound or version >> 16 != 1 or version & 255:
            raise ValueError('Invalid SPIR-V header')
        return 'SPIR-V', {'version': f'1.{(version >> 8) & 255}', 'generator': generator, 'id_bound': bound}
    if head.startswith(b'9FPR'):
        return 'RPF9', inspect_rpf9(head, size)
    if head.startswith((b'ADAT', b'TADA')):
        return 'AWC', inspect_awc(stream, head, size)
    if head.startswith(b'RDS5'):
        return 'TTPF', inspect_ttpf(stream, head, size)
    return None, {}


def inventory(root):
    assets, candidates, invalid = [], [], []
    files = 0
    for path, size in source_files(root):
        files += 1
        row = {'path': path.relative_to(root).as_posix(), 'bytes': size}
        try:
            with path.open('rb') as stream:
                format_name, details = inspect_header(stream, size)
                if format_name:
                    row.update(format=format_name, **details)
                    if format_name in ('PNG', 'DDS', 'ATRAC9', 'WAVE', 'SPIR-V'):
                        stream.seek(0)
                        row['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
                    assets.append(row)
                elif path.suffix.lower() in ASSET_SUFFIXES:
                    candidates.append(row)
        except (ValueError, struct.error) as error:
            invalid.append({**row, 'error': str(error)})
    return {'source': str(root), 'files_checked': files, 'format_counts': dict(Counter(row['format'] for row in assets)),
            'assets': sorted(assets, key=lambda row: row['path']), 'unconfirmed_candidates': sorted(candidates, key=lambda row: row['path']),
            'invalid_headers': sorted(invalid, key=lambda row: row['path'])}


def begin_output(source, output, report):
    if os.path.lexists(output):
        raise FileExistsError('Output directory already exists')
    output = output.resolve()
    if os.path.lexists(output) or output == source or source in output.parents:
        raise ValueError('Output must be a new directory outside the source directory')
    work = output.with_name(output.name + '.incomplete-' + uuid.uuid4().hex)
    work.mkdir(parents=True)
    report['incomplete_directory'] = str(work)
    return output, work


def publish_output(output, work, report):
    if os.path.lexists(output):
        raise ValueError('Output directory appeared during processing')
    work.rename(output)
    report.pop('incomplete_directory')
    report['output_directory'] = str(output)


def read_compute_capture(path):
    limit = 36 * 1024 * 1024
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if not 104 <= len(data) <= limit or data[:12] != b'APS5COMP\x01\x00\x00\x00':
        raise ValueError('Unsupported AnyPS5 compute capture signature, version or size')
    _, _, subgroup, vendor, device, driver, api = struct.unpack_from('<7I', data, 12)
    cursor = 104

    def take(size):
        nonlocal cursor
        end = cursor + size
        if end > len(data):
            raise ValueError('Truncated compute capture record')
        start, cursor = cursor, end
        return data[start:end]

    def count(maximum):
        value, = struct.unpack('<I', take(4))
        if value > maximum:
            raise ValueError('Compute capture record count is out of range')
        return value

    name = take(count(255)).decode('utf-8', errors='replace')
    descriptor_sets = count(8)
    total_bindings = 0
    for _ in range(descriptor_sets):
        take(4)
        bindings = count(4096 - total_bindings)
        total_bindings += bindings
        take(20 * bindings)
    push_constants = count(1)
    take(12 * push_constants)
    specialization = count(4096)
    take(8 * specialization)
    shader = take(4 * count(8 * 1024 * 1024))
    if len(shader) < 20 or not shader.startswith(SPIRV) or cursor != len(data):
        raise ValueError('Missing SPIR-V module or unexpected trailing compute capture bytes')
    return shader, {'gpu': name, 'vendor_id': vendor, 'device_id': device, 'driver_version': driver,
                    'vulkan_api_version': api, 'required_subgroup_size': subgroup, 'descriptor_sets': descriptor_sets,
                    'descriptor_bindings': total_bindings, 'push_constant_ranges': push_constants,
                    'specialization_constants': specialization}


def anyps5_graphics_tool(build, name):
    build = build.resolve(strict=True)
    executable = build / 'core/libs/prx/libSceAgcDriver' / (name + '.exe' if os.name == 'nt' else name)
    if not executable.is_file():
        raise ValueError(f'The specified AnyPS5 build has no {name} executable')
    env = os.environ.copy()
    if os.name == 'nt':
        libraries = build / 'core/libs/libs'
        env['PATH'] = os.pathsep.join([str(libraries / 'unpatched'), str(libraries),
                                      str(compiler_directory(build)), env.get('PATH', '')])
    return executable, env


def replay_shaders(source, output, build, target, report):
    requests = sorted(path for path, _ in source_files(source) if path.suffix.lower() in ('.req', '.aps5cp'))
    if not requests:
        raise ValueError('No shader requests or compute captures found in the supplied directory')
    validator = shutil.which('spirv-val')
    if validator is None:
        raise ValueError('spirv-val is required on PATH')
    env = os.environ.copy()
    if any(path.suffix.lower() == '.req' for path in requests):
        if build is None:
            raise ValueError('--anyps5-build is required for .req shader requests')
        executable, env = anyps5_graphics_tool(build, 'agc_shader_replay')
        report['replay_binary_sha256'] = digest(executable)
    output, work = begin_output(source, output, report)
    rows = []
    report.update(validator_sha256=digest(Path(validator)), validation_target=target, requests=rows)
    for request in requests:
        relative = request.relative_to(source)
        destination = work / relative.parent
        destination.mkdir(parents=True, exist_ok=True)
        row = {'request': relative.as_posix(), 'request_sha256': digest(request), 'validated': False}
        rows.append(row)
        shader = destination / (request.name + '.spv')
        if request.suffix.lower() == '.aps5cp':
            try:
                shader_bytes, row['capture'] = read_compute_capture(request)
            except (ValueError, struct.error) as error:
                row['error'] = str(error)
                continue
            shader.write_bytes(shader_bytes)
        else:
            mode = ['--prepare'] if request.name.endswith('.prepare.req') else []
            result = subprocess.run([str(executable), *mode, '--spv', str(request)], cwd=destination, env=env,
                                    capture_output=True, timeout=120)
            (destination / (request.name + '.replay.log')).write_bytes(result.stdout + result.stderr)
            row['replay_exit_code'] = result.returncode
            if result.returncode != 0 or not shader.is_file():
                continue
        validation = subprocess.run([validator, '--target-env', target, str(shader)], capture_output=True, timeout=120)
        (destination / (request.name + '.validation.log')).write_bytes(validation.stdout + validation.stderr)
        row['validation_exit_code'] = validation.returncode
        if digest(request) != row['request_sha256']:
            row['error'] = 'Source shader request changed during replay'
        elif validation.returncode == 0:
            row.update(validated=True, shader=shader.relative_to(work).as_posix(), shader_sha256=digest(shader), bytes=shader.stat().st_size)
    if not all(row['validated'] for row in rows):
        report['error'] = 'Shader replay or SPIR-V validation failed. See the logs in the incomplete directory.'
        return 5
    publish_output(output, work, report)
    return 0


def verify_pipelines(source, output, build, gpu, report):
    captures = sorted(path for path, _ in source_files(source) if path.suffix.lower() == '.aps5cp')
    if not captures:
        raise ValueError('No AnyPS5 compute pipeline captures found')
    executable, env = anyps5_graphics_tool(build, 'agc_compute_pipeline_replay')
    env.update(ANYPS5_REQUIRE_VULKAN='1', ANYPS5_NO_SHADER_CACHE='1', APS5_NO_PIPELINE_JOURNAL='1',
               APS5_PIPELINE_STATS='0')
    env.pop('APS5_DUMP_COMPUTE_PIPELINES', None)
    if gpu:
        env['ANYPS5_GPU'] = gpu
    gpu = env.get('ANYPS5_GPU')
    validation_requested = 'VK_LAYER_KHRONOS_validation' in env.get('VK_INSTANCE_LAYERS', '')
    if validation_requested:
        current_debug = env.get('VK_LOADER_DEBUG', '')
        env['VK_LOADER_DEBUG'] = ','.join(part for part in (current_debug, 'layer') if part)

    output, work = begin_output(source, output, report)
    rows = []
    report.update(build_directory=str(build.resolve()), executable_sha256=digest(executable),
                  requested_gpu=gpu, capture_checks=rows,
                  validation_layer_requested=validation_requested)
    for source_capture in captures:
        relative = source_capture.relative_to(source)
        destination = work / relative.parent
        destination.mkdir(parents=True, exist_ok=True)
        evidence = destination / (source_capture.name + '.pipeline.json')
        log_path = destination / (source_capture.name + '.pipeline.log')
        row = {'capture': relative.as_posix(), 'capture_sha256': digest(source_capture), 'verified': False}
        rows.append(row)
        process = subprocess.run([str(executable), str(source_capture), '--output', str(evidence)],
                                 cwd=destination, env=env, capture_output=True, timeout=120)
        log = process.stdout + process.stderr
        log_path.write_bytes(log)
        layer_loaded = b'Insert instance layer "VK_LAYER_KHRONOS_validation"' in log
        validation_error = any(token in log for token in (b'VUID-', b'SYNC-HAZARD', b'Validation Error'))
        row.update(exit_code=process.returncode, validation_layer_loaded=layer_loaded, validation_error=validation_error,
                   log=log_path.relative_to(work).as_posix())
        if process.returncode != 0:
            replay_errors = [line.removeprefix('compute pipeline replay: ') for line in
                             process.stderr.decode('utf-8', errors='replace').splitlines()
                             if line.startswith('compute pipeline replay: ')]
            row['error'] = replay_errors[-1] if replay_errors else 'AnyPS5 pipeline replay failed'
            continue
        try:
            info = json.loads(evidence.read_text(encoding='utf-8'))
            row.update(captured_device=info['captured_device'], replay_device=info['replay_device'],
                       spirv_words=info['spirv_words'], dispatch_count=info['dispatch_count'],
                       required_subgroup_size=info['required_subgroup_size'])
            if info['dispatch_count'] != 0 or info['application_cache_policy'] != 'fresh':
                raise ValueError('Expected a fresh pipeline creation with no dispatch')
            if gpu and gpu.casefold() not in info['replay_device']['name'].casefold():
                raise ValueError('Replay selected a different GPU')
            if validation_error or (validation_requested and not layer_loaded):
                raise ValueError('Vulkan validation reported an error or its requested layer was unavailable')
            if digest(source_capture) != row['capture_sha256']:
                raise ValueError('Source pipeline capture changed during verification')
        except (OSError, ValueError, KeyError) as error:
            row['error'] = str(error)
            continue
        row.update(verified=True, replay_report=evidence.relative_to(work).as_posix())
    if not all(row['verified'] for row in rows):
        report['error'] = 'Pipeline creation or verification failed. See the logs in the incomplete directory.'
        return 5
    publish_output(output, work, report)
    return 0
