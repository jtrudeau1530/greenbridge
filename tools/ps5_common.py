"""Shared helpers for readable PS5 inputs and AnyPS5 build outputs."""

import hashlib
from pathlib import Path
import struct


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def copy_file(source, target):
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        while data := incoming.read(1024 * 1024):
            outgoing.write(data)
            hasher.update(data)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during staging: ' + str(source))
    return {'path': str(source), 'bytes': before.st_size, 'sha256': hasher.hexdigest(), 'mtime_ns': before.st_mtime_ns}


def native_pe(path):
    with path.open('rb') as source:
        head = source.read(64)
        if len(head) != 64 or head[:2] != b'MZ':
            raise ValueError('Native Windows runtime must contain PE files: ' + str(path))
        source.seek(struct.unpack_from('<I', head, 60)[0])
        if source.read(6) != b'PE\0\0\x64\x86':
            raise ValueError('Native Windows runtime is not x86-64 PE: ' + str(path))


def compiler_directory(build):
    for line in (build / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith(('CMAKE_CXX_COMPILER:FILEPATH=', 'CMAKE_CXX_COMPILER:STRING=', 'CMAKE_CXX_COMPILER:UNINITIALIZED=')):
            return Path(line.split('=', 1)[1]).parent
    raise ValueError('Cannot identify the runtime build compiler directory')
