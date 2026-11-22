"""Compare a readable PS5 application tree with exported PlayGo metadata."""

import hashlib
import json
from pathlib import PurePosixPath
import struct

from ps5_content import source_files
from ps5_playgo import parse, path_hash


IDENTITY_FIELDS = ('titleId', 'contentId', 'contentVersion')
PLAYGO_FILES = ('playgo-chunk.dat', 'playgo-hash-table.dat', 'playgo-ficm.dat')


def read_identity(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f'{path} must hold a JSON object')
    return {key: data.get(key) for key in IDENTITY_FIELDS}, hashlib.sha256(raw).hexdigest()


def catalog_names(catalog, version, assignments):
    if catalog is None:
        return {}
    data = json.loads(catalog.read_bytes())
    names = {}
    for row in data['versions'][version]['matching_paths']:
        name = row['candidate']
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or path.as_posix() != name:
            raise ValueError(f'Invalid PlayGo catalog path: {name}')
        key = f'{path_hash(name):016x}'
        if row['hash'] != key or assignments.get(key) != row['chunk'] or key in names:
            raise ValueError(f'PlayGo catalog entry disagrees with the reference: {name}')
        names[key] = name
    return names


def compare_app0(app0, reference, catalog=None):
    expected_id, reference_digest = read_identity(reference / 'param.json')
    if not all(isinstance(value, str) and value for value in expected_id.values()):
        raise ValueError('Reference param.json is missing a title ID, content ID or content version')

    blobs = [(reference / name).read_bytes() for name in PLAYGO_FILES]
    try:
        playgo = parse(*blobs)
    except struct.error as error:
        raise ValueError(f'Truncated PlayGo reference: {error}') from error
    assignments = {row['path_hash']: row['chunk'] for row in playgo['file_assignments']}
    names = catalog_names(catalog, expected_id['contentVersion'], assignments)

    matches = {}
    skipped_paths = 0
    total_files = 0
    param = None
    for path, size in source_files(app0):
        total_files += 1
        relative = path.relative_to(app0).as_posix()
        if relative.casefold() == 'sce_sys/param.json':
            if param is not None:
                raise ValueError('Multiple application paths match sce_sys/param.json')
            param = path
        try:
            key = f'{path_hash(relative):016x}'
        except (UnicodeError, ValueError):
            skipped_paths += 1
            continue
        if key in assignments:
            matches.setdefault(key, []).append({'path': relative, 'bytes': size})

    observed_id, digest = read_identity(param) if param else (None, None)
    fields = {name: {'reference': value, 'application': observed_id.get(name) if observed_id else None}
              for name, value in expected_id.items()}
    identity_matches = observed_id == expected_id
    found_names = []
    absent_names = []
    for key, name in sorted(names.items(), key=lambda pair: pair[1]):
        paths = matches.get(key, [])
        if len(paths) == 1 and paths[0]['path'].casefold() == name.casefold() and paths[0]['bytes'] > 0:
            found_names.append(paths[0])
        else:
            absent_names.append(name)
    collisions = [{'hash': key, 'paths': rows} for key, rows in matches.items() if len(rows) > 1]
    result = {
        'reference': str(reference), 'reference_param_sha256': reference_digest,
        'identity': fields, 'identity_matches': identity_matches, 'param_bytes_match': digest == reference_digest,
        'reference_playgo_sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in zip(PLAYGO_FILES, blobs)},
        'application_files_seen': total_files, 'paths_outside_supported_hash_format': skipped_paths,
        'reference_path_hashes': len(assignments), 'matched_path_hashes': len(matches),
        'matched_paths': [{'hash': key, 'chunk': playgo['chunks'][assignments[key]]['label'], 'paths': rows}
                          for key, rows in sorted(matches.items())],
        'unmatched_path_hashes': [{'hash': key, 'chunk': playgo['chunks'][chunk]['label']}
                                  for key, chunk in assignments.items() if key not in matches],
        'catalog_paths_present': found_names, 'catalog_paths_missing_empty_or_ambiguous': absent_names,
        'ambiguous_matches': collisions, 'filesystem_payload_bytes_verified': False,
    }
    if not identity_matches:
        result['error'] = 'Application title ID, content ID or content version differs from the reference or is missing'
    elif absent_names or collisions:
        result['error'] = 'Known PlayGo paths are missing, empty or ambiguous'
    return result
