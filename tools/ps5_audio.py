"""Export readable PS5 audio assets into verified PCM WAV files."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

from ps5_content import begin_output, inventory, publish_output
from ps5_common import digest


def verify_wav(path, samples, rate, channels):
    if min(samples, rate, channels) <= 0:
        raise ValueError('Decoder returned invalid audio geometry')
    with wave.open(str(path), 'rb') as audio:
        geometry = (audio.getnframes(), audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        if geometry != (samples, rate, channels, 2):
            raise ValueError('Decoded WAV differs from the declared PCM16 playback geometry')
        audio.setpos(samples - 1)
        if len(audio.readframes(1)) != channels * 2:
            raise ValueError('Decoded WAV data is truncated')


def export_atrac9(source, output, track, decoder):
    if 'encoder_delay_samples' not in track:
        raise ValueError('Unsupported ATRAC9 sample count or delay metadata')
    path = Path(track['path'])
    first, samples = track['encoder_delay_samples'], track['declared_samples']
    wav = output / path.parent / (path.name + '.wav')
    wav.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run([decoder, '-nostdin', '-n', '-v', 'error', '-xerror', '-err_detect', 'explode',
                              '-protocol_whitelist', 'file', '-f', 'wav', '-i', str(source / path), '-map', '0:a:0',
                              '-af', f'atrim=start_sample={first}:end_sample={first + samples}', '-map_metadata', '-1',
                              '-c:a', 'pcm_s16le', str(wav)], cwd=wav.parent, capture_output=True, timeout=180)
    (wav.parent / (path.name + '.decode.log')).write_bytes(process.stderr)
    if process.returncode != 0:
        raise ValueError(f'FFmpeg failed with exit code {process.returncode}')
    verify_wav(wav, samples, track['sample_rate'], track['channels'])
    return {'decoder_exit_code': process.returncode, 'wav': wav.relative_to(output).as_posix(),
            'wav_sha256': digest(wav), 'sample_frames': samples}


def run_audio_command(command, log, tool):
    process = subprocess.run(command, cwd=log.parent, capture_output=True, timeout=180)
    log.write_bytes(process.stdout + process.stderr)
    if process.returncode != 0:
        raise ValueError(f'{tool} failed with exit code {process.returncode}. See {log.name}')
    return process.stdout


def require_readable_awc(bank):
    with bank.open('rb') as stream:
        header = stream.read(8)
    if len(header) != 8 or header[:4] not in (b'ADAT', b'TADA'):
        raise ValueError('AWC signature changed during export')
    endian = 'little' if header.startswith(b'ADAT') else 'big'
    if int.from_bytes(header[4:8], endian) & 0x00080000:
        raise ValueError('Encrypted AWC audio data requires readable input')


def export_awc(source, output, track, decoder, row):
    path = Path(track['path'])
    bank = source / path
    require_readable_awc(bank)
    directory = output / path
    directory.mkdir(parents=True)
    subsongs = row['subsongs'] = []
    total = track['subsong_count']
    for index in range(1, total + 1):
        stem = f'{index:04d}'
        wav = directory / (stem + '.wav')
        prefix = [str(decoder), '-i', '-s', str(index)]
        metadata = json.loads(run_audio_command([*prefix, '-I', str(bank)], directory / (stem + '.metadata.log'), 'vgmstream'))
        stream = metadata['streamInfo']
        if (stream['index'], stream['total']) != (index, total) or 'awc' not in metadata['metadataSource'].casefold():
            raise ValueError(f'vgmstream reported an unexpected AWC subsong at index {index}')
        samples, rate, channels = metadata['playSamples'], metadata['sampleRate'], metadata['channels']
        run_audio_command([*prefix, '-W', '1', '-o', str(wav), str(bank)], directory / (stem + '.decode.log'), 'vgmstream')
        verify_wav(wav, samples, rate, channels)
        subsongs.append({'index': index, 'encoding': metadata['encoding'], 'sample_frames': samples,
                         'sample_rate': rate, 'channels': channels, 'wav': wav.relative_to(output).as_posix(),
                         'wav_sha256': digest(wav)})


def export_awc_opus(source, output, track, extractor, ffmpeg, row):
    path = Path(track['path'])
    bank = source / path
    require_readable_awc(bank)
    directory = output / path
    directory.mkdir(parents=True)
    run_audio_command([sys.executable, '-I', str(extractor), str(bank), str(directory), '--jobs', '1'],
                      directory / 'opus-export.log', 'Leonida AWC extractor')
    ogg_directory = directory / path.stem
    expected = {entry['stream_id'] + '.ogg' for entry in track['opus_sfx_streams']}
    if not ogg_directory.is_dir() or {item.name for item in ogg_directory.iterdir()} != expected:
        raise ValueError('Opus outputs do not match the AWC subsong IDs')
    subsongs = row['subsongs'] = []
    for index, entry in enumerate(track['opus_sfx_streams'], 1):
        stream_id = entry['stream_id']
        ogg = ogg_directory / (stream_id + '.ogg')
        wav = directory / (stream_id + '.wav')
        run_audio_command([ffmpeg, '-nostdin', '-n', '-v', 'error', '-xerror', '-protocol_whitelist', 'file',
                           '-f', 'ogg', '-i', str(ogg), '-map', '0:a:0', '-map_metadata', '-1', '-c:a', 'pcm_s16le', str(wav)],
                          directory / (stream_id + '.decode.log'), 'FFmpeg')
        verify_wav(wav, entry['sample_frames'], 48000, entry['channels'])
        subsongs.append({'index': index, 'stream_id': stream_id, 'encoding': 'Opus',
                         'sample_frames': entry['sample_frames'], 'sample_rate': 48000, 'channels': entry['channels'],
                         'ogg': ogg.relative_to(output).as_posix(), 'ogg_sha256': digest(ogg),
                         'wav': wav.relative_to(output).as_posix(), 'wav_sha256': digest(wav)})


def export_audio(source, output, report, awc_decoder=None, awc_opus_tool=None):
    scanned = inventory(source)
    assets = [row for row in scanned['assets'] if row['format'] in ('ATRAC9', 'AWC')]
    skipped = report['omitted_files'] = [
        {'source': row['path'], 'reason': 'Encrypted AWC audio data'} for row in assets if row.get('encrypted_audio')
    ]
    skipped += [{'source': row['path'], 'reason': row['error']} for row in scanned['invalid_headers']
                if Path(row['path']).suffix.lower() in ('.awc', '.at9')]
    tracks = [row for row in assets if not row.get('encrypted_audio')]
    if not tracks:
        raise ValueError('No readable ATRAC9 or AWC audio found')

    ffmpeg = None
    if any(row['format'] == 'ATRAC9' or row.get('opus_sfx_streams') for row in tracks):
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise ValueError('ffmpeg is required on PATH for ATRAC9 or Opus export')
        report['decoder_sha256'] = digest(Path(ffmpeg))
    if any(row.get('opus_sfx_streams') for row in tracks):
        if awc_opus_tool is None or not awc_opus_tool.is_file():
            raise ValueError('Leonida awc_opus.py is required for the recognized Opus SFX profile')
        awc_opus_tool = awc_opus_tool.resolve(strict=True)
        report['opus_extractor_sha256'] = digest(awc_opus_tool)
    if any(row['format'] == 'AWC' and not row.get('opus_sfx_streams') for row in tracks):
        awc_decoder = awc_decoder or shutil.which('vgmstream-cli')
        if awc_decoder is None or not Path(awc_decoder).is_file():
            raise ValueError('A vgmstream-cli executable is required for AWC export')
        awc_decoder = Path(awc_decoder).resolve(strict=True)
        report['awc_decoder_sha256'] = digest(awc_decoder)

    output, work = begin_output(source, output, report)
    rows = report['tracks'] = []
    for track in tracks:
        path = source / track['path']
        row = {'source': track['path'], 'format': track['format'], 'exported': False}
        rows.append(row)
        try:
            row['source_sha256'] = track.get('sha256') or digest(path)
            if track['format'] == 'ATRAC9':
                row.update(export_atrac9(source, work, track, ffmpeg))
            elif track.get('opus_sfx_streams'):
                row['profile'] = 'Leonida Opus SFX'
                export_awc_opus(source, work, track, awc_opus_tool, ffmpeg, row)
            else:
                export_awc(source, work, track, awc_decoder, row)
            if digest(path) != row['source_sha256']:
                raise ValueError('Source audio changed during export')
        except (OSError, ValueError, KeyError, EOFError, wave.Error, subprocess.TimeoutExpired) as error:
            row['error'] = str(error)
            continue
        row['exported'] = True
    if not all(row['exported'] for row in rows):
        report['error'] = 'Audio export failed. See the track results in the incomplete directory.'
        return 5
    publish_output(output, work, report)
    return 0
