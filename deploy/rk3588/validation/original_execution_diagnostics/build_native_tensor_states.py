"""Root-only board build for the private CPython metadata prototype; no Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler-command', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() not in ('aarch64', 'arm64') or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Run only with the board production CPython 3.11')
    source = Path(__file__).with_name('native_tensor_states.c').resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / ('_v76_tensor_states' + sysconfig.get_config_var('EXT_SUFFIX'))
    includes = list(dict.fromkeys([sysconfig.get_path('include'), sysconfig.get_path('platinclude')]))
    if not any((Path(path) / 'Python.h').is_file() for path in includes):
        raise RuntimeError('Production Python.h is unavailable; do not use incompatible headers')
    command = shlex.split(args.compiler_command) + [
        '-std=c11', '-O3', '-fPIC', '-shared', '-Wall', '-Wextra', '-Werror',
        *['-I' + path for path in includes], str(source), '-o', str(output)]
    report = dict(schema='v76_private_native_tensor_state_build', status='started',
        source_sha256=digest(source), python=sys.version, machine=platform.machine(),
        command=command, includes=includes, torch_imported=False, model_or_SDK_executed=False)
    started = time.perf_counter()
    try:
        with (args.output_dir / 'compiler.log').open('wb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=120, check=False)
        report['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Compiler returned ' + str(result.returncode))
        if digest(source) != report['source_sha256']:
            raise RuntimeError('Native source changed during build')
        report.update(status='compiled_not_model_validated', source_unchanged=True,
            library=dict(file=output.name, bytes=output.stat().st_size, sha256=digest(output)))
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        (args.output_dir / 'build.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
