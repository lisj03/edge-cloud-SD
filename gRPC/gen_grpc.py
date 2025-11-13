#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    here = Path(__file__).parent
    proto = here / 'sd.proto'
    if not proto.exists():
        print('sd.proto not found', file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.check_call([
            sys.executable, '-m', 'grpc_tools.protoc',
            f'-I{here}',
            f'--python_out={here}',
            f'--grpc_python_out={here}',
            str(proto)
        ])
        print('Generated gRPC stubs: sd_pb2.py, sd_pb2_grpc.py')
    except subprocess.CalledProcessError as e:
        print('Failed to generate stubs:', e, file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
