#!/usr/bin/env python3
"""Download private GitHub Release weights using gh auth, then verify SHA256."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];m=json.loads((root/'checkpoints/manifest.json').read_text())
for name,info in m['files'].items():
 path=root/'checkpoints'/name
 if not path.exists():subprocess.run(['gh','release','download',m['release'],'--repo',m['repo'],'--pattern',name,'--dir',str(path.parent)],check=True)
 assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],f'Checkpoint mismatch: {name}'
 print('VERIFIED',name)
