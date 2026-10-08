"""Verify and expand the versioned numeric data archives after cloning."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((R/'data/archive_manifest.json').read_text())
for d in manifest['parts']:
 p=R/d['path'];assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
 with zipfile.ZipFile(p) as z:
  for member in z.infolist():
   target=(R/member.filename).resolve()
   assert target.is_relative_to((R/'data').resolve()) and not member.is_dir(),member.filename
  z.extractall(R)
for d in manifest['original_files']:
 p=R/d['path'];assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
print('Verified and extracted',manifest['total_files'],'numeric data files.')
