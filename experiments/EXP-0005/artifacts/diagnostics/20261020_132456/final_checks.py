import subprocess,json
from pathlib import Path
commands=json.loads(Path('commands.json').read_text())
result={'status':'running','completed':[]}
try:
 for command in commands:
  subprocess.run(command,check=True,timeout=1200)
  result['completed'].append(command)
 result['status']='passed'
except BaseException as exc:
 result.update(status='failed',error=repr(exc))
 raise
finally:
 Path('final_checks.json').write_text(json.dumps(result,indent=2))
