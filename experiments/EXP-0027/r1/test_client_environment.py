"""Regression: an armed server must not prevent benchmark CLI initialization."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

source = Path(__file__).with_name('service_flow_r4.py')
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_client_command')
receipts = []
scope = {'write': lambda path, value: receipts.append(value)}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
run = scope['run_client_command']
server = {'VLLM_FL_TOPP_ROW_PARALLEL': '1', 'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW': '1',
          'VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE': '/already/armed', 'PYTHONPATH': '/fixed', 'VLLM_PLUGINS': 'fl'}
getenv = lambda candidate: server
core = SimpleNamespace(env=getenv)
observed = []
def command(argv, folder, name, **kw):
    observed.append((argv, dict(core.env(True)), kw))
    return 17
core.run_command = command
argv = ['vllm', 'bench', 'serve', '--max-concurrency', '64']
assert run(core, argv, Path('/unused'), 'diagnostic', timeout=123) == 17
assert observed[0][0] is argv and observed[0][2] == {'timeout': 123}
expected = dict(server, VLLM_FL_TOPP_ROW_PARALLEL_SHADOW='0', VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE='')
assert observed[0][1] == expected and receipts[0] == expected
assert core.env is getenv and server['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW'] == '1'
assert server['VLLM_FL_TOPP_ROW_PARALLEL_SHADOW_ARM_FILE'] == '/already/armed'
def failed(*args, **kw):
    raise ValueError('injected client failure')
core.run_command = failed
try:
    run(core, argv, Path('/unused'), 'failure')
except ValueError:
    pass
else:
    raise AssertionError('Client failure was swallowed')
assert core.env is getenv
result = dict(status='passed', checks=['client_flags_isolated_argv_unchanged', 'server_environment_unmodified', 'environment_restored_after_client_failure'], scope='CPU role separation only; actual CLI preflight pending')
Path(__file__).with_name('client_environment_cpu_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
