"""CPU regression for diagnostic admission; does not test GPU numerics."""
import ast
import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def main(source_path):
    source = source_path.read_text()
    tree = ast.parse(source)
    budget_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_ShadowBudget')
    scope = {'Path': Path}
    exec(compile(ast.Module(body=[budget_node], type_ignores=[]), str(source_path), 'exec'), scope)
    Budget = scope['_ShadowBudget']
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append({'name': name, 'passed': True})
    def rejects(fn, kind):
        try:
            fn()
        except kind:
            return True
        return False
    with tempfile.TemporaryDirectory() as directory:
        arm = Path(directory) / 'workload.arm'
        budget = Budget(str(arm))
        check('invalid_path_rejected', all(rejects(lambda p=p: Budget(p), ValueError) for p in [None, '', 'relative.arm']))
        check('warmup_B64_does_not_spend_quota', not any(budget.should_check(64) for _ in range(1000)) and not budget.passed)
        check('prearm_record_rejected', rejects(lambda: budget.record_pass(64), RuntimeError))
        arm.touch(exist_ok=False)
        check('stale_arm_rejected_on_new_service', rejects(lambda: Budget(str(arm)), ValueError))
        old_checks = 0
        for b in range(17, 33):
            for _ in range(2):
                old_checks += 1
                assert budget.should_check(b)
                budget.record_pass(b)
        check('r0_failure_reproduced_r1_B64_reserved', old_checks == 32 and not old_checks < 32 and budget.should_check(64))
        for _ in range(2):
            assert budget.should_check(64)
            budget.record_pass(64)
        check('repeated_B64_bounded', not any(budget.should_check(64) for _ in range(1000)) and budget.passed[64] == 2)
        check('overquota_record_rejected', rejects(lambda: budget.record_pass(64), RuntimeError))
        check('failed_compare_would_not_consume', budget.should_check(33) and budget.should_check(33) and 33 not in budget.passed)
        for b in range(17, 65):
            while budget.should_check(b):
                budget.record_pass(b)
        check('all_48_batches_bounded_to_96', len(budget.passed) == 48 and sum(budget.passed.values()) == 96)
        check('unsupported_batches_excluded', all(not budget.should_check(b) for b in [0, 1, 16, 65, 128]))
    # Evaluate the actual install expression with an exploding constructor.
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_topp_row_parallel')
    assignment = next(n for n in install.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'shadow_budget' for t in n.targets))
    def forbidden(*args):
        raise AssertionError('Shadow-disabled branch touched diagnostic budget')
    check('shadow_off_skips_constructor_and_file', eval(compile(ast.Expression(assignment.value), '<shadow-off>', 'eval'), {'shadow': False, '_ShadowBudget': forbidden}) is None)
    result = {'status': 'passed', 'scope': 'CPU diagnostic budget regression only; GPU checks pending', 'checks': checks,
              'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'gpu_validation': 'not_run', 'deployed': False}
    Path(__file__).with_name('cpu_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('topp_row_parallel.py'))
    main(parser.parse_args().source)
