"""Publish already-passed EXP14 setup; correct scanner escaping without rerunning GPU work."""
import prepare_integration as setup
commits=setup.read(setup.EXP/'candidate_commits.json')
assert commits=={'vllm-plugin-FL':'82c6a143f6984f946df42125f1a8699799e8045f','FlagGems':setup.GEMS}
checks=setup.read(setup.EXP/'route_cpu_checks.json')
assert checks['status']=='passed' and len(checks['checks'])==11
for name,expected in commits.items():
    assert setup.git(setup.WT/name,'rev-parse','HEAD')==expected
    assert not setup.git(setup.WT/name,'status','--porcelain')
gate=setup.read(setup.EXP/'candidate_validation.json')
for name,files in gate['code_hashes'].items():
    for relative,expected in files.items():assert setup.sha(setup.WT/name/relative)==expected
setup.step('Fix publication scanner double escaping that matched its own regex source; all 11 CPU checks already passed; source commits unchanged')
setup.publish_prepared(commits,checks['checks'])
