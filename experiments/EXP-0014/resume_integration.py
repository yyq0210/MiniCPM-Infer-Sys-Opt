"""Resume EXP14 after fixing only the synthetic log fixture's format matching."""
import prepare_integration as setup

commits=setup.read(setup.EXP/'candidate_commits.json')
assert commits['vllm-plugin-FL']=='82c6a14'+setup.git(setup.WT/'vllm-plugin-FL','rev-parse','HEAD')[7:]
assert commits['FlagGems']==setup.GEMS
for name,expected in commits.items():
    assert setup.git(setup.WT/name,'rev-parse','HEAD')==expected
    assert not setup.git(setup.WT/name,'status','--porcelain')
gate=setup.read(setup.EXP/'candidate_validation.json')
for name,files in gate['code_hashes'].items():
    for relative,expected in files.items():assert setup.sha(setup.WT/name/relative)==expected
setup.step('Resume CPU checks after correcting fixture substitution: runtime log has expanded layer names, not literal %s; inference source unchanged')
try:
    setup.finish_preparation(commits,
        'decode_warps=4 decode_stages=1 segments=16; layer=%s',
        'decode_warps=4 decode_stages=1 segments=16 decode_layout=1 decode_rows=8; layer=%s')
except BaseException as exc:
    setup.step('Resumed integration validation failed; preserve evidence',error=repr(exc))
    raise
