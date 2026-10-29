import ast, re
from common import sha, write

def layout_route(run,folder,reference,arm):
    assert arm in ("A","B")
    log=(folder/'server.log').read_text(errors='replace')
    pattern=re.compile(r'LeanScale MLP layout active: prefix=model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj) shape=\((\d+), (\d+)\) old_stride=\((\d+), (\d+)\) new_stride=\((\d+), (\d+)\) exact_values=True parameter_preserved=True bytes=(\d+)')
    matches=pattern.findall(log)
    assert len(matches)==(84 if arm=='B' else 0), ('Unexpected layout conversion count',arm,len(matches))
    if arm=='A': assert 'LeanScale MLP layout active:' not in log
    expected={(str(i),p) for i in range(42) for p in ['gate_up_proj','down_proj']}
    assert {(r[0],r[1]) for r in matches}==(expected if arm=='B' else set())
    for r in matches:
        n,k=(12288,2048) if r[1]=='gate_up_proj' else (2048,6144)
        assert tuple(map(int,r[2:]))==(n,k,k,1,1,n,n*k*2),r
    # Require generated matmuls to read engine weight arguments directly in
    # the new orientation. A fresh buffer/copy in the forward path is rejected.
    modules=[]
    for p in list((run/'candidate_cache').rglob('*.py'))+list((run/'inductor_cache').rglob('*.py')):
        text=p.read_text(errors='replace')
        if 'extern_kernels.mm' not in text:continue
        tree=ast.parse(text); inputs=set(); counts={'gate_up_proj':0,'down_proj':0}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Name) and node.value.id=='args':
                inputs.update(n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name))
        for node in ast.walk(tree):
            if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='extern_kernels'
                    and node.func.attr=='mm' and len(node.args)>=2):continue
            weight=node.args[1]
            if not (isinstance(weight,ast.Call) and isinstance(weight.func,ast.Name)
                    and weight.func.id=='reinterpret_tensor' and len(weight.args)>=3):continue
            try:shape=ast.literal_eval(weight.args[1]);stride=ast.literal_eval(weight.args[2])
            except (ValueError,TypeError):continue
            if shape not in [(2048,12288),(6144,2048)]:continue
            name='gate_up_proj' if shape==(2048,12288) else 'down_proj'
            expected_stride=(shape[1],1) if arm=='B' else (1,shape[0])
            assert stride==expected_stride,(p.name,name,stride)
            assert isinstance(weight.args[0],ast.Name) and weight.args[0].id in inputs, 'Forward weight packing detected'
            counts[name]+=1
        if any(counts.values()):
            modules.append(dict(path=str(p.relative_to(run)),sha256=sha(p),direct_weight_mm_counts=counts))
    # vLLM partitions at attention boundaries: one MLP per compiled block.
    unique_modules = {m['sha256']: m for m in modules}
    assert len(unique_modules) == 42, ('Expected 42 distinct MLP blocks', len(unique_modules))
    assert all(m['direct_weight_mm_counts'] == {'gate_up_proj': 1, 'down_proj': 1}
               for m in unique_modules.values())
    fx_coverage = []
    for graph_path in (run/'candidate_cache').rglob('computation_graph.py'):
        coverage = {}
        graph_tree = ast.parse(graph_path.read_text())
        for node in ast.walk(graph_tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch._C._nn.linear'
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Name)):
                continue
            match = re.search(r'modules_layers_modules_(\d+)_modules_mlp_modules_(gate_up_proj|down_proj)_parameters_weight_',
                              node.args[1].id, re.IGNORECASE)
            if match:
                key = (match[1], match[2])
                coverage[key] = coverage.get(key, 0) + 1
        if coverage:
            assert set(coverage) == expected and all(n == 1 for n in coverage.values()), coverage
            fx_coverage.append(dict(path=str(graph_path.relative_to(run)),sha256=sha(graph_path),
                                    logical_layers=42,gate_up_nodes=42,down_nodes=42))
    assert fx_coverage, 'No full FX graph with all 84 original MLP weight arguments'
    def capacity(p):
        values=re.findall(r'GPU KV cache size: ([\d,]+) tokens',p.read_text())
        assert len(values)==1,values
        return int(values[0].replace(',',''))
    old,new=capacity(reference/'server.log'),capacity(folder/'server.log')
    assert new>=.99*old, ('KV capacity regressed by more than predeclared 1%',old,new)
    evidence=dict(status='passed',arm=arm,converted_gate_up=42 if arm=='B' else 0,converted_down=42 if arm=='B' else 0,
                  exact_weight_values=True,parameter_objects_preserved=True,
                  one_time_conversions=84 if arm=='B' else 0,compiled_modules=modules,fx_coverage=fx_coverage,compiled_mlp_blocks=len(unique_modules),
                  reference_kv_tokens=old,candidate_kv_tokens=new,
                  kv_capacity_change_percent=(new/old-1)*100)
    write(folder/'layout_route_evidence.json',evidence)
    return evidence
