#!/usr/bin/env python3
"""Select complete original B attention updates from real step1 metadata.

Only custom_quantize_layers changes. Reuse the exact original calibration
text and CONFIG; no .data contents, learned arrays, NN or SDK are accessed.
Requested FP16 placement and eligible INT8 MLP interiors require real compiler
evidence. Prior policies and generated payloads remain immutable.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

PARENT_SELECTOR_SHA = '25db1ca6c66fcea6d84b921c3f8f8382d0d16eeaa9d8f6f75d7ae5b9b804da72'
STEP1_SOURCE_SHA = 'a9b6f6b591e7ec7fef39f04ece167115445f445eecd87f996cdb5921035a20f6'
STEP1_B_SHA = 'b334bf5abe5935d6d5d316d3f46fc583fdba02bb0d6c4286321bc7514e8baa69'
BASE_POLICY_B_SHA = 'a37e9117af424b0f5101bce05453a5536f2b96c706773d88adfe2396d4d5b20c'
PREFIXES = ('frame0/batch0/', 'frame0/batch1/', 'global0/')
ATTENTION_OP_COUNTS = dict(exNorm=3, Conv=4, Reshape=2, Split=1, Mul=4, Add=2, exSDPAttention=1)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def parent_mapper(path=None):
    path = Path(path or Path(__file__).with_name('select_hybrid_precision_policy.py')).resolve(strict=True)
    require(sha(path) == PARENT_SELECTOR_SHA, 'Frozen boundary mapper differs')
    spec = importlib.util.spec_from_file_location('_v128_frozen_boundary_selector', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def derive_mapping(model, cfg, semantic, base_source=None):
    """Graph-edge closure only; compatible with the frozen mapper interface."""
    require(semantic['variant'] == 'B', 'Only the original generated B graph is selected')
    base = parent_mapper(base_source).derive_mapping(model, cfg, semantic)
    nodes = list(model.graph.node); by_name = {n.name: n for n in nodes}
    require(len(nodes) == 93 and len(by_name) == len(nodes), 'Actual frozen B generated graph differs')
    producer = {t: i for i, n in enumerate(nodes) for t in n.output}
    require(len(producer) == sum(len(n.output) for n in nodes), 'Duplicate generated tensor producer')
    initializers = {t.name for t in model.graph.initializer}
    catalog = cfg['quantize_parameters']
    selected_roles = {r['tensor']: set(r['roles']) for r in base['selector_rows']}
    attention_rows, eligible_mlp_rows, attention_outputs, sdpa_outputs = [], [], set(), []

    def closure(endpoint, boundary, prefix):
        pending, seen, terminals = [endpoint], set(), set()
        while pending:
            tensor = pending.pop()
            if tensor == boundary:
                terminals.add(tensor); continue
            if tensor in initializers: continue
            require(tensor in producer, 'Dynamic branch escapes its exact original highway: ' + tensor)
            index = producer[tensor]
            if index in seen: continue
            node = nodes[index]
            require(node.name.startswith(prefix), 'Branch crosses another original block: ' + node.name)
            seen.add(index); pending.extend(t for t in node.input if t)
        require(terminals == {boundary}, 'Original branch input dependency is incomplete')
        return sorted(seen)

    def node_row(index):
        n = nodes[index]
        return dict(index=index, name=n.name, op_type=n.op_type, inputs=list(n.input), outputs=list(n.output))

    for prefix in PREFIXES:
        residual = by_name[prefix + 'residual1']
        require(residual.op_type == 'Add' and len(residual.input) == 2, 'Original attention residual is required')
        indices = closure(residual.input[1], residual.input[0], prefix)
        branch = [nodes[i] for i in indices]
        require(len(indices) == 17 and dict(Counter(n.op_type for n in branch)) == ATTENTION_OP_COUNTS,
            'Complete original norm/QKV/QK-norm/RoPE/SDPA/projection branch required')
        require(branch[0].name == prefix + 'norm1#1' and branch[0].input[0] == residual.input[0] and
            branch[-1].name.startswith(prefix + 'attn.proj.matmul#') and
            any(n.name.startswith(prefix + 'attn.qkv.matmul#') for n in branch) and
            any(n.name.startswith(prefix + 'attn.q_norm#') for n in branch) and
            any(n.name.startswith(prefix + 'attn.k_norm#') for n in branch), 'Original attention semantic roles differ')
        outputs = [t for n in branch for t in n.output]
        require(len(outputs) == 19 and all(t in catalog and catalog[t]['dtype'] in ('int8', 'float16', 'float32')
            for t in outputs), 'Each requested attention output needs its real generated catalogue entry')
        require(not set(outputs) & set(base['requested_float16_layers']), 'Base boundary policy unexpectedly selects an attention interior')
        role = prefix + 'complete_attention_update'
        for tensor in outputs: selected_roles.setdefault(tensor, set()).add(role)
        attention_outputs.update(outputs)
        sdpa = [n for n in branch if n.op_type == 'exSDPAttention']
        require(len(sdpa) == 1 and len(sdpa[0].input) == 3, 'Original complete SDPA is required')
        sdpa_outputs.extend(sdpa[0].output)
        attention_rows.append(dict(original_role=role, original_residual_add=prefix + 'residual1',
            highway_input=residual.input[0], attention_update_input=residual.input[1],
            dynamic_terminals=[residual.input[0]], nodes=[node_row(i) for i in indices],
            requested_outputs=outputs, original_full_heads_keys_and_scale_unchanged=True))
        final = by_name[prefix + 'output']
        require(final.op_type == 'Add' and final.input[0] == residual.output[0], 'Original MLP skip dependency differs')
        mlp_indices = closure(final.input[1], final.input[0], prefix)
        require(any(nodes[i].name.startswith(prefix + 'norm2#') for i in mlp_indices) and
            any(nodes[i].name.startswith(prefix + 'mlp.fc1.matmul#') for i in mlp_indices) and
            any(nodes[i].name.startswith(prefix + 'mlp.fc2.matmul') for i in mlp_indices) and
            any(nodes[i].op_type == 'exGelu' for i in mlp_indices), 'Original complete MLP closure required')
        mlp_outputs = [t for i in mlp_indices for t in nodes[i].output]
        require(not set(mlp_outputs) & set(selected_roles), 'MLP interior was accidentally protected')
        eligible_mlp_rows.append(dict(original_role=prefix + 'MLP_update', highway_input=final.input[0],
            update_input=final.input[1], nodes=[node_row(i) for i in mlp_indices],
            eligible_unselected_catalogue_outputs=[t for t in mlp_outputs if t in catalog],
            INT8_placement_established=False))

    require(len(attention_outputs) == 57 and len(base['requested_float16_layers']) == 23 and len(selected_roles) == 80,
        'Exact base23 plus attention57 selector union required')
    rows = []
    for tensor in sorted(selected_roles):
        n = nodes[producer[tensor]]
        rows.append(dict(tensor=tensor, roles=sorted(selected_roles[tensor]), actual_node_name=n.name,
            actual_op_type=n.op_type, actual_inputs=list(n.input), original_generated_dtype=catalog[tensor]['dtype'],
            actual_conversion_attributes={a.name: a.s.decode('utf-8') for a in n.attribute if a.name in ('i_dtype', 'o_dtype')},
            catalogue_member=True))
    mapping = copy.deepcopy(base)
    mapping.update(requested_float16_layers={r['tensor']: 'float16' for r in rows}, selector_rows=rows,
        parent_selector_source_sha256=PARENT_SELECTOR_SHA, base_boundary_selector_count=23,
        attention_role_mapping=attention_rows, requested_attention_outputs=sorted(attention_outputs),
        attention_selector_count=57, SDPA_only_static_comparison_outputs=sorted(sdpa_outputs),
        SDPA_only_policy_created_or_executed=False, MLP_eligible_role_mapping=eligible_mlp_rows,
        update_branch_inputs_selected=True, attention_update_inputs_selected=True,
        MLP_update_inputs_selected=False, MLP_update_outputs_not_selected=True,
        original_base_boundary_policy_preserved=True,
        actual_precision_placement_requires_compiler_evidence=True)
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step1-receipt', type=Path, required=True); p.add_argument('--step1-sha256', required=True)
    p.add_argument('--base-selector-source', type=Path, default=Path(__file__).with_name('select_hybrid_precision_policy.py'))
    p.add_argument('--base-policy', type=Path, default=Path(__file__).parent / 'policy-B-r1/selector-policy.json')
    p.add_argument('--output-dir', type=Path, required=True); a = p.parse_args(); sys.dont_write_bytecode = True
    receipt = a.step1_receipt.resolve(strict=True)
    require(a.step1_sha256 == STEP1_B_SHA and sha(receipt) == STEP1_B_SHA, 'Exact successful original B-r3 step1 required')
    step1 = json.loads(receipt.read_text())
    require(step1['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step1' and
        step1['status'] == 'generated_manual_hybrid_step1_not_compiled_or_accepted' and step1['variant'] == 'B' and
        step1['script_sha256'] == STEP1_SOURCE_SHA and step1['sdk_released'] and step1['all_SDK_operations_succeeded'] and
        not step1['cleanup_errors'] and step1['configuration']['remove_reshape'] is False, 'Clean unchanged original calibration configuration required')
    base_policy_file = a.base_policy.resolve(strict=True)
    require(sha(base_policy_file) == BASE_POLICY_B_SHA, 'Frozen original B boundary policy differs')
    base_policy = json.loads(base_policy_file.read_text())
    require(base_policy['step1_receipt_sha256'] == STEP1_B_SHA and base_policy['variant'] == 'B', 'Base policy lineage differs')
    generated = {}
    for key, suffix in (('model', '.model'), ('data', '.data'), ('original_cfg', '.quantization.cfg')):
        matches = [r for r in step1['generated_files'] if r['file'].endswith(suffix)]
        require(len(matches) == 1, 'Unique actual generated artifact required')
        generated[key] = matches[0]; path = receipt.parent / matches[0]['file']
        require(not Path(matches[0]['file']).is_absolute() and '..' not in Path(matches[0]['file']).parts and
            path.stat().st_size == matches[0]['bytes'], 'Owned actual generated artifact differs')
        if key != 'data': require(sha(path) == matches[0]['sha256'], 'Generated metadata pin differs')
    sem_row = step1['public_semantic_index']; sem_path = receipt.parent / sem_row['file']
    require(sha(sem_path) == sem_row['sha256'], 'Original semantic index differs')
    semantic = json.loads(sem_path.read_text())
    require(semantic['variant'] == 'B' and semantic['graph_sha256'] == step1['graph']['sha256'], 'Original graph lineage differs')
    import onnx
    import yaml
    model = onnx.load_model(receipt.parent / generated['model']['file'], load_external_data=False)
    original_bytes = (receipt.parent / generated['original_cfg']['file']).read_bytes(); cfg = yaml.safe_load(original_bytes)
    require(cfg.get('custom_quantize_layers') == {}, 'Original unedited generated cfg is required')
    original_mapping = parent_mapper(a.base_selector_source).derive_mapping(model, cfg, semantic)
    require(all(base_policy.get(k) == v for k, v in original_mapping.items()), 'Recomputed frozen base mapping differs')
    mapping = derive_mapping(model, cfg, semantic, a.base_selector_source); del model
    header, separator, rest = original_bytes.partition(b'\n')
    require(header == b'custom_quantize_layers: {}' and separator and rest.startswith(b'quantize_parameters:'), 'Original calibration text layout differs')
    edited_bytes = yaml.safe_dump(dict(custom_quantize_layers=mapping['requested_float16_layers']), sort_keys=False).encode() + rest
    edited = yaml.safe_load(edited_bytes)
    require(edited['quantize_parameters'] == cfg['quantize_parameters'] and list(edited) == list(cfg), 'Calibration catalogue changed')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    edited_path = out / 'pair-B-attention-fp16.quantization.cfg'; edited_path.write_bytes(edited_bytes)
    generated['edited_cfg'] = dict(file=edited_path.name, bytes=len(edited_bytes), sha256=sha(edited_path))
    policy = dict(schema='v128_real_generated_B_full_attention_FP16_selector_policy',
        status='requested_policy_not_SDK_or_placement_verified', script_sha256=sha(Path(__file__)), variant='B',
        step1_receipt_sha256=STEP1_B_SHA, base_policy_sha256=BASE_POLICY_B_SHA, graph_sha256=step1['graph']['sha256'],
        generated_inputs=generated, public_semantic_index=sem_row, configuration=step1['configuration'],
        selectors_verified_against_actual_generated_model=True, only_custom_quantize_layers_changed=True,
        original_quantization_section_byte_preserved=True, original_quantization_section_sha256=hashlib.sha256(rest).hexdigest(),
        NN_SDK_or_data_payload_read=False, learned_tensor_arrays_extracted=False,
        protected_precision_placement_established=False, eligible_interior_INT8_established=False, requested_policy_only=True,
        heldout_numerical_results_used_for_selection=False, **mapping,
        placement_limits='Actual compiler/dump must prove all three attention updates and highway FP16 plus remaining MLP INT8. Requested cfg is neither accuracy nor speed acceptance.')
    target = out / 'selector-policy.json'; target.write_text(json.dumps(policy, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(variant='B', selector_count=80, attention_selector_count=57,
        policy_sha256=sha(target), edited_cfg_sha256=sha(edited_path)), sort_keys=True))


if __name__ == '__main__':
    main()
