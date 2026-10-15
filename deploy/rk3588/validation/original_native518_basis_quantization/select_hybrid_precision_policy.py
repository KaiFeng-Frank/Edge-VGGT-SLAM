#!/usr/bin/env python3
"""Select manual FLOAT16 boundaries from actual step1 protobuf/YAML metadata.

This reads no tensor values or .data payload and invokes no NN/SDK operation.
Selection is a requested policy; only a real step2/dump can prove placement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

STEP1_SOURCE_SHA = 'a9b6f6b591e7ec7fef39f04ece167115445f445eecd87f996cdb5921035a20f6'
RESIDUAL_NAMES = tuple(prefix + suffix for prefix in ('frame0/batch0/', 'frame0/batch1/', 'global0/')
                       for suffix in ('residual1', 'output'))
BASIS_NAMES = ('__v126_initial_basis', '__v126_restore_frame_output', '__v126_restore_global_output')
LAYOUT_OPS = {'Reshape', 'Transpose', 'Split', 'Slice', 'Concat', 'Identity', 'exDataConvert'}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def derive_mapping(model, cfg, semantic):
    """Use only graph edges, names, op kinds and cfg dtype metadata."""
    nodes = list(model.graph.node); by_name = {n.name: n for n in nodes}
    require(len(by_name) == len(nodes), 'Generated node names must be unique')
    initializers = {i.name for i in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    require(inputs == ['input'] and [o.name for o in model.graph.output] == ['frame_output', 'global_output'], 'Actual generated pair IO differs')
    producer = {}
    for node in nodes:
        for tensor in node.output:
            require(tensor and tensor not in producer, 'Duplicate generated tensor producer')
            producer[tensor] = node
    catalog = cfg['quantize_parameters']; require(isinstance(catalog, dict), 'Generated quantization catalogue required')
    public = {n['name']: n for n in semantic['nodes']}
    require(all(public[name]['op_type'] == 'Add' for name in RESIDUAL_NAMES), 'Six original residual roles required')
    residuals = {name: by_name[name] for name in RESIDUAL_NAMES}
    require(all(n.op_type == 'Add' and len(n.input) == 2 and len(n.output) == 1 for n in residuals.values()), 'Original residual Add identities changed')
    basis = {}
    if semantic['variant'] == 'B':
        for name in BASIS_NAMES:
            require(public[name]['op_type'] == 'MatMul', 'Original basis MatMul role differs')
            matches = [n for n in nodes if n.name.startswith(name) and n.op_type in ('Conv', 'MatMul')]
            require(len(matches) == 1, 'Actual generated basis producer not uniquely identified: ' + name)
            basis[matches[0].name] = name
    else:
        require(semantic['variant'] in ('A', 'F') and not any(n.name.startswith(BASIS_NAMES) for n in nodes), 'Unexpected basis operator')
    selected = {}; visits = set(); terminals = set()

    def walk(tensor, role):
        if tensor in inputs:
            terminals.add((role, tensor)); return
        require(tensor not in initializers and tensor in producer, 'Protected data edge lacks a real producer: ' + tensor)
        node = producer[tensor]
        require(tensor in catalog and catalog[tensor]['dtype'] in ('int8', 'float16', 'float32'), 'Protected tensor lacks a supported real catalogue entry: ' + tensor)
        selected.setdefault(tensor, set()).add(role)
        pair = (tensor, role)
        if pair in visits: return
        visits.add(pair)
        if node.name in residuals:
            # Only the original skip/highway input. Never recurse into the update.
            walk(node.input[0], role); return
        if node.name in basis:
            require(len(node.input) >= 2 and all(t in initializers for t in node.input[1:]), 'Basis producer has unexpected dynamic update inputs')
            walk(node.input[0], role); return
        require(node.op_type in LAYOUT_OPS, 'Protected highway crosses an unapproved arithmetic producer: ' + node.name)
        data = [t for t in node.input if t and t not in initializers]
        require(data and (node.op_type == 'Concat' or len(data) == 1), 'Unexpected generated layout inputs')
        for parent in data: walk(parent, role)

    residual_rows = []
    for name, node in residuals.items():
        walk(node.output[0], name)
        residual_rows.append(dict(original_role=name, actual_node_name=node.name, op_type=node.op_type,
            original_public_inputs=public[name]['inputs'], original_public_outputs=public[name]['outputs'],
            highway_input=node.input[0], update_input=node.input[1], output=node.output[0],
            update_producer=producer[node.input[1]].name, update_producer_op=producer[node.input[1]].op_type))
    for name in ('frame_output', 'global_output'): walk(name, 'canonical_exit/' + name)
    basis_rows = []
    for actual, original in basis.items():
        node = by_name[actual]
        for tensor in node.output: walk(tensor, 'basis_boundary/' + original)
        basis_rows.append(dict(original_role=original, actual_node_name=actual, op_type=node.op_type,
            inputs=list(node.input), outputs=list(node.output), original_public_inputs=public[original]['inputs'],
            original_public_outputs=public[original]['outputs']))
    require(all(row['update_input'] not in selected for row in residual_rows), 'Weighted update endpoint was accidentally protected')
    require('input_int8' in selected and all(name + '_int8' in selected for name in ('frame_output', 'global_output')), 'Actual input/output quantizer edge protection incomplete')
    rows = []
    for tensor in sorted(selected):
        node = producer[tensor]
        conversions = {a.name: a.s.decode('utf-8') for a in node.attribute if a.name in ('i_dtype', 'o_dtype')}
        rows.append(dict(tensor=tensor, roles=sorted(selected[tensor]), actual_node_name=node.name,
            actual_op_type=node.op_type, actual_inputs=list(node.input), original_generated_dtype=catalog[tensor]['dtype'],
            actual_conversion_attributes=conversions, catalogue_member=True))
    correspondences = []
    for original in semantic['nodes']:
        name = original['name']
        named = [n.name for n in nodes if n.name == name or n.name.startswith(name + '#') or n.name.startswith(name + '_')]
        correspondences.append(dict(original_index=original['index'], original_node_name=name,
            original_op_type=original['op_type'], named_generated_nodes=named,
            named_correspondence_is_not_a_fusion_or_precision_proof=True))
    return dict(requested_float16_layers={row['tensor']: 'float16' for row in rows}, selector_rows=rows,
        residual_role_mapping=residual_rows, basis_role_mapping=basis_rows,
        highway_terminals=[dict(role=r, tensor=t) for r, t in sorted(terminals)],
        public_operator_named_correspondence=correspondences, generated_node_count=len(nodes),
        generated_initializer_count=len(initializers), quantization_catalogue_count=len(catalog),
        update_branch_inputs_selected=False, all_selectors_are_actual_producer_outputs_and_catalogue_members=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step1-receipt', type=Path, required=True); p.add_argument('--step1-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True); a = p.parse_args(); sys.dont_write_bytecode = True
    receipt = a.step1_receipt.resolve(strict=True); require(sha(receipt) == a.step1_sha256, 'Step1 receipt pin differs')
    step1 = json.loads(receipt.read_text())
    require(step1['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step1' and
        step1['status'] == 'generated_manual_hybrid_step1_not_compiled_or_accepted' and step1['script_sha256'] == STEP1_SOURCE_SHA and
        step1['sdk_released'] and step1['all_SDK_operations_succeeded'] and not step1['cleanup_errors'], 'Clean actual frozen step1 required')
    generated = {}
    for key, suffix in (('model', '.model'), ('data', '.data'), ('original_cfg', '.quantization.cfg')):
        matches = [r for r in step1['generated_files'] if r['file'].endswith(suffix)]
        require(len(matches) == 1, 'Unique generated ' + suffix + ' required'); generated[key] = matches[0]
        path = receipt.parent / matches[0]['file']
        require(path.stat().st_size == matches[0]['bytes'], 'Generated file size differs')
        # .data is never opened here; its recorded pin will be rehashed board-side.
        if key != 'data': require(sha(path) == matches[0]['sha256'], 'Generated metadata pin differs')
    sem_row = step1['public_semantic_index']; sem_path = receipt.parent / sem_row['file']
    require(sha(sem_path) == sem_row['sha256'], 'Semantic index pin differs'); semantic = json.loads(sem_path.read_text())
    require(semantic['variant'] == step1['variant'] and semantic['graph_sha256'] == step1['graph']['sha256'], 'Source semantic graph identity differs')
    import onnx
    import yaml
    model = onnx.load_model(receipt.parent / generated['model']['file'], load_external_data=False)
    original_bytes = (receipt.parent / generated['original_cfg']['file']).read_bytes()
    cfg = yaml.safe_load(original_bytes); require(cfg.get('custom_quantize_layers') == {}, 'Unedited generated cfg required')
    mapping = derive_mapping(model, cfg, semantic); del model
    header, separator, rest = original_bytes.partition(b'\n')
    require(header == b'custom_quantize_layers: {}' and separator and rest.startswith(b'quantize_parameters:'), 'Unexpected cfg text layout; do not rewrite calibration text')
    replacement = yaml.safe_dump(dict(custom_quantize_layers=mapping['requested_float16_layers']), sort_keys=False).encode()
    edited_bytes = replacement + rest
    edited = yaml.safe_load(edited_bytes)
    require(edited['quantize_parameters'] == cfg['quantize_parameters'] and list(edited) == list(cfg), 'Calibration catalogue changed')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    edited_path = out / ('pair-' + step1['variant'] + '-protected.quantization.cfg'); edited_path.write_bytes(edited_bytes)
    generated['edited_cfg'] = dict(file=edited_path.name, bytes=len(edited_bytes), sha256=sha(edited_path))
    policy = dict(schema='v128_real_generated_hybrid_selector_policy', status='requested_policy_not_SDK_or_placement_verified',
        script_sha256=sha(Path(__file__)), variant=step1['variant'], step1_receipt_sha256=a.step1_sha256,
        graph_sha256=step1['graph']['sha256'], generated_inputs=generated, public_semantic_index=sem_row,
        selectors_verified_against_actual_generated_model=True, only_custom_quantize_layers_changed=True,
        original_quantization_section_byte_preserved=True, original_quantization_section_sha256=hashlib.sha256(rest).hexdigest(),
        NN_SDK_or_data_payload_read=False, protected_precision_placement_established=False, eligible_interior_INT8_established=False,
        requested_policy_only=True, **mapping,
        placement_limits='Actual step2 acceptance and compiled precision/conversion evidence remain required. Float IO or an _int8 suffix alone proves neither internal precision nor removal of a quantizer.')
    (out / 'selector-policy.json').write_text(json.dumps(policy, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(variant=step1['variant'], selector_count=len(mapping['requested_float16_layers']),
        policy_sha256=sha(out / 'selector-policy.json'), edited_cfg_sha256=sha(edited_path)), sort_keys=True))


if __name__ == '__main__':
    main()
