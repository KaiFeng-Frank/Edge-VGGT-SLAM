"""Private board-only ModelExecutor for fresh original native518 image pairs.

Inputs are the caller's original preprocessed pixels, never fixed replay files.
All original24 DINO layers/frame,48 aggregator blocks,4 camera updates,2 DPT
frames and the original pose codec execute. The frozen v100 baseline models
are used without the v105 candidate or reference substitution.
An optional single previous-tail cache retains only owned pre-final-norm DINO
output and exact prepared pixel bytes, before any pair-role special tokens.
SDK contexts are sequential and every returned canonical array owns its bytes.
This adapter does not establish SLAM precision, trajectory or realtime acceptance.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
S100 = BASE/'experiments/vggt-v100-source-r1'
CONFIG_SHA = '7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
PARENT_SHA = '253d8897dd34a050c47092d581432de4cc6f6cf66455b739343abf5a2d9210a7'
PROTOCOL_SHA = 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87'
CODEC_SHA = '845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b'
COUNTERS = ('contexts_created','contexts_destroyed','runs_completed',
            'output_buffers_acquired','output_buffers_released')
SHAPE = (2,3,392,518)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Native518Executor:
    """Synchronous original-model boundary; one owning thread, one optional previous-tail entry.

    Startup checkpoint/source/compile verification is separate from predict.
    Per-call timing includes preparation and every model hash/init/run/destroy;
    final canonical copies, validation and receipt hashes are recorded separately.
    The caller performs original518 preprocessing and controls the bounded NPU
    clock/thermal stage. This class never changes clocks, governors or drivers.
    """
    def __init__(self, config=S100/'full-native-sequential-config-r1.json', *,
                 config_sha256=CONFIG_SHA, frozen_source_dir=S100,
                 capture_intermediates=False, source_root=BASE/'src',
                 expected_npu_hz=1000000000, reuse_previous_tail=True, core_mask=1):
        require(platform.machine()=='aarch64' and
                b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
                'Actual model execution is restricted to RK3588/S')
        require(type(capture_intermediates) is bool and not capture_intermediates,
                'This fresh-pair adapter retains final outputs only')
        require(expected_npu_hz in (300000000,1000000000),'Bounded supported observed NPU clock required')
        require(type(reuse_previous_tail) is bool and core_mask in (1,2,4,7),'Explicit original cache/core contract required')
        self.reuse_previous_tail, self.core_mask = reuse_previous_tail, core_mask
        self._tail_pixels = self._tail_encoded = self._tail_key = None
        self._tail_output_sha256 = None
        self._cache_stats = dict(lookups=0,hits=0,misses=0,commits=0,drops=0)
        self.owner = threading.get_ident()
        self.closed = self.failed = self.active = False
        self.prefix = None
        self.pins = {}
        self._held = []
        self._totals = dict.fromkeys(COUNTERS,0)
        self.calls_attempted = self.calls_completed = 0
        self.cleanup_errors = []
        self.last_call = self.last_timings = {}
        self.device = 'cpu'
        self.repo = Path(source_root).resolve(strict=True)
        self.frozen = Path(frozen_source_dir).resolve(strict=True)
        self.expected_npu_hz = expected_npu_hz
        oldaff, oldenv, oldpath = os.sched_getaffinity(0), dict(os.environ), sys.path[:]
        tick = time.perf_counter()
        try:
            require(config_sha256==CONFIG_SHA,'Exact frozen baseline configuration required')
            conf = json.loads(self.pin(config,CONFIG_SHA).read_text())
            self.pin(__file__,sha(__file__))
            self.pin(self.frozen/'run_native_full_model_sequential.py',PARENT_SHA)
            for name,digest in conf['sources'].items():
                require(Path(name).name==name,'Frozen helper must be a direct sibling')
                self.pin(self.frozen/name,digest)
            os.sched_setaffinity(0,{4,5,6,7})
            os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            sys.path[:0] = [str(self.repo),str(self.repo/'third_party/vggt')]
            import numpy as np
            import torch
            from vggt_slam import model_execution
            self.pin(model_execution.__file__,PROTOCOL_SHA)
            self.protocol = model_execution
            self.capabilities = model_execution.ExecutionCapabilities('rk3588-native518-original-tail-reuse',False)
            self.np, self.torch = np, torch
            require(torch.get_num_threads()==torch.get_num_interop_threads()==1,
                    'Caller must configure original CPU execution with one Torch thread')
            prefix = module(self.frozen/'native_original_prefix.py','_v110_original_prefix')
            heads = module(self.frozen/'native_heads_streaming.py','_v110_original_heads')
            retained = module(self.frozen/'run_native_retained_groups.py','_v110_original_compilation_contract')
            sequential = module(self.frozen/'native_sequential_runtime.py','_v110_original_sequential_runtime')
            self.OwnedContext, self.NativeHeads = heads.OwnedContext, heads.NativeHeads
            self.NativeSequentialGroups = sequential.NativeSequentialGroups
            self.prefix = prefix.OriginalPrefix(repo=self.repo)
            self.pin(self.repo/'third_party/vggt/vggt/utils/pose_enc.py',CODEC_SHA)
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            self.decoder = pose_encoding_to_extri_intri
            self.dino_rows = []
            for i,row in enumerate(conf['dino']):
                manifest = self.pin(row['file'],row['sha256'])
                m = json.loads(manifest.read_text())
                require(m['schema']=='original_native518_single_block_FP16_compile' and
                        m['status']=='compiled_not_quality_accepted' and m['role']=='dino' and
                        m['logical_shape']==[1,1041,1024] and not m['configuration']['model_pruning'] and
                        not m['build_options']['do_quantization'] and m['sdk_released'] and not m['cleanup_errors'],
                        'Original complete DINO group compile required')
                e = json.loads((Path(m['onnx']['file']).parent/'export.json').read_text())
                require(e['group_index']==i and e['indices']==list(range(6*i,6*i+6)) and
                        e['graph']['sha256']==m['onnx']['sha256'] and e['proof']['original_parameter_count']==84 and
                        e['proof']['namespace_inverse_protobuf_checks_passed'] and e['source_and_inputs_unchanged'],
                        'All original DINO layers in order required')
                self.dino_rows.append(dict(file=manifest.parent/m['rknn']['file'],sha256=m['rknn']['sha256']))
            require(len(self.dino_rows)==4,'Four complete original DINO groups required')
            compiler = conf['aggregator_compiler']
            self.models,lineage = retained.load_compilations(
                [Path(row['file']) for row in conf['aggregator']],
                [row['sha256'] for row in conf['aggregator']],self.frozen/compiler,conf['sources'][compiler])
            self.initialization = dict(schema='v110_original_native518_ModelExecutor',
                script_sha256=self.pins[Path(__file__).resolve()],config_sha256=CONFIG_SHA,
                model_execution_protocol_sha256=PROTOCOL_SHA,parent_runner_sha256=PARENT_SHA,
                frozen_helper_source_hashes=conf['sources'],checkpoint=self.prefix.native['checkpoint'],
                original_prefix_source_hashes=self.prefix.native['source_hashes'],
                original_prefix_reference_sha256=prefix.NATIVE_SHA,
                original_DINO_reference_sha256=prefix.DINO_SHA,baseline_aggregator_lineage=lineage,
                selected_baseline_aggregator_models=copy.deepcopy(self.models),candidate_dependency=False,
                reuse_previous_tail=reuse_previous_tail,requested_core_mask=core_mask,
                full_original_DINO_layers_per_frame=24,full_original_aggregator_blocks=48,
                original_camera_refinements=4,original_depth_frames=2,
                initial_checkpoint_size_and_SHA_checked_before_deserialization=True,
                prefix_reference_images_are_lineage_only_not_prediction_inputs=True,
                source_images_or_preprocessed_reference_hash_not_used_to_select_inputs=True,
                startup_seconds=time.perf_counter()-tick,precision_trajectory_or_realtime_accepted=False)
        except BaseException as exc:
            self.failed = True
            exc.native_executor = self
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note('Partial executor cleanup: '+str(cleanup))
            raise
        finally:
            os.sched_setaffinity(0,oldaff)
            os.environ.clear();os.environ.update(oldenv)
            sys.path[:]=oldpath

    def pin(self, path, digest):
        path = Path(path).resolve(strict=True)
        require(sha(path)==digest,'Pinned original source/artifact changed: '+str(path))
        self.pins[path] = digest
        return path

    def verify_unchanged(self):
        if self.prefix is not None:
            self.prefix.verify_unchanged()
        for path,digest in self.pins.items():
            require(sha(path)==digest,'Pinned executor source/receipt changed: '+str(path))

    def _feature_key(self):
        # Encoder-only identity: CPU preparation/finalnorm source and checkpoint,
        # all four actual original DINO graph identities and selected core mask.
        # Camera/register role tokens are deliberately generated after this cache.
        state=dict(config_sha256=self.initialization['config_sha256'],checkpoint=self.initialization['checkpoint'],
            preparation_source_sha256=self.initialization['frozen_helper_source_hashes']['native_original_prefix.py'],
            DINO_models=[dict(file=str(row['file']),sha256=row['sha256']) for row in self.dino_rows],
            core_mask=self.core_mask,shape=list(SHAPE),dtype='float32',
            mean_raw_sha256=hashlib.sha256(self.prefix.mean.numpy().tobytes()).hexdigest(),
            std_raw_sha256=hashlib.sha256(self.prefix.std.numpy().tobytes()).hexdigest())
        return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()

    def _lookup_previous_tail(self, pixels, identity, pixel_sha256):
        if not self.reuse_previous_tail:return None
        self._cache_stats['lookups']+=1
        key=(identity,tuple(pixels.shape),str(pixels.dtype),pixel_sha256)
        if (self._tail_key==key and self._tail_encoded is not None and
                pixels.tobytes()==self._tail_pixels):
            value=self._tail_encoded
            require(value.dtype==self.np.float32 and value.shape==(1,1041,1024) and
                value.flags.owndata and value.flags.c_contiguous and not value.flags.writeable and
                self.np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest()==self._tail_output_sha256,
                'Owned original cached encoder output changed')
            self._cache_stats['hits']+=1
            return self.np.array(value,copy=True,order='C')
        self._cache_stats['misses']+=1
        return None

    def _commit_previous_tail(self, pixels, encoded, identity, pixel_sha256):
        if not self.reuse_previous_tail:return
        require(pixels.shape==SHAPE[1:] and pixels.dtype==self.np.float32 and pixels.flags.c_contiguous and
            encoded.shape==(1,1041,1024) and encoded.dtype==self.np.float32 and self.np.isfinite(encoded).all(),
            'Exact original last-frame encoder output required')
        raw_pixels=pixels.tobytes()
        require(hashlib.sha256(raw_pixels).hexdigest()==pixel_sha256,'Tail pixels changed before cache commit')
        owned=self.np.array(encoded,copy=True,order='C')
        require(owned.flags.owndata,'One owned tail encoder entry required')
        owned.flags.writeable=False
        self._tail_pixels,self._tail_encoded=raw_pixels,owned
        self._tail_key=(identity,tuple(pixels.shape),str(pixels.dtype),pixel_sha256)
        self._tail_output_sha256=hashlib.sha256(owned.tobytes()).hexdigest()
        self._cache_stats['commits']+=1

    def _drop_previous_tail(self):
        if self._tail_encoded is not None:self._cache_stats['drops']+=1
        self._tail_pixels=self._tail_encoded=self._tail_key=None
        self._tail_output_sha256=None

    def _cache_metrics(self):
        return dict(self._cache_stats,enabled=self.reuse_previous_tail,
            cache_entries=int(self._tail_encoded is not None),
            cache_bytes=(len(self._tail_pixels)+self._tail_encoded.nbytes) if self._tail_encoded is not None else 0,
            policy='Only the immediately preceding successful pair last frame; exact prepared byte identity',
            payload='Owned FP32 original24-layer DINO output BEFORE original finalnorm and pair-role tokens',
            aggregator_head_pose_or_prediction_reuse=False,new_tail_always_computed=True)

    @staticmethod
    def _advance_DINO_group(context, encoded, cache_hit):
        # Cached first-frame output has already traversed all24 original layers.
        # Every new tail traverses this group; an old tail never replaces it.
        return [value if index==0 and cache_hit else context.run([value])[0]
                for index,value in enumerate(encoded)]

    def predict(self, images, *, compute_similarity=False):
        require(threading.get_ident()==self.owner and not self.closed and not self.failed and not self.active,
                'Executor requires its idle original owner thread')
        if type(compute_similarity) is not bool:
            raise TypeError('compute_similarity must be a bool')
        if compute_similarity:
            raise self.protocol.UnsupportedExecutionError('Original native518 executor supports max_loops=0 only')
        self.protocol.validate_images(images)
        require(images.shape==SHAPE,'Exactly two original native518 [2,3,392,518] frames required')
        np,torch,prefix = self.np,self.torch,self.prefix
        self.calls_attempted+=1;self.last_timings={}
        started=time.perf_counter();oldaff,oldenv,oldpath=os.sched_getaffinity(0),dict(os.environ),sys.path[:]
        owners=[];dinos=[];agg=heads=None;result=None
        pixels=normalized=prepared=encoded=input_tokens=aggregated=outputs=owned_pixels=pending_tail=None
        r=dict(index=self.calls_attempted-1,status='running',input_shape=list(SHAPE),
            input_raw_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
            input_frame_raw_sha256=[hashlib.sha256(images[i].tobytes()).hexdigest() for i in range(2)],
            actual_fresh_caller_pixels=True,fixed_reference_pixel_substitution=False,prediction_or_aggregator_cache_used=False,
            previous_tail_cache_enabled=self.reuse_previous_tail,tail_cache_hit=False,
            original_model_only=True,intermediate_reference_substitution=False,cleanup_errors=[])
        self.last_call=r
        self.active=True
        try:
            os.sched_setaffinity(0,{4,5,6,7})
            os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            sys.path[:0]=[str(self.repo),str(self.repo/'third_party/vggt')]
            identity=self._feature_key()
            cached_first=self._lookup_previous_tail(images[0],identity,r['input_frame_raw_sha256'][0])
            hit=cached_first is not None
            r.update(tail_cache_hit=hit,encoder_frame_indices_computed=[1] if hit else [0,1],
                reused_frame_raw_sha256=r['input_frame_raw_sha256'][0] if hit else None,
                requested_core_mask=self.core_mask,feature_computation_key=identity)
            try:
                agg=self.NativeSequentialGroups(self.models,source_root=self.repo,expected_npu_hz=self.expected_npu_hz,core_mask=self.core_mask)
            except BaseException as exc:
                agg=getattr(exc,'native_retained_owner',None)
                raise
            owners.append(('aggregator',agg))
            before=agg.observe();tick=time.perf_counter();times={};sub=time.perf_counter()
            # Exact original Prefix.prepare normalization/token preparation,
            # applied to owned fresh caller pixels. No file decode/resize here.
            owned_pixels=np.array(images,copy=True,order='C')
            with torch.inference_mode():
                pixels=torch.from_numpy(owned_pixels)
                normalized=((pixels.unsqueeze(0)-prefix.mean)/prefix.std).view(2,3,392,518)
                prepared=[prefix.dino.prepare_tokens_with_masks(normalized[i:i+1].contiguous(),masks=None) for i in range(2)]
                require(all(tuple(value.shape)==(1,1041,1024) for value in prepared),'Full original native tokens required')
            times['original_image_token_preparation_seconds']=time.perf_counter()-sub
            encoded=[np.ascontiguousarray(value.numpy()) for value in prepared]
            if hit:encoded[0]=cached_first
            sub=time.perf_counter()
            dino_parts=[]
            for row in self.dino_rows:
                part=time.perf_counter();context=None
                try:
                    try:
                        context=self.OwnedContext(row['file'],row['sha256'],input_shapes=[(1,1041,1024)],
                            output_shapes=[(1,1041,1024)],input_names=['input'],output_names=['output'],
                            core_mask=self.core_mask,flags=0,source_root=self.repo)
                    except BaseException as exc:
                        context=getattr(exc,'owned_context',None)
                        if context is not None:
                            dinos.append(context);owners.append(('dino'+str(len(dinos)-1),context))
                        raise
                    dinos.append(context);owners.append(('dino'+str(len(dinos)-1),context))
                    encoded=self._advance_DINO_group(context,encoded,hit)
                finally:
                    if context is not None:
                        context.close()
                dino_parts.append(time.perf_counter()-part)
                print(json.dumps(dict(stage='fresh_native_DINO_group',call=r['index'],group=len(dino_parts)-1,
                                      seconds=dino_parts[-1])),flush=True)
            times['original_DINO24_execution_with_lifecycle_seconds']=time.perf_counter()-sub
            times['DINO_group_lifecycle_seconds']=dino_parts
            pending_tail=encoded[1]
            sub=time.perf_counter();input_tokens=prefix.finish_dino(encoded);times['original_norm_special_concat_seconds']=time.perf_counter()-sub
            sub=time.perf_counter();aggregated=agg.run(input_tokens,capture=False);times['all48_aggregator_and_consumers_seconds']=time.perf_counter()-sub
            print(json.dumps(dict(stage='fresh_native_all48_aggregator',call=r['index'],
                                  seconds=times['all48_aggregator_and_consumers_seconds'])),flush=True)
            sub=time.perf_counter()
            try:
                heads=self.NativeHeads(source_root=self.repo,core_mask=self.core_mask)
            except BaseException as exc:
                heads=getattr(exc,'native_heads',None)
                raise
            owners.append(('heads',heads))
            outputs=heads.run(aggregated['retained_outputs'],capture=False);heads.close()
            times['camera_and_depth_heads_with_lifecycle_seconds']=time.perf_counter()-sub
            sub=time.perf_counter()
            with torch.inference_mode():
                extrinsics,intrinsics=self.decoder(torch.from_numpy(outputs['pose_enc']),image_size_hw=(392,518))
            outputs['extrinsics']=extrinsics.numpy().copy();outputs['intrinsics']=intrinsics.numpy().copy()
            times['original_pose_codec_seconds']=time.perf_counter()-sub
            elapsed=time.perf_counter()-tick;after=agg.observe()
            sub=time.perf_counter()
            result={canonical:np.array(outputs[raw][0],copy=True,order='C') for canonical,raw in
                    (('pose_enc','pose_enc'),('depth','depth'),('depth_conf','depth_conf'),
                     ('extrinsic','extrinsics'),('intrinsic','intrinsics'))}
            result.update(images=images,image_match_ratio=None)
            result=self.protocol.validate_predictions(result,images,compute_similarity=False)
            require(all(value.flags.owndata for key,value in result.items() if key not in ('images','image_match_ratio')),
                    'Canonical output must own stable CPU bytes')
            r.update(status='completed',complete_image_to_heads_seconds=elapsed,stage_seconds=times,before=before,after=after,
                head_timings=copy.deepcopy(heads.last_timings),aggregator_timings=copy.deepcopy(agg.last_timings),
                final_outputs={key:dict(shape=list(value.shape),dtype=str(value.dtype),tensor_bytes=value.nbytes,
                    raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),finite=bool(np.isfinite(value).all()),owned=True)
                    for key,value in result.items() if key not in ('images','image_match_ratio')},
                canonical_copy_validation_and_hash_seconds=time.perf_counter()-sub)
            self.last_timings=dict(times,complete_image_to_heads_seconds=elapsed)
        except BaseException as exc:
            self._drop_previous_tail()
            self.failed=True;exc.native_executor=self
            r.update(status='failed',error=repr(exc))
            raise
        finally:
            # Constructor failures can return a reachable partial owner before
            # it was appended to the ordinary completed-construction list.
            for name,owner in (('heads',heads),('aggregator',agg)):
                if owner is not None and all(owner is not item[1] for item in owners):
                    owners.append((name,owner))
            resources={};rows=[];held=[]
            for name,owner in reversed(owners):
                try:
                    owner.close()
                except BaseException as exc:
                    r['cleanup_errors'].append(name+': '+repr(exc))
                try:
                    ledger=owner.metrics();resources[name]=ledger
                    rows.extend(ledger['contexts'] if name=='heads' else [ledger])
                    if ledger.get('live_contexts') or ledger.get('live_output_buffers'):
                        held.append((name,owner,ledger))
                except BaseException as exc:
                    r['cleanup_errors'].append(name+' metrics: '+repr(exc));held.append((name,owner,None))
            self._held=held;r['resources']=resources
            totals={key:sum(row[key] for row in rows) for key in COUNTERS}
            for key in COUNTERS:
                self._totals[key]+=totals[key]
            r['SDK_totals']=totals
            r['resources_balanced']=(not held and totals['contexts_created']==totals['contexts_destroyed'] and
                totals['output_buffers_acquired']==totals['output_buffers_released'] and
                all(not row['live_contexts'] and not row['live_output_buffers'] and not row['cleanup_errors'] for row in rows))
            r['complete_expected_execution']=(totals['contexts_created']==totals['contexts_destroyed']==10 and
                totals['runs_completed']==(14 if r['tail_cache_hit'] else 18) and
                totals['output_buffers_acquired']==totals['output_buffers_released']==(19 if r['tail_cache_hit'] else 23))
            try:
                self.verify_unchanged()
                require(hashlib.sha256(images.tobytes()).hexdigest()==r['input_raw_sha256'],'Caller pixels mutated')
                r['source_and_inputs_unchanged']=True
            except BaseException as exc:
                r['source_and_inputs_unchanged']=False;r['cleanup_errors'].append(str(exc))
            os.sched_setaffinity(0,oldaff);os.environ.clear();os.environ.update(oldenv);sys.path[:]=oldpath
            r.update(affinity_restored=os.sched_getaffinity(0)==oldaff,environment_restored=dict(os.environ)==oldenv,
                     sys_path_restored=sys.path==oldpath)
            self.active=False
            if r['status']=='completed' and (r['cleanup_errors'] or not r['resources_balanced'] or not r['complete_expected_execution']):
                self.failed=True;r['status']='failed_cleanup';result=None
            if r['status']=='completed':
                try:
                    require(identity==self._feature_key(),'Encoder model/config key changed during call')
                    self._commit_previous_tail(images[1],pending_tail,identity,r['input_frame_raw_sha256'][1])
                    self.calls_completed+=1
                except BaseException as exc:
                    self.failed=True;r['status']='failed_cleanup';result=None
                    r['cleanup_errors'].append('Cache commit: '+repr(exc))
            if r['status']!='completed':self._drop_previous_tail()
            r['cache_after_call']=self._cache_metrics()
            pixels=normalized=prepared=encoded=input_tokens=aggregated=outputs=owned_pixels=pending_tail=cached_first=None
            self.cleanup_errors.extend(r['cleanup_errors'])
            r['complete_predict_boundary_seconds']=time.perf_counter()-started
        if r['status']!='completed':
            error=RuntimeError('Original native executor cleanup/execution incomplete: '+repr(r['cleanup_errors']))
            error.native_executor=self
            raise error
        return result

    def close(self):
        require(threading.get_ident()==self.owner and not self.active,'Close requires idle original owner')
        self._drop_previous_tail()
        errors=[];held=[]
        for name,owner,prior in self._held:
            try:
                owner.close()
                current=owner.metrics()
                current_rows=current['contexts'] if name=='heads' else [current]
                prior_rows=(prior['contexts'] if name=='heads' else [prior]) if prior else []
                for key in COUNTERS:
                    self._totals[key]+=sum(row[key] for row in current_rows)-sum(row[key] for row in prior_rows)
                if current.get('live_contexts') or current.get('live_output_buffers'):
                    held.append((name,owner,current))
            except BaseException as exc:
                errors.append(name+': '+repr(exc));held.append((name,owner,prior))
        self._held=held
        try:
            self.verify_unchanged()
        except BaseException as exc:
            errors.append(str(exc))
        self.closed=not held
        if self.closed:
            self.prefix=None
        self.cleanup_errors.extend(errors)
        if errors or held:
            self.failed=True
            raise RuntimeError('Native executor owners/pins not clean: '+repr(errors))

    def metrics(self):
        return copy.deepcopy(dict(schema='v110_original_native518_ModelExecutor_metrics',
            SDK_totals=self._totals,calls_attempted=self.calls_attempted,calls_completed=self.calls_completed,
            closed=self.closed,failed=self.failed,active=self.active,
            live_contexts=sum(owner.metrics().get('live_contexts',0) for _,owner,_ in self._held),
            live_output_buffers=sum(owner.metrics().get('live_output_buffers',0) for _,owner,_ in self._held),
            resources_balanced=not self._held and self._totals['contexts_created']==self._totals['contexts_destroyed'] and
                self._totals['output_buffers_acquired']==self._totals['output_buffers_released'],
            maximum_simultaneously_live_SDK_contexts=1,cleanup_errors=self.cleanup_errors,
            no_cached_predictions_or_closed_owner_history=True,previous_tail_cache=self._cache_metrics(),requested_core_mask=self.core_mask,
            system_or_frequency_writes=False,expected_npu_hz=self.expected_npu_hz,
            initialization=getattr(self,'initialization',None),last_call=self.last_call,
            precision_trajectory_realtime_or_8GB_accepted=False))


OriginalNative518Executor = Native518Executor
