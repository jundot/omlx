#!/usr/bin/env python3
"""Verify SSD PLE construction and a few real rows, without the full model."""
import argparse
import importlib
import logging


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    import mlx.core as mx
    import mlx_vlm.utils as utils
    from jangh2_bridge.checkpoint import inspect_checkpoint
    from jangh2_bridge.loader import ple_runtime_context
    from jangh2_bridge.metadata import ple_layer_indices
    from jangh2_bridge.ple_selftest import run
    run()
    checkpoint=inspect_checkpoint(args.model)
    architecture=importlib.import_module('mlx_vlm.models.qwen4_exp')
    language=importlib.import_module('mlx_vlm.models.qwen4_exp.language')
    print('Qwen4 language module:',language.__file__,flush=True)
    config=utils.load_config(checkpoint.root)
    config.setdefault('text_config',config.pop('llm_config',{}))
    config.setdefault('vision_config',{})
    cfg=architecture.ModelConfig.from_dict(config)
    cfg=utils.update_module_configs(cfg,architecture,config,['text','vision','perceiver','projector','audio'])
    defaults=getattr(utils,'apply_generation_config_defaults',None)
    if callable(defaults): cfg=defaults(cfg,config)
    text=cfg.text_config
    # Match Qwen4ExpDecoderLayer: config IDs are one-based; tensor indices and
    # the n-gram constructor are zero-based. Preserve the configured ordinal.
    tensor_layers=ple_layer_indices(text.ple_layer_ids,getattr(text,'num_hidden_layers',None))
    state=None; opened=[]
    try:
        with ple_runtime_context(language,checkpoint) as state:
            for index,layer_idx in enumerate(tensor_layers):
                # This builds only n-gram hashing and the SSD reader: no expert,
                # attention, vision, embedding-token or output-head weights.
                wrapper=language.Qwen4ExpNGramEmbedding(text,text.ple_embed_dim,layer_idx,index)
                reader=wrapper.ngram_embedding; opened.append(reader)
                rows=getattr(reader,'row_count',None)
                if rows is None: rows=reader.shard_offsets[-1]
                width=getattr(reader,'row_width',getattr(reader,'dims',None))
                shards=int(text.split_ngram_parts)
                first_size=int(rows)//shards+(int(rows)%shards>0)
                ids=sorted(set(i for i in (0,first_size-1,first_size,int(rows)-1) if 0<=i<rows))
                values=reader(mx.array(ids,dtype=mx.int64))
                mx.eval(values)
                if tuple(values.shape)!=(len(ids),width):
                    raise AssertionError(f'Unexpected PLE result shape: {values.shape}')
                if not bool(mx.all(mx.isfinite(values)).item()):
                    raise AssertionError('Non-finite PLE row values')
                print(f'PASS checkpoint PLE config_id={text.ple_layer_ids[index]} -> layers.{layer_idx} '
                      f'(ple_index={index}): {state.kind}, '
                      f'rows={rows}, width={width}, sampled_ids={ids}, dtype={values.dtype}',flush=True)
                mapping_stats=getattr(reader,'mapping_stats',None)
                if callable(mapping_stats):
                    print('PLE file mappings:', mapping_stats(), flush=True)
                del values,wrapper
            state.validate_complete()
        print('PLE preflight passed. Full model was NOT loaded.',flush=True)
    finally:
        if state is not None: state.close()
        for reader in opened:
            close=getattr(reader,'close',None)
            if callable(close): close()

if __name__=='__main__':
    main()
