#!/usr/bin/env python3
"""Check the local tokenizer, thinking modes and real oMLX hooks; no model weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from jangh2_bridge.chat_policy import ChatPolicy, prompt_reasoning_state, token_reasoning_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--report', default=None)
    args=parser.parse_args()
    root=Path(args.model).expanduser().resolve()
    if not (root/'config.json').is_file(): parser.error('Local model directory has no config.json')
    # Local-only lookup. No downloads or remote model Python execution.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
    if not getattr(tokenizer,'chat_template',None):
        template=root/'chat_template.jinja'
        if not template.is_file(): raise RuntimeError('No local chat template was found')
        tokenizer.chat_template=template.read_text()
    from jangh2_bridge.chat_compat import install
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.scheduler import Scheduler
    policy=ChatPolicy.from_environment()
    report={'schema':1,'kind':'local-tokenizer-and-serving-contract',
            'thinking_default':policy.thinking,'reasoning_effort_default':policy.reasoning_effort,
            'full_model_loaded':False,'cases':[]}
    messages=[{'role':'user','content':'Reply with the word hello.'}]
    tools=[{'type':'function','function':{'name':'bridge_probe_tool','description':'A template-only test; never executed.',
            'parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}}]
    undo=install(root,policy=policy)
    try:
        # The constructor does not load models; template methods use the injected
        # local tokenizer and the existing oMLX role/tool formatting code.
        engine=VLMBatchedEngine(str(root))
        engine._tokenizer=tokenizer
        for enabled in (False,True):
            options={'enable_thinking':enabled,'reasoning_effort':'low'}
            prompt=engine._apply_chat_template(messages,tools=tools,chat_template_kwargs=options)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            text_state=prompt_reasoning_state(prompt)
            token_state=token_reasoning_state(tokenizer,ids)
            if text_state is not enabled or token_state is not enabled:
                raise AssertionError(f'Thinking mode mismatch: requested={enabled}, text={text_state}, tokens={token_state}')
            if 'bridge_probe_tool' not in prompt:
                raise AssertionError('The template discarded the tool definition')
            fake_scheduler=SimpleNamespace(tokenizer=tokenizer,
                model=SimpleNamespace(_vlm_model=SimpleNamespace(_jangh2_bridge_model=True)))
            detected=Scheduler._detect_needs_think_prefix(fake_scheduler,SimpleNamespace(prompt_token_ids=ids))
            if detected is not enabled:
                raise AssertionError('oMLX scheduler detected the wrong reasoning state')
            report['cases'].append({'enable_thinking':enabled,'open_think_detected':detected,
                                    'tool_definition_preserved':True,'prompt_tokens':len(ids)})
            print(f'PASS: thinking={enabled}; template, tokenizer and oMLX scheduler agree; tools retained')
        prompt=engine._apply_chat_template(messages)
        report['effective_default_open_thinking']=prompt_reasoning_state(prompt)
        print('Effective no-override template default:',
              'thinking on' if report['effective_default_open_thinking'] else 'direct answer')
        report['status']='passed'
    finally:
        undo()
    if args.report:
        path=Path(args.report).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('Report:',path)
    print('PASS: local chat compatibility; no model weights loaded and no API calls made.')

if __name__=='__main__':main()
