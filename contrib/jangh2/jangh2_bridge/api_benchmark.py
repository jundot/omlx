"""OpenAI SSE measurements with server-reported token counts only."""
from __future__ import annotations
import hashlib
import json
import time
from urllib.request import Request,urlopen


def consume_events(lines, *, started, clock=time.perf_counter, observer=None, deadline=None):
    first=None;last=None;chunks=0;content_chars=0;reasoning_chars=0
    first_content=None;first_reasoning=None;finish_reasons=[]
    usage={};completed=False;digest=hashlib.sha256();pending=[]
    def dispatch(data):
        nonlocal first,last,chunks,content_chars,reasoning_chars,usage,completed,first_content,first_reasoning
        now=clock()
        if data.strip()=='[DONE]':
            completed=True;return
        try: value=json.loads(data)
        except (json.JSONDecodeError,TypeError) as exc:
            raise RuntimeError('Malformed JSON in SSE data') from exc
        if not isinstance(value,dict):raise RuntimeError('SSE event is not a JSON object')
        if value.get('error'):raise RuntimeError(f'Server stream error: {value["error"]}')
        if observer is not None:observer(value)
        if isinstance(value.get('usage'),dict):usage=value['usage']
        output=False
        for choice in value.get('choices',[]):
            reason=choice.get('finish_reason')
            if isinstance(reason,str) and reason not in finish_reasons: finish_reasons.append(reason)
            delta=choice.get('delta') or {}
            for key in ('content','reasoning_content','reasoning'):
                text=delta.get(key)
                if isinstance(text,str) and text:
                    if key=='content':
                        content_chars+=len(text)
                        if first_content is None:first_content=now-started
                    else:
                        reasoning_chars+=len(text)
                        if first_reasoning is None:first_reasoning=now-started
                    digest.update(key.encode()+b'\x00'+text.encode('utf-8'))
                    output=True
        if output:
            chunks+=1
            if first is None:first=now-started
            last=now-started
    for raw in lines:
        if deadline is not None and clock()>deadline:raise TimeoutError('Streaming probe exceeded its wall-time limit')
        text=raw.decode('utf-8') if isinstance(raw,bytes) else raw
        line=text.rstrip('\r\n')
        if not line:
            if pending:dispatch('\n'.join(pending));pending=[]
            if completed:break
        elif line.startswith('data:'):
            pending.append(line[5:].lstrip(' '))
        # Ignore SSE event/id/retry/comment fields; they are not model output.
    if pending and not completed:dispatch('\n'.join(pending))
    elapsed=clock()-started
    if not completed:raise RuntimeError('Streaming connection closed before [DONE]')
    if first is None:raise RuntimeError('Stream completed without any textual output')
    tokens=usage.get('completion_tokens')
    if type(tokens) is not int or tokens<0:tokens=None
    return {
        'completed':True,'first_output_seconds':first,'last_output_seconds':last,
        'first_content_seconds':first_content,'first_reasoning_seconds':first_reasoning,
        'finish_reasons':finish_reasons,'truncated':'length' in finish_reasons,
        'total_seconds':elapsed,'nonempty_output_chunks':chunks,
        'completion_tokens':tokens,'prompt_tokens':usage.get('prompt_tokens'),
        'end_to_end_tokens_per_second':tokens/elapsed if tokens is not None and elapsed>0 else None,
        'content_characters':content_chars,'reasoning_characters':reasoning_chars,
        'output_sha256':digest.hexdigest(),
        'measurement_note':'Rate includes request handling and prompt work, not just decode. Token count comes from server usage, never SSE chunk count.',
    }


def request_measurement(base_url,model,prompt,*,max_tokens=128,api_key=None,timeout=600,include_usage=True,chat_template_kwargs=None):
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'max_tokens':max_tokens,'temperature':0,'stream':True}
    if include_usage:payload['stream_options']={'include_usage':True}
    if chat_template_kwargs is not None:payload['chat_template_kwargs']=dict(chat_template_kwargs)
    headers={'Content-Type':'application/json','Accept':'text/event-stream'}
    if api_key:headers['Authorization']='Bearer '+api_key
    request=Request(base_url.rstrip('/')+'/chat/completions',
                    data=json.dumps(payload).encode('utf-8'),headers=headers,method='POST')
    started=time.perf_counter()
    # API key is never returned, printed or written into benchmark reports.
    with urlopen(request,timeout=timeout) as response:
        return consume_events(response,started=started)
