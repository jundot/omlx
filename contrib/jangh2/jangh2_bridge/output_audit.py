"""Small, bounded live-output diagnostics. Nothing here filters served answers."""
from __future__ import annotations
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def collect_stream(lines, *, max_characters=65536):
    content=[];reasoning=[];finish=None;usage={};pending=[];size=0;done=False
    def dispatch(data):
        nonlocal finish,usage,size,done
        if data.strip()=='[DONE]':done=True;return
        value=json.loads(data)
        if not isinstance(value,dict) or value.get('error'):
            raise RuntimeError('Server reported an error in the output audit stream')
        if isinstance(value.get('usage'),dict):usage=value['usage']
        for choice in value.get('choices',[]):
            if choice.get('index',0)!=0:continue
            delta=choice.get('delta') or {}
            for key,target in (('content',content),('reasoning_content',reasoning),('reasoning',reasoning)):
                text=delta.get(key)
                if isinstance(text,str):
                    size+=len(text)
                    if size>max_characters:raise RuntimeError('Output audit exceeded its bounded response limit')
                    target.append(text)
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    for raw in lines:
        line=(raw.decode('utf-8') if isinstance(raw,bytes) else raw).rstrip('\r\n')
        if line.startswith('data:'):
            pending.append(line[5:].lstrip(' '))
            if sum(map(len,pending))>262144:raise RuntimeError('Oversized SSE event')
        elif not line:
            if pending:dispatch('\n'.join(pending));pending=[]
            if done:break
    if pending and not done:dispatch('\n'.join(pending))
    if not done:raise RuntimeError('Output stream ended before [DONE]')
    return {'content':''.join(content),'reasoning':''.join(reasoning),'finish_reason':finish,'usage':usage}


def inspect_reply(reply, *, expected, thinking=False):
    """Observe the fixed probe's response; do not rewrite or repair it."""
    content=reply.get('content') or '';reasoning=reply.get('reasoning') or ''
    self_talk='we need respond to user' in content.casefold() or 'we need to respond to the user' in content.casefold()
    markup='<think>' in content or '</think>' in content
    truncated=reply.get('finish_reason')=='length'
    if isinstance(expected,dict):
        try:matches=json.dumps(json.loads(content),sort_keys=True)==json.dumps(expected,sort_keys=True)
        except (ValueError,TypeError):matches=False
    else:matches=content.strip()==expected
    return {'passed':matches and not markup and not truncated and (thinking or not reasoning),
            'matches_expected':matches,'self_talk_in_answer':self_talk,
            'reasoning_markup_in_answer':markup,'truncated':truncated,
            'content_characters':len(content),'reasoning_characters':len(reasoning),
            'finish_reason':reply.get('finish_reason')}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward a user's API key to a different URL or host.
        raise HTTPError(req.full_url,code,'Redirect refused by local output audit',headers,fp)


def request_reply(base_url,model,prompt,*,thinking=False,stream=True,api_key=None,max_tokens=256,timeout=300):
    url=urlsplit(base_url)
    if (url.scheme not in ('http','https') or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError('Use an HTTP(S) base URL without credentials, query or fragment')
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'max_tokens':max_tokens,'temperature':0,'stream':stream,
             'chat_template_kwargs':{'enable_thinking':thinking,'reasoning_effort':'low'}}
    if stream:payload['stream_options']={'include_usage':True}
    headers={'Content-Type':'application/json'}
    if api_key:headers['Authorization']='Bearer '+api_key
    request=Request(base_url.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers,method='POST')
    with build_opener(_NoRedirect()).open(request,timeout=timeout) as response:
        if stream:return collect_stream(response)
        data=response.read(1048577)
        if len(data)>1048576:raise RuntimeError('Oversized JSON output audit response')
        obj=json.loads(data)
        if obj.get('error'):raise RuntimeError('Server reported an output audit error')
        choices=obj.get('choices') or []
        if not choices:raise RuntimeError('No response choices')
        choice=choices[0];message=choice.get('message') or {}
        return {'content':message.get('content') or '',
                'reasoning':message.get('reasoning_content') or message.get('reasoning') or '',
                'finish_reason':choice.get('finish_reason'),'usage':obj.get('usage',{})}
