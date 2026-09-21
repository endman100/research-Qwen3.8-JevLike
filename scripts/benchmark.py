"""Run byte-exact historical harnesses in a fresh, portable output directory.
No service restart, prompt truncation, or automatic inference retry. Run one server.
"""
from pathlib import Path
import argparse, json, shutil, subprocess, sys
import httpx
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['paired','segmented'],default='segmented')
    p.add_argument('--profile',choices=['32k4','6k71'],default='6k71')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();ctx=32768 if a.profile=='32k4' else 6144
    if a.protocol=='paired' and a.profile!='32k4':p.error('Historical paired baseline uses 32k4. Use segmented for 6k71.')
    with httpx.Client(base_url='http://127.0.0.1:8000',trust_env=False,timeout=30) as c:
        c.get('/health').raise_for_status();m=c.get('/v1/models');m.raise_for_status()
        model=next(x for x in m.json()['data'] if x['id']=='qwen3.8-27b')
        if model['max_model_len']!=ctx:raise RuntimeError('Server context differs; launch matching profile first.')
        cfg=json.loads((ROOT/'data/experiment_config.json').read_text(encoding='utf-8'))
        for word,tid in cfg['binary_token_ids'].items():
            r=c.post('/tokenize',json={'model':'qwen3.8-27b','prompt':word,'add_special_tokens':False});r.raise_for_status()
            if r.json()['tokens']!=[tid]:raise RuntimeError('Tokenizer changed; use the recorded model revision.')
        templates=json.loads((ROOT/'data/request_templates.json').read_text(encoding='utf-8'))
        seqs=[]
        for payload in templates.values():
            r=c.post('/tokenize',json={'model':'qwen3.8-27b','messages':payload['messages'],'chat_template_kwargs':{'enable_thinking':False},'add_generation_prompt':True});r.raise_for_status();seq=r.json()['tokens']
            if len(seq)+1>ctx:raise RuntimeError('Context overflow: do not truncate')
            seqs.append(seq)
        prefix=[]
        for col in zip(*seqs):
            if len(set(col))!=1:break
            prefix.append(col[0])
        old=json.loads((ROOT/'data/prefix_tokens.json').read_text())
        if prefix!=old:raise RuntimeError('Shared-prefix token IDs differ from recorded checkpoint.')
        v=c.get('/version');v.raise_for_status()
        if v.json().get('version')!='0.29.0':raise RuntimeError('Historical harness requires vLLM 0.29.0')
    print(json.dumps({'preflight':'passed','context':ctx,'prefix_tokens':len(prefix),'prompt_range':[min(map(len,seqs)),max(map(len,seqs))]}),flush=True)
    if a.preflight_only:return
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ['taxonomy.json','request_templates.json','prefix_tokens.json']:
        shutil.copy2(ROOT/'data'/name,a.out/name)
    shutil.copy2(ROOT/'data/experiment_config.json',a.out/'previous_experiment_config.json')
    shutil.copy2(ROOT/'data/previous_B_bool.json',a.out/'previous_B_bool.json')
    (a.out/'preflight.json').write_text(json.dumps({'binary_token_ids':cfg['binary_token_ids']}),encoding='utf-8')
    source='paired.py' if a.protocol=='paired' else ('segmented4.py' if a.profile=='32k4' else 'segmented71.py')
    shutil.copy2(ROOT/'historical'/source,a.out/'run_historical.py')
    # These scripts locate inputs next to themselves; copying preserves code bytes.
    subprocess.run([sys.executable,'-u',str((a.out/'run_historical.py').resolve())],check=True)
if __name__=='__main__':main()
