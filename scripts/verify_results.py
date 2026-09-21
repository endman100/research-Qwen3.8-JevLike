"""Verify the published historical records without GPU access or new inference."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, io, json, math, re, statistics, zipfile
ROOT=Path(__file__).resolve().parents[1]
def require(ok, text):
    if not ok: raise ValueError(text)
def probability(lt,lf):
    require(all(math.isfinite(v) for v in (lt,lf)), 'Non-finite logprob')
    m=max(lt,lf); return math.exp(lt-m)/(math.exp(lt-m)+math.exp(lf-m))
def verify():
    read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
    cats=[r['category'] for r in read('data/taxonomy.json')]
    require(len(cats)==len(set(cats))==71, 'Taxonomy size')
    def check_dict(d):
        require(set(d)==set(cats) and all(type(v) is bool for v in d.values()),'71 bool keys')
    ids=set(); binary_count=0; dict_count=0; observations={}; inflights=[]
    manifest=read('results/provenance.json')
    require(hashlib.sha256((ROOT/'results/records.zip').read_bytes()).hexdigest()==manifest['records_zip_sha256'],'Archive hash')
    for name,h in manifest['historical_code_sha256'].items():
        require(hashlib.sha256((ROOT/'historical'/name).read_bytes()).hexdigest()==h,'Historical code modified: '+name)
    with zipfile.ZipFile(ROOT/'results/records.zip') as z:
        for entry in manifest['source_bytes_copied_without_changes']:
            require(hashlib.sha256(z.read(entry['archive_path'])).hexdigest()==entry['sha256'],'Evidence bytes changed')
        load=lambda p:json.loads(z.read(p))
        def register(obj):
            require(obj['id'] not in ids,'Duplicate formal response');ids.add(obj['id'])
            require(obj['model']=='qwen3.8-27b','Wrong served model')
        def binary(obj,p,d,cat):
            nonlocal binary_count
            register(obj);binary_count+=1
            require(obj['usage']['completion_tokens']==1,'Not single-token')
            details=obj['usage'].get('completion_tokens_details') or {}
            require(details.get('reasoning_tokens',0)==0,'Thinking tokens')
            ch=obj['choices'][0];msg=ch['message'];require(msg['content'] in ('true','false'),'Not binary')
            require(not msg.get('reasoning') and not msg.get('reasoning_content'),'Thinking text')
            pos=ch['logprobs']['content'];require(len(pos)==1,'Wrong logits positions')
            lp={x['token']:x['logprob'] for x in pos[0]['top_logprobs']}
            pt=probability(lp['true'],lp['false'])
            require(math.isclose(pt,p['p_true'],abs_tol=1e-12),'Probability mismatch')
            require(math.isclose(1-pt,p['p_false'],abs_tol=1e-12),'Probability sum')
            require((pt>=.5)==d[cat],'Threshold mismatch')
        paired=load('01_json_vs_binary/results.json'); require(len(paired['warm_pairs'])==10,'Paired n')
        require(read('data/taxonomy.json')==paired['config']['taxonomy'],'Original taxonomy differs')
        require((ROOT/'data/system.txt').read_text(encoding='utf-8')==paired['config']['system'],'System differs')
        require((ROOT/'data/task.txt').read_text(encoding='utf-8')==paired['config']['task'],'Task differs')
        require(hashlib.sha256((ROOT/'data/taxonomy.json').read_bytes()).hexdigest()==read('data/provenance.json')['taxonomy_sha256'],'Taxonomy source bytes')
        observations['A_json_warm']=[];observations['B4_warm']=[]
        for row in paired['warm_pairs']:
            tag=f"round{row['round']:02d}"
            for m in ['A','B']:
                r=row[m];check_dict(r['output']);dict_count+=1
                observations['A_json_warm' if m=='A' else 'B4_warm'].append(r['seconds'])
            a=load(f'01_json_vs_binary/raw/{tag}_A_response.json');register(a)
            require(a['choices'][0]['finish_reason']=='stop','Incomplete JSON')
            require(json.loads(a['choices'][0]['message']['content'])==row['A']['output'],'A raw mismatch')
            require(a['usage']['completion_tokens']==649,'A token count')
            bs=load(f'01_json_vs_binary/raw/{tag}_B_responses.json');require(set(bs)==set(cats),'B count')
            for cat in cats:binary(bs[cat],row['B']['probabilities'][cat],row['B']['output'],cat)
            require(sum(row['A']['output'][k]!=row['B']['output'][k] for k in cats)==6,'A/B difference')
        branches4=[];prefix71=[];assembly71=[];monitor71=[]
        for group,key in [('02_binary4_segmented','B4_cold_staged'),('03_binary71_6k','B71_6k_cold_staged')]:
            state=load(group+'/segmented_results.json');require(len(state['runs'])==10,'Segment n')
            observations[key]=[]
            for r in state['runs']:
                tag=r['trial'];check_dict(r['output']);dict_count+=1;t=r['timings']
                observations[key].append(t['total_observed_wall_s'])
                require(math.isclose(sum(t[k] for k in ['prepare_s','prefix_wall_s','between_phase_metrics_s','parallel71_wall_s','assemble_s']),t['total_observed_wall_s'],abs_tol=1e-8),'Stage sum')
                require(t['server_parallel_decode_sum_s']==0,'Unexpected decode interval')
                require(r['prefix_metrics']['prefix_cache_hits_total']==0,'Prefix not cold')
                require(r['parallel_metrics']['prefix_cache_hits_total']==333984,'Cache hit total')
                require(r['parallel_metrics']['num_preemptions_total']==0,'Preemption')
                prime=load(f'{group}/raw/{tag}_prefix.json');register(prime)
                require(prime['usage']['prompt_tokens']==4953 and prime['usage']['completion_tokens']==1,'Prime tokens')
                raw=load(f'{group}/raw/{tag}_branches.json');require(set(raw)==set(cats),'Branch coverage')
                for cat in cats:
                    binary(raw[cat],r['probabilities'][cat],r['output'],cat)
                    require(4983<=raw[cat]['usage']['prompt_tokens']<=4991,'Input length')
                for stage,expected in [('prefix_metrics',1),('parallel_metrics',71)]:
                    require(r[stage]['request_count']==expected,'Finished count')
                    for m in ['request_prefill_time_seconds','request_decode_time_seconds','time_to_first_token_seconds']:
                        require(r[stage][m]['count']==expected,'Timing count')
                if group.startswith('02'):branches4.extend(p['http_response_s'] for p in r['probabilities'].values())
                else:
                    prefix71.append(t['prefix_wall_s']);assembly71.append(t['assemble_s']);monitor71.append(t['between_phase_metrics_s'])
                    events=[]
                    for p in r['probabilities'].values():events += [(p['http_start_s'],1),(p['http_done_s'],-1)]
                    active=peak=0
                    for _,change in sorted(events):active+=change;peak=max(peak,active)
                    inflights.append(peak);require(peak==71,'71 HTTP concurrency not reached')
        require(len(ids)==2160 and binary_count==2130 and dict_count==40,'Formal sample count')
    summary=read('results/summary.json')
    for k,v in observations.items():
        require(v==summary['statistics'][k]['samples'],'Sample mismatch')
        require(math.isclose(statistics.median(v),summary['statistics'][k]['median'],abs_tol=1e-10),'Median mismatch')
    theory=read('results/theory.json');ideal=statistics.mean(prefix71)+statistics.mean(branches4)+statistics.mean(assembly71)
    require(math.isclose(ideal,theory['ideal_without_monitor_s'],abs_tol=1e-12),'Theory calculation')
    a,b,c=[read('results/'+m+'_71_bool.json') for m in ['A','B4','B71']]
    require(sum(a[k]!=b[k] for k in cats)==6 and sum(b[k]!=c[k] for k in cats)==3,'Changes')
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__','.venv','runs']):continue
        if p.suffix in ['.json','.md','.txt','.csv','.py','.sh','.patch']:
            text=p.read_text(encoding='utf-8-sig')
            require(not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',text),'Possible secret: '+p.name)
            require(not re.search(r'\b[A-Z]:[\\/](?:Users|ChatOnlineData|skills)|/home/[a-z0-9_-]+/|/mnt/[de]/',text),'Private machine path: '+p.name)
    return {'passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'formal_API_responses':len(ids),'binary_probability_pairs':binary_count,'complete_71_key_dicts':dict_count,'latency_observations':40,'raw_evidence_entries_hashed':len(manifest['source_bytes_copied_without_changes']),'high_concurrency_each_round':inflights,'ideal_s_without_monitor':ideal,'ideal_s_with_monitor':ideal+statistics.mean(monitor71),'A_vs_B4_differences':6,'B4_vs_B71_differences':3,'new_inference_run':False,'scope':'Recorded data integrity, arithmetic and privacy; not accuracy labels, performance rerun, or authenticity of model weight bytes.'}
if __name__=='__main__':
    result=verify();(ROOT/'results/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
