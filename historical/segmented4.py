"""Stage-timed 71-label binary benchmark against an already-running vLLM server.
Prefix priming: exact shared prompt token prefix, max_tokens=1 (discard output).
Each measured task uses a fresh cache_salt, followed by 71 unchanged binary
prompts at client concurrency 4. Timings are client wall intervals and server
request-event intervals, NOT CUDA kernel profiling. No server/model changes.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv, hashlib, json, math, statistics, subprocess, time, uuid
import httpx
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parent
MODEL='qwen3.8-27b'
WORKERS=4
REPEATS=10
CONFIG=json.loads((ROOT/'previous_experiment_config.json').read_text(encoding='utf-8'))
TEMPLATES=json.loads((ROOT/'request_templates.json').read_text(encoding='utf-8'))
PREFIX=json.loads((ROOT/'prefix_tokens.json').read_text(encoding='utf-8'))
CATS=list(TEMPLATES)
PREVIOUS=json.loads((ROOT/'previous_B_bool.json').read_text(encoding='utf-8'))
VALIDATOR=Draft202012Validator(CONFIG['schema'])
assert len(CATS)==71
CLIENT=httpx.Client(base_url='http://127.0.0.1:8000',timeout=httpx.Timeout(120,connect=10),limits=httpx.Limits(max_connections=8,max_keepalive_connections=8),trust_env=False)
METRIC_TIMES=['request_queue_time_seconds','request_prefill_time_seconds','request_decode_time_seconds','request_inference_time_seconds','time_to_first_token_seconds','e2e_request_latency_seconds']
( ROOT/'raw' ).mkdir(exist_ok=True)
( ROOT/'metrics' ).mkdir(exist_ok=True)

def save(path:Path,x:object):
 text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)
 for attempt in range(8):
  try:
   path.write_text(text,encoding='utf-8');return
  except PermissionError:
   if attempt==7: raise
   time.sleep(.05*(attempt+1))

def snapshot():
 r=CLIENT.get('/metrics');r.raise_for_status();raw=r.text;d={}
 for line in raw.splitlines():
  if not line.startswith('vllm:') or '_bucket{' in line or '_created{' in line:continue
  label,v=line.rsplit(' ',1);name=label.split('{')[0]
  if '{' in label and 'model_name="'+MODEL+'"' not in label:continue
  try:d[name]=d.get(name,0.)+float(v)
  except ValueError:pass
 return {'parsed':d,'raw':raw}

def delta(before,after):
 a,b=before['parsed'],after['parsed']
 return {k:b[k]-a.get(k,0.) for k in b}

def snap_after(before,expected):
 t=time.perf_counter()
 for _ in range(30):
  s=snapshot();d=delta(before,s)
  n=d.get('vllm:request_success_total',0.)
  if n==expected:return s
  if n>expected:raise RuntimeError('Other inference traffic detected: '+str(n)+' vs '+str(expected))
  time.sleep(.02)
 raise RuntimeError('Metrics did not catch up within '+str(time.perf_counter()-t)+' seconds')

def interpret(d,expected):
 n=d['vllm:request_success_total'];assert n==expected
 out={'request_count':int(n)}
 for metric in METRIC_TIMES:
  k='vllm:'+metric
  count=d[k+'_count'];total=d[k+'_sum']
  assert count==expected,(metric,count,expected)
  out[metric]={'sum_s':total,'mean_s':total/count,'count':int(count)}
 for name in ['prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total','generation_tokens_total','num_preemptions_total']:
  out[name]=d.get('vllm:'+name,0.)
 q=out['prefix_cache_queries_total'];h=out['prefix_cache_hits_total']
 out['prefix_cache_hit_fraction']=h/q if q else 0.
 out['uncached_input_tokens']=q-h
 assert out['num_preemptions_total']==0
 return out

def post(path,p):
 r=CLIENT.post(path,json=p)
 if r.is_error:raise RuntimeError(f'{r.status_code}: {r.text[:3000]}')
 obj=r.json();assert obj['model']==MODEL
 return obj

def one_binary(cat,p,submitted,phase_start):
 started=time.perf_counter();http_start=time.perf_counter()
 obj=post('/v1/chat/completions',p);done=time.perf_counter()
 return {'category':cat,'raw':obj,'client_timing':{'submitted_s':submitted-phase_start,'worker_start_s':started-phase_start,'http_start_s':http_start-phase_start,'http_done_s':done-phase_start,'client_queue_s':started-submitted,'http_response_s':done-http_start}}

def assemble(items):
 found={x['category']:x for x in items};assert set(found)==set(CATS)
 values={};probabilities={}
 for c in CATS:
  obj=found[c]['raw'];ch=obj['choices'][0];token=ch['message']['content']
  assert token in ('true','false') and obj['usage']['completion_tokens']==1
  assert obj['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0)==0
  positions=ch['logprobs']['content'];assert len(positions)==1
  lp={x['token']:x['logprob'] for x in positions[0]['top_logprobs']}
  lt,lf=lp['true'],lp['false'];assert all(math.isfinite(x) and x>-9990 for x in (lt,lf))
  m=max(lt,lf);pt=math.exp(lt-m)/(math.exp(lt-m)+math.exp(lf-m))
  values[c]=pt>=.5
  probabilities[c]={'p_true':pt,'p_false':1-pt,'logprob_true':lt,'logprob_false':lf,'sampled_token':token,'sampled_matches_threshold':(token=='true')==values[c],**found[c]['client_timing']}
 assert len(values)==71 and all(type(x) is bool for x in values.values())
 VALIDATOR.validate(values)
 wire=json.dumps(values,ensure_ascii=False,separators=(',',':'))
 return values,probabilities,wire

def trial(tag,salt):
 before=snapshot()
 for gauge in ['vllm:num_requests_running','vllm:num_requests_waiting']:
  assert before['parsed'].get(gauge,0)==0, 'Server busy'
 t0=time.perf_counter()
 prime={'model':MODEL,'prompt':PREFIX,'add_special_tokens':False,'max_tokens':1,'temperature':0,'top_p':1,'top_k':-1,'seed':20260920,'allowed_token_ids':list(CONFIG['binary_token_ids'].values()),'cache_salt':salt}
 branches={c:p|{'cache_salt':salt} for c,p in TEMPLATES.items()}
 t_ready=time.perf_counter()
 prefix_result=post('/v1/completions',prime)
 assert prefix_result['usage']['prompt_tokens']==len(PREFIX) and prefix_result['usage']['completion_tokens']==1
 t_prefix_done=time.perf_counter()
 middle=snap_after(before,1)
 t_branch_start=time.perf_counter()
 items=[]
 with ThreadPoolExecutor(max_workers=WORKERS) as pool:
  futures=[]
  for c,p in branches.items():
   submitted=time.perf_counter();futures.append(pool.submit(one_binary,c,p,submitted,t_branch_start))
  for f in as_completed(futures):items.append(f.result())
 t_branch_done=time.perf_counter()
 output,probs,wire=assemble(items)
 t_end=time.perf_counter()
 after=snap_after(middle,71)
 prefix_metrics=interpret(delta(before,middle),1)
 branch_metrics=interpret(delta(middle,after),71)
 assert prefix_metrics['prefix_cache_hits_total']==0, 'Prefix was not cache-cold'
 assert prefix_metrics['generation_tokens_total']==1 and branch_metrics['generation_tokens_total']==71
 assert branch_metrics['prefix_cache_hits_total']>0,'No prefix reuse detected'
 timings={'prepare_s':t_ready-t0,'prefix_wall_s':t_prefix_done-t_ready,'between_phase_metrics_s':t_branch_start-t_prefix_done,'parallel71_wall_s':t_branch_done-t_branch_start,'assemble_s':t_end-t_branch_done,'total_observed_wall_s':t_end-t0}
 timings['sum_stages_excluding_monitor_s']=timings['prepare_s']+timings['prefix_wall_s']+timings['parallel71_wall_s']+timings['assemble_s']
 timings['server_prefix_prefill_s']=prefix_metrics['request_prefill_time_seconds']['sum_s']
 timings['server_parallel_prefill_mean_s']=branch_metrics['request_prefill_time_seconds']['mean_s']
 timings['server_parallel_ttft_mean_s']=branch_metrics['time_to_first_token_seconds']['mean_s']
 timings['server_parallel_queue_mean_s']=branch_metrics['request_queue_time_seconds']['mean_s']
 timings['server_parallel_decode_sum_s']=branch_metrics['request_decode_time_seconds']['sum_s']
 result={'trial':tag,'cache_salt':salt,'timings':timings,'stage_boundary_offsets_s':{'start':0,'ready':t_ready-t0,'prefix_done':t_prefix_done-t0,'parallel_start':t_branch_start-t0,'parallel_done':t_branch_done-t0,'end':t_end-t0},'prefix_tokens':len(PREFIX),'prefix_response_id':prefix_result['id'],'prefix_metrics':prefix_metrics,'parallel_metrics':branch_metrics,'output':output,'probabilities':probs,'response_ids':[x['raw']['id'] for x in items],'true_categories':[c for c in CATS if output[c]],'differences_from_previous_B':[c for c in CATS if output[c]!=PREVIOUS[c]],'wire_bytes':len(wire.encode()),'utc_completed':datetime.now(timezone.utc).isoformat()}
 save(ROOT/'raw'/(tag+'_prefix.json'),prefix_result)
 save(ROOT/'raw'/(tag+'_branches.json'),{x['category']:x['raw'] for x in items})
 for name,metric in [('before',before),('after_prefix',middle),('after_branches',after)]:
  save(ROOT/'metrics'/(tag+'_'+name+'.json'),metric['parsed'])
  (ROOT/'metrics'/(tag+'_'+name+'.prom')).write_text(metric['raw'],encoding='utf-8')
 save(ROOT/(tag+'_71_bool.json'),output);save(ROOT/(tag+'_71_probabilities.json'),probs);save(ROOT/(tag+'_result.json'),result)
 print(json.dumps({'trial':tag,'prefix_s':round(timings['prefix_wall_s'],6),'prefix_server_prefill_s':round(timings['server_prefix_prefill_s'],6),'parallel71_s':round(timings['parallel71_wall_s'],6),'assemble_ms':round(timings['assemble_s']*1000,3),'monitor_ms':round(timings['between_phase_metrics_s']*1000,3),'total_s':round(timings['total_observed_wall_s'],6),'branch_server_prefill_mean_ms':round(timings['server_parallel_prefill_mean_s']*1000,3),'decode_sum_s':timings['server_parallel_decode_sum_s'],'cache_hits':branch_metrics['prefix_cache_hits_total'],'true_categories':result['true_categories'],'previous_diff':result['differences_from_previous_B']},ensure_ascii=False),flush=True)
 return result

def stats(xs):return {'n':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),'min':min(xs),'max':max(xs),'stdev':statistics.stdev(xs) if len(xs)>1 else 0.}

def main():
 CLIENT.get('/health').raise_for_status()
 assert not (ROOT/'segmented_results.json').exists(), 'Use new directory for a fresh experiment; do not overwrite results.'
 session=uuid.uuid4().hex
 metadata={'started_utc':datetime.now(timezone.utc).isoformat(),'model':MODEL,'vllm_version':CLIENT.get('/version').json(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader'],text=True).strip(),'workers':WORKERS,'server_max_num_seqs':4,'repetitions':REPEATS,'source_task':CONFIG['task'],'source_system':CONFIG['system'],'shared_prefix_token_count':len(PREFIX),'single_case_only':True,'warmup_excluded':True,'fresh_cache_salt_each_trial':True,'no_server_model_changes':True,'token_ids':CONFIG['binary_token_ids'],'threshold':.5,'timing_definition':'Client wall stages; prefix uses /v1/completions on exact LCP token IDs, with one discarded token because API rejects max_tokens=0. Parallel stage completes all 71 requests incl. client queue, HTTP/JSON receive. Assembly includes both logprobs, sigmoid-normalization, 71-bool validation and JSON serialization. Midpoint metrics collection separately timed and included only in total_observed_wall_s. Prefill/decode server intervals come from histogram count/sum deltas; concurrent request sums are NOT critical-path wall time or CUDA kernel time. No per-request engine profiling was enabled.','request_templates_sha256':hashlib.sha256((ROOT/'request_templates.json').read_bytes()).hexdigest(),'prefix_tokens_sha256':hashlib.sha256((ROOT/'prefix_tokens.json').read_bytes()).hexdigest()}
 save(ROOT/'segmented_config.json',metadata)
 print('Warm-up: complete staged path; excluded.',flush=True)
 warmup=trial('warmup',session+'-warmup')
 runs=[]
 for i in range(1,REPEATS+1):
  runs.append(trial(f'run{i:02d}',session+f'-run{i:02d}'))
  save(ROOT/'segmented_results.json',{'config':metadata,'warmup':warmup,'runs':runs})
 summary={'config':metadata,'stages':{k:stats([r['timings'][k] for r in runs]) for k in runs[0]['timings']},'all_10_outputs_identical':all(r['output']==runs[0]['output'] for r in runs),'previous_B_exact_matches':sum(not r['differences_from_previous_B'] for r in runs),'true_categories':runs[0]['true_categories'],'cache_hits_by_trial':[r['parallel_metrics']['prefix_cache_hits_total'] for r in runs],'branch_cache_hit_fraction_by_trial':[r['parallel_metrics']['prefix_cache_hit_fraction'] for r in runs],'formal_request_count':sum(len(r['response_ids'])+1 for r in runs),'completed_utc':datetime.now(timezone.utc).isoformat()}
 save(ROOT/'segmented_summary.json',summary)
 fields=['trial']+list(runs[0]['timings'])
 with (ROOT/'stage_timings.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in runs:w.writerow({'trial':r['trial'],**r['timings']})
 with (ROOT/'per_category_timings_probabilities.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['trial','category','value']+list(runs[0]['probabilities'][CATS[0]])
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in runs:
   for c in CATS:w.writerow({'trial':r['trial'],'category':c,'value':r['output'][c],**r['probabilities'][c]})
 print('SUMMARY '+json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
 try:main()
 finally:CLIENT.close()
