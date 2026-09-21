"""71-label experiment: serve, benchmark, or verify historical results.
A compact refactor; original harnesses and raw evidence remain in Git history.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def inputs(cfg):
    cats = [row['category'] for row in cfg['taxonomy']]
    require(len(cats) == len(set(cats)) == 71, 'Expected 71 unique categories')
    taxonomy = '\n'.join(f"{r['category']} | {r['description']}" for r in cfg['taxonomy'])
    common = f"TAXONOMY (all 71 categories):\n{taxonomy}\n\nUSER TASK:\n{cfg['task']}\n\n"
    schema = {'type': 'object', 'properties': {c: {'type': 'boolean'} for c in cats},
              'required': cats, 'additionalProperties': False}
    base = {'model': cfg['served_model'], **cfg['sampling']}
    def payload(suffix):
        return {**base, 'messages': [{'role': 'system', 'content': cfg['system']},
                                    {'role': 'user', 'content': common + suffix}]}
    binary = {}
    for cat in cats:
        binary[cat] = {**payload(cfg['suffix_binary'].format(category=cat)),
                      'max_tokens': 1, 'structured_outputs': {'choice': ['true', 'false']},
                      'allowed_token_ids': list(cfg['binary_token_ids'].values()),
                      'logprobs': True, 'top_logprobs': 2,
                      'logprob_token_ids': list(cfg['binary_token_ids'].values())}
    full = {**payload(cfg['suffix_json']), 'max_tokens': 2048,
            'structured_outputs': {'json': schema, 'disable_any_whitespace': True}}
    return cats, schema, full, binary


def probability(lp_true, lp_false):
    require(all(math.isfinite(x) and x > -9990 for x in (lp_true, lp_false)),
            'Missing or invalid candidate logprobs')
    maximum = max(lp_true, lp_false)
    a, b = math.exp(lp_true - maximum), math.exp(lp_false - maximum)
    return a / (a + b)


def validate_output(output, cats):
    require(isinstance(output, dict) and set(output) == set(cats), 'Incomplete 71-key dict')
    require(all(type(v) is bool for v in output.values()), 'Non-boolean output')


def verify(cfg):
    cats, _, _, binary = inputs(cfg)
    require(digest(binary) == cfg['preflight']['binary_requests_sha256'], 'Prompt changed')
    data = load(ROOT / 'results.json')
    require(len(data['runs']) == 40, 'Expected 40 recorded trials')
    pairs = 0
    for row in data['runs']:
        validate_output(row['output'], cats)
        require(row['total_s'] > 0, 'Invalid latency')
        for cat, p in row.get('probabilities', {}).items():
            value = probability(p['logprob_true'], p['logprob_false'])
            require(abs(value - p['p_true']) < 1e-12, 'Probability mismatch')
            require(row['output'][cat] == (value >= cfg['threshold']), 'Decision mismatch')
            pairs += 1
        if 'timings' in row:
            t = row['timings']
            total = sum(t[k] for k in ('prepare_s', 'prefix_wall_s', 'between_phase_metrics_s',
                                      'parallel71_wall_s', 'assemble_s'))
            require(abs(total - row['total_s']) < 1e-8, 'Stage sum mismatch')
    for method, summary in data['summary'].items():
        samples = [r['total_s'] for r in data['runs'] if r['method'] == method]
        require(len(samples) == 10, 'Expected 10 repetitions per method')
        require(abs(statistics.median(samples) - summary['median_s']) < 1e-10, 'Median mismatch')
        require(abs(statistics.mean(samples) - summary['mean_s']) < 1e-10, 'Mean mismatch')
    require(pairs == 2130, 'Missing probability pairs')
    theory = data['theory']
    ideal = theory['prefix_mean_s'] + theory['branch_mean_proxy_s'] + theory['assembly_mean_s']
    require(abs(ideal - theory['ideal_without_monitor_s']) < 1e-12, 'Theory mismatch')
    print(json.dumps({'verified_trials': 40, 'probability_pairs': pairs,
                      'gpu_rerun': False, 'summary': data['summary']}, indent=2))


def serve(cfg, profile):
    settings = cfg['profiles'][profile]
    executable = shutil.which('vllm')
    require(executable is not None, 'Install the pinned vLLM environment first')
    model = os.environ.get('MODEL_PATH', cfg['model'])
    revision = [] if 'MODEL_PATH' in os.environ else ['--revision', cfg['revision']]
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER='0', VLLM_USE_FLASHINFER_SAMPLER='0')
    command = [executable, 'serve', model, *revision, '--served-model-name', cfg['served_model'],
               '--host', '127.0.0.1', '--port', '8000', '--max-model-len', str(settings['context']),
               '--max-num-seqs', str(settings['max_num_seqs']), '--max-num-batched-tokens',
               str(cfg['max_num_batched_tokens']), '--enable-prefix-caching',
               '--kv-cache-dtype', 'fp8_e4m3', '--gpu-memory-utilization', '0.90',
               '--enable-chunked-prefill', '--reasoning-parser', 'qwen3',
               '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
               '--attention-backend', 'TRITON_ATTN', '--enforce-eager',
               '--safetensors-load-strategy', 'eager']
    subprocess.run(command, env=env, check=True)


class Benchmark:
    def __init__(self, cfg, client, profile):
        from jsonschema import Draft202012Validator
        self.cfg, self.client = cfg, client
        self.settings = cfg['profiles'][profile]
        self.cats, schema, self.full, self.binary = inputs(cfg)
        self.validator = Draft202012Validator(schema)
        self.prefix = []

    def post(self, path, payload):
        response = self.client.post(path, json=payload)
        response.raise_for_status()  # Never automatically retry inference.
        return response.json()

    def preflight(self):
        self.client.get('/health').raise_for_status()
        version = self.client.get('/version'); version.raise_for_status()
        require(version.json()['version'] == self.cfg['vllm_version'], 'vLLM version differs')
        models = self.client.get('/v1/models'); models.raise_for_status()
        model = next(x for x in models.json()['data'] if x['id'] == self.cfg['served_model'])
        require(model['max_model_len'] == self.settings['context'], 'Launch the matching profile')
        for word, token in self.cfg['binary_token_ids'].items():
            obj = self.post('/tokenize', {'model': model['id'], 'prompt': word, 'add_special_tokens': False})
            require(obj['tokens'] == [token], 'Tokenizer differs')
        sequences = []
        for payload in self.binary.values():
            seq = self.post('/tokenize', {'model': model['id'], 'messages': payload['messages'],
                            'chat_template_kwargs': {'enable_thinking': False},
                            'add_generation_prompt': True})['tokens']
            require(len(seq) + 1 <= self.settings['context'], 'Input too long; do not truncate')
            sequences.append(seq)
        for column in zip(*sequences):
            if len(set(column)) != 1:
                break
            self.prefix.append(column[0])
        require(digest(self.prefix) == self.cfg['preflight']['prefix_sha256'], 'Shared prefix differs')

    def metrics(self):
        response = self.client.get('/metrics'); response.raise_for_status()
        out = {}
        for line in response.text.splitlines():
            if not line.startswith('vllm:') or '_bucket{' in line or '_created{' in line:
                continue
            label, value = line.rsplit(' ', 1)
            if '{' in label and f'model_name="{self.cfg["served_model"]}"' not in label:
                continue
            key = label.split('{')[0]
            out[key] = out.get(key, 0) + float(value)
        return out

    def after(self, before, expected):
        for _ in range(30):
            current = self.metrics()
            delta = {k: v - before.get(k, 0) for k, v in current.items()}
            count = delta.get('vllm:request_success_total', 0)
            require(count <= expected, 'Other inference traffic detected')
            if count == expected:
                return current, delta
            time.sleep(0.02)
        raise RuntimeError('Metrics did not catch up')

    def request(self, category, payload, submitted, phase_start):
        started = time.perf_counter()
        obj = self.post('/v1/chat/completions', payload)
        ended = time.perf_counter()
        return category, obj, {'worker_start_s': started - phase_start,
                              'client_queue_s': started - submitted,
                              'http_response_s': ended - started,
                              'http_done_s': ended - phase_start}

    def trial(self, kind, salt):
        before = self.metrics()
        require(not before.get('vllm:num_requests_running', 0)
                and not before.get('vllm:num_requests_waiting', 0), 'Server busy')
        start = time.perf_counter()
        requests = {c: p | {'cache_salt': salt} for c, p in self.binary.items()}
        full = self.full | {'cache_salt': salt}
        ready = time.perf_counter()
        prefix_raw, prefix_delta = None, None
        middle = before
        if kind == 'staged':
            prefix_raw = self.post('/v1/completions', {'model': self.cfg['served_model'],
                'prompt': self.prefix, 'add_special_tokens': False, 'max_tokens': 1,
                'temperature': 0, 'top_p': 1, 'top_k': -1, 'seed': self.cfg['sampling']['seed'],
                'allowed_token_ids': list(self.cfg['binary_token_ids'].values()), 'cache_salt': salt})
            require(prefix_raw['usage']['prompt_tokens'] == len(self.prefix), 'Prefix length differs')
            prefix_end = time.perf_counter()
            middle, prefix_delta = self.after(before, 1)
            require(prefix_delta['vllm:prefix_cache_hits_total'] == 0, 'Prefix must be cold')
        else:
            prefix_end = ready
        phase_start = time.perf_counter()
        raw, per_request = {}, {}
        if kind == 'A':
            raw['A'] = self.post('/v1/chat/completions', full)
        else:
            with ThreadPoolExecutor(max_workers=self.settings['workers']) as pool:
                futures = [pool.submit(self.request, c, p, time.perf_counter(), phase_start)
                           for c, p in requests.items()]
                for future in as_completed(futures):
                    cat, response, timing = future.result()
                    raw[cat], per_request[cat] = response, timing
        phase_end = time.perf_counter()
        probabilities = {}
        if kind == 'A':
            require(raw['A']['choices'][0]['finish_reason'] == 'stop', 'Truncated JSON')
            output = json.loads(raw['A']['choices'][0]['message']['content'])
        else:
            output = {}
            for cat in self.cats:
                obj = raw[cat]; choice = obj['choices'][0]
                require(obj['usage']['completion_tokens'] == 1, 'Expected one token')
                require(choice['message']['content'] in ('true', 'false'), 'Invalid answer')
                positions = choice['logprobs']['content']
                require(len(positions) == 1, 'Expected first-token logprobs')
                lp = {x['token']: x['logprob'] for x in positions[0]['top_logprobs']}
                pt = probability(lp['true'], lp['false'])
                probabilities[cat] = {'p_true': pt, 'p_false': 1 - pt,
                    'logprob_true': lp['true'], 'logprob_false': lp['false']}
                output[cat] = pt >= self.cfg['threshold']
        for obj in raw.values():
            require(obj['model'] == self.cfg['served_model'], 'Wrong model')
            require(obj['usage'].get('completion_tokens_details', {}).get('reasoning_tokens', 0) == 0,
                    'Unexpected reasoning tokens')
        validate_output(output, self.cats); self.validator.validate(output)
        json.dumps(output, ensure_ascii=False, separators=(',', ':'))
        end = time.perf_counter()
        _, branch_delta = self.after(middle, 1 if kind == 'A' else 71)
        timings = {'prepare_s': ready - start, 'prefix_s': prefix_end - ready,
                   'monitor_s': phase_start - prefix_end, 'classification_s': phase_end - phase_start,
                   'assembly_s': end - phase_end, 'total_s': end - start}
        return {'kind': kind, 'salt': salt, 'timings': timings, 'output': output,
                'probabilities': probabilities, 'per_request': per_request, 'raw': raw,
                'prefix_raw': prefix_raw, 'prefix_metrics': prefix_delta, 'branch_metrics': branch_delta}


def run(cfg, args):
    import httpx
    require(args.out is not None, 'Specify a new --out file')
    require(not args.out.exists(), 'Output exists; no overwrite')
    require(args.repeats > 0, 'Repetitions must be positive')
    require(args.protocol != 'paired' or args.profile == '32k4', 'Paired baseline requires 32k4')
    workers = cfg['profiles'][args.profile]['workers']
    limit = 80 if workers == 71 else 8
    with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=240,
                      limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit)) as client:
        bench = Benchmark(cfg, client, args.profile); bench.preflight()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        state = {'profile': args.profile, 'protocol': args.protocol, 'experiment': cfg,
                 'utc_started': datetime.now(timezone.utc).isoformat(), 'warmup': [], 'cold': [], 'runs': []}
        with args.out.open('x', encoding='utf-8') as f:
            json.dump(state, f)
        session = uuid.uuid4().hex
        def record(bucket, kind, salt):
            result = bench.trial(kind, salt); state[bucket].append(result)
            args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({'stage': bucket, 'kind': kind, **result['timings']}), flush=True)
        if args.protocol == 'paired':
            for kind in ('A', 'B'):
                record('warmup', kind, session + '-warm-' + kind)
            for kind in ('B', 'A'):
                record('cold', kind, session + '-measured-' + kind)
            for i in range(args.repeats):
                for kind in (('A', 'B') if i % 2 == 0 else ('B', 'A')):
                    record('runs', kind, session + '-measured-' + kind)
        else:
            record('warmup', 'staged', session + '-warm')
            for i in range(args.repeats):
                record('runs', 'staged', session + f'-{i}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--serve', choices=['32k4', '6k71'])
    modes.add_argument('--verify', action='store_true')
    parser.add_argument('--profile', choices=['32k4', '6k71'], default='6k71')
    parser.add_argument('--protocol', choices=['paired', 'segmented'], default='segmented')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    cfg = load(ROOT / 'experiment.json')
    if args.verify:
        verify(cfg)
    elif args.serve:
        serve(cfg, args.serve)
    else:
        run(cfg, args)


if __name__ == '__main__':
    main()
