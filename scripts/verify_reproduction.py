#!/usr/bin/env python3
"""Check an independent final-checkpoint evaluation against recorded results."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('--reward-atol',type=float,default=.005);a=p.parse_args()
root=Path(__file__).resolve().parents[1];expected=json.loads((root/'reports/final/evaluation.json').read_text());actual=json.loads(a.evaluation.read_text())
assert actual['num_envs']==expected['num_envs']==280
assert len(actual['per_env'])==len(expected['per_env'])==280
worst=0
for ref,got in zip(expected['per_env'],actual['per_env']):
 assert ref['motion_id']==got['motion_id']
 assert got['completed'] and abs(ref['survived_s']-got['survived_s'])<1e-6,(ref,got)
 diff=abs(ref['mean_reward']-got['mean_reward']);worst=max(worst,diff);assert diff<=a.reward_atol,(ref,got)
print(f'PASS: 280/280 completed, durations match, maximum reward difference={worst:.8f}; tolerance={a.reward_atol}')
