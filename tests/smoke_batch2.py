import sys
sys.path.insert(0, '.')
print('=== Batch 1+2 Smoke Test (Drafter A fix verification) ===')
print('Question: How do I fix a commit made on the wrong branch?')
print()

from src.agents.batch1_intake import run_batch_1
from src.agents.batch2_drafters import run_batch_2

state = run_batch_1('How do I fix a commit made on the wrong branch?')
cls = state.get('classification')
queries = state.get('search_queries')
dense = len(state.get('dense_chunks', []))
keyword = len(state.get('keyword_chunks', []))
print(f'Batch 1: classification={cls}, queries={queries}')
print(f'  dense_chunks={dense}, keyword_chunks={keyword}')
print()

state = run_batch_2(state)
print('Batch 2 complete.')
ctx_count = len(state.get('context_pack', []))
ctx_score = state.get('context_score')
print(f'  context_pack={ctx_count}, context_score={ctx_score}')
print()

draft_a = state.get('draft_a', '')
draft_b = state.get('draft_b', '')
print('--- Draft A preview (first 300 chars) ---')
print(repr(draft_a[:300]))
print()
print('--- Draft B preview (first 300 chars) ---')
print(repr(draft_b[:300]))
print()
print('Total traces:', len(state.get('traces', [])))
print('Tokens used:', state.get('tokens_used', 0))

if 'Drafter error:' in draft_a or 'LLM call failed' in draft_a:
    print('RESULT: FAIL - Drafter A still erroring')
else:
    print('RESULT: PASS - Drafter A produced a response')
