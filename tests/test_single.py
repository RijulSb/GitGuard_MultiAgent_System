import sys
sys.path.insert(0, '.')
from src.graph.graph import run_pipeline

state = run_pipeline('What is a Git repository and how does it store data?', request_id='test-single-q1-parallel')
print('=== Pipeline Result ===')
print('Abstained:', state.get('abstained'))
print('Risk level:', state.get('risk_level'))
print('Traces count:', len(state.get('traces', [])))
print('Total duration ms:', state.get('total_duration_ms'))
print('Verified answer length:', len(state.get('verified_answer', '')))
print('Final response preview:', repr(state.get('final_response', '')[:200]))
for t in state.get('traces', []):
    agent = t.get('agent', '')
    action = t.get('action', '')
    duration = t.get('duration_ms', 0)
    print(f"  - {agent}: {action} ({duration}ms)")
