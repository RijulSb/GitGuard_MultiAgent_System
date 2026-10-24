import time
from src.agents.batch1_intake import manager_intake, query_analyst, dense_retriever, keyword_retriever
from src.agents.batch2_drafters import reranker_agent, drafter_a_agent, drafter_b_agent

state = {"question": "How do I force push my branch to remote?"}
print("1. manager intake...")
state = manager_intake(state)

print("2. query analyst...")
t0 = time.time()
state = query_analyst(state)
print(f"   done in {time.time()-t0:.2f}s")

print("3. retrievers...")
t0 = time.time()
s_dense = dense_retriever(state)
s_kw = keyword_retriever(state)
state["dense_chunks"] = s_dense["dense_chunks"]
state["keyword_chunks"] = s_kw["keyword_chunks"]
print(f"   done in {time.time()-t0:.2f}s")

print("4. reranker...")
state = reranker_agent(state)
print(f"   context count: {len(state['context_pack'])}")

print("5. testing drafter_a...")
t0 = time.time()
res_a = drafter_a_agent(state)
print(f"   drafter_a done in {time.time()-t0:.2f}s: draft len {len(res_a.get('draft_a', ''))}")

print("6. testing drafter_b...")
t0 = time.time()
res_b = drafter_b_agent(state)
print(f"   drafter_b done in {time.time()-t0:.2f}s: draft len {len(res_b.get('draft_b', ''))}")
