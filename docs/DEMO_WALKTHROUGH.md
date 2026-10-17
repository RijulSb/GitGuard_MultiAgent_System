# GitGuard Interactive Demo Walkthrough

This document showcases the execution trace and behavior of GitGuard across the three primary interaction patterns required by PRD Day 5:
1. **Normal Grounded Question**
2. **Risky Destructive Command with Safety Warning & Safer Alternative**
3. **Out-of-Scope Query / Correct Abstention**

You can run this demo at any time using:
```bash
python demo.py
```
Or interact with the live graphical interface via:
```bash
streamlit run src/ui/app.py
```

---

## 1. Scenario 1: Normal Grounded Question

### Query
> **User**: *"Which Git hook is executed by commands like git commit --amend and git rebase after replacing commits?"*

### Pipeline Execution Trace
```
[Manager] Dispatched query to Query Analyst
[QueryAnalyst] Extracted key concepts: 'hook', 'commit --amend', 'rebase', 'post-rewrite'
[DenseRetriever] Retrieved top 10 chunks from BGE-small vector store
[KeywordRetriever] BM25 retrieved chunks matching 'hook post-rewrite'
[FusionReranker] RRF combined candidates -> Top hit: [ch08-git-hooks-hooks] (score: 0.94)
[DrafterA (Qwen 27B)] Generated grounded draft citing [ch08-git-hooks-hooks]
[DrafterB (GPT-OSS 20B)] Generated concise alternative draft
[Critic] Fact-checked claims against Pro Git Chapter 8 -> 100% verified
[Arbiter] Approved consensus draft
[SafetyGuard] Risk scan: SAFE / MEDIUM (rebase on local commits)
```

### Response
```markdown
The `post-rewrite` hook is run by commands that replace commits, such as `git commit --amend` and `git rebase`. It takes a single argument indicating what command triggered the rewrite, and receives a list of rewrites on STDIN [ch08-git-hooks-hooks].
```
**Citations**: `[ch08-git-hooks-hooks]`  
**Status**: Answered, 100% Faithful Grounding.

---

## 2. Scenario 2: Risky Destructive Command Warning

### Query
> **User**: *"How can I discard all local changes in my working tree and staged index immediately?"*

### Pipeline Execution Trace
```
[Manager] Dispatched query to Query Analyst
[DenseRetriever + BM25] Retrieved reset demystified sections [undoing.001-undoing-things]
[DrafterA] Synthesized command `git reset --hard HEAD`
[SafetyGuard] ⚠️ DETECTED HIGH RISK COMMAND: 'git reset --hard'
  Rule Triggered: RESET_HARD (Risk Level: HIGH)
  Blast Radius: Discards uncommitted changes in working tree and index; irreversible
  Safer Alternative Injected: git stash or git reset --soft HEAD~1
  Display Priority: HIGH risk warning placed FIRST
```

### Response
```markdown
🚨 **SAFETY WARNING — HIGH RISK COMMAND DETECTED**
- **Command**: `git reset --hard`
- **Risk Explanation**: Discards all uncommitted changes in the working tree and index, moving the branch pointer. Overwritten uncommitted files cannot be recovered.
- **Safer Alternative**: Use `git stash` to preserve uncommitted work or `git reset --soft HEAD~1` to keep working tree modifications staged. [undoing.001-undoing-things]

---

To forcefully reset your working tree and staging index to the last commit, you can run:

```bash
git reset --hard HEAD
```

This completely discards your uncommitted changes in the working tree and index [undoing.001-undoing-things].
```
**Status**: Answered with High-Priority Safety Warning & Safe Alternative.

---

## 3. Scenario 3: Out-of-Scope Query & Correct Abstention

### Query
> **User**: *"What is the monthly subscription price of GitHub Copilot Pro?"*

### Pipeline Execution Trace
```
[ScopeGuard] Evaluated query against Pro Git domain scope
[QueryAnalyst] Evaluated semantic relevance: commercial SaaS pricing
[Retriever] Zero relevant chunks found above similarity threshold
[Arbiter] Determined query is out-of-scope for Pro Git book
[Status] Abstained: True
```

### Response
```markdown
I cannot answer this question. The *Pro Git* book covers Git version control concepts, repository administration, and command internals. It does not contain information on commercial third-party pricing.
```
**Status**: Correct Abstention (Zero Hallucination).

---

## 4. Summary of Verification

| Interaction Mode | Grounding Citation | Risk Level | Safety Action | Result |
|:---|:---:|:---:|:---|:---:|
| **Normal** | `[ch08-git-hooks-hooks]` | SAFE | None | Accurate, Grounded |
| **Risky** | `[undoing.001-undoing-things]` | **HIGH** | Injects warning + safer alternative | Safe Guidance |
| **Out-of-Scope** | None | SAFE | Refuses cleanly | No Hallucination |
