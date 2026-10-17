# GitGuard Failure Review & Root Cause Analysis (PRD Day 4 Task 8)

This document analyzes the 15 lowest-performing questions from the evaluation run to identify system bottlenecks and inform future improvements.

## Root Cause Breakdown (15 Worst Cases)

| Root Cause | Count | % of Failures | Remediation Strategy |
|---|---|---|---|
| Over-refusal / False Abstention | 14 | 93.3% | Loosen reranker threshold from 0.35 to 0.30 for how-to queries. |
| Retrieval Miss | 0 | 0.0% | Expand dense_top_k to 25 and add query expansion synonyms. |
| Bad Draft / Synthesis Error | 0 | 0.0% | Provide clearer few-shot formatting in Drafter A prompt. |
| Hallucinated Citation / Ungrounded | 1 | 6.7% | Stricter Arbiter rejection when cited section does not match retrieved chunk IDs. |
| Unanswerable Hallucination | 0 | 0.0% | Query Analyst out-of-scope regex filter update for non-Git tooling. |

## Detailed Failure Case Studies

### Case 1: [q003] - Diagnosed: over-refusal / false abstention
- **Question**: Which version of Git was used to write the book, and what is the recommendation regarding version compatibility?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: The book was written using Git version 2. Since Git preserves backwards compatibility, any recent version should work fine, though some commands might act slightly differently in ancient versions.
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 2: [q004] - Diagnosed: over-refusal / false abstention
- **Question**: Which environment variable, if set, disables the use of the system-wide configuration file in Git?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: GITCONFIGNOSYSTEM
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 3: [q008] - Diagnosed: over-refusal / false abstention
- **Question**: What are the two specific actions performed by the `git checkout` command when switching back to a different branch?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: It moves the HEAD pointer to point to the target branch and reverts the files in the working directory to the snapshot that branch points to.
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 4: [q009] - Diagnosed: hallucinated citation / ungrounded
- **Question**: What command-line option should be passed to `git svn clone` to prevent the inclusion of `git-svn-id` metadata in commit messages?
- **Answerable**: True | **Abstained**: False
- **Reference Answer**: --no-metadata
- **System Answer**:
  > The git‑svn tools let you work with a Subversion repository from Git, though they behave like a limited version of Git and can introduce translation issues that may confuse you and your collaborators【client-svn.016-git-svn-summary】
- **Judge Correctness**: 0.0 (Reason: Candidate answer does not provide the required '--no-metadata' option; it gives unrelated information.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 0.0

### Case 5: [q012] - Diagnosed: over-refusal / false abstention
- **Question**: What is the primary function of Git submodules?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: Submodules allow you to keep a Git repository as a subdirectory of another Git repository, letting you clone another repository into your project while keeping commits separate.
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 6: [q014] - Diagnosed: over-refusal / false abstention
- **Question**: How do you configure Git to keep your local version of a specific file (e.g., database.xml) during a merge instead of creating a conflict?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: Set the attribute `database.xml merge=ours` and define a dummy driver with `git config --global merge.ours.driver true`.
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 7: [q015] - Diagnosed: over-refusal / false abstention
- **Question**: What are the three specific actions supported by Git credential helpers, and what is the behavior of the 'get' action if the helper does not have stored credentials?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: The three actions are 'get' (request for username/password), 'store' (save credentials), and 'erase' (purge credentials). If the helper does not know anything useful for the 'get' action, it can simply exit with no output.
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 8: [q018] - Diagnosed: over-refusal / false abstention
- **Question**: What specific metadata and security features are stored in annotated tags but not in lightweight tags?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: Annotated tags store the tagger name, email, and date, a tagging message, and can be signed and verified with GPG.
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 9: [q020] - Diagnosed: over-refusal / false abstention
- **Question**: Which environment variable must be set in the Apache configuration to allow Git to serve repositories to unauthenticated clients without requiring the git-daemon-export-ok file?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: GITHTTPEXPORT_ALL
- **System Answer**:
  > Not found in the book.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 10: [q022] - Diagnosed: over-refusal / false abstention
- **Question**: What configuration setting allows you to sign tags without specifying the GPG key ID every time?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: user.signingkey
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 11: [q024] - Diagnosed: over-refusal / false abstention
- **Question**: What command can be used to retrieve the SHA-1s of the submodule commits recorded in both branches when a submodule merge conflict occurs?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: git diff
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 12: [q025] - Diagnosed: over-refusal / false abstention
- **Question**: What command is used to stop a temporary GitWeb instance started with the webrick handler?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: git instaweb --httpd=webrick --stop
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 13: [q027] - Diagnosed: over-refusal / false abstention
- **Question**: What programming languages are used for the sample hook scripts distributed with Git?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: Perl or Bash
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 14: [q028] - Diagnosed: over-refusal / false abstention
- **Question**: What does the -s option in the git svn clone command imply regarding the repository structure?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: It implies the standard layout, which is equivalent to specifying -T trunk -b branches -t tags.
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0

### Case 15: [q029] - Diagnosed: over-refusal / false abstention
- **Question**: What command should be used to verify that a Git bundle file is valid and contains the necessary ancestors to be imported into a repository?
- **Answerable**: True | **Abstained**: True
- **Reference Answer**: git bundle verify
- **System Answer**:
  > Request timed out or exceeded token budget.
- **Judge Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 | **Citation Accuracy**: 1.0
