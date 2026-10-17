# GitGuard LLM-as-a-Judge Sanity Check (20 Hand-Checked Questions)

| ID | Type | Answerable | Abstained | Correctness | Faithfulness | Citation Acc | Agree? |
|---|---|---|---|---|---|---|---|
| q001 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q002 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q003 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q004 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q005 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q006 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q007 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q008 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q009 | Answerable | True | False | 0.0 | 1.0 | 0.0 | CHECK |
| q010 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q011 | Answerable | True | False | 1.0 | 1.0 | 1.0 | YES |
| q012 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q013 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q014 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q015 | Answerable | True | True | 0.0 | 1.0 | 1.0 | CHECK |
| q061 | Unanswerable | False | True | 1.0 | 1.0 | 1.0 | YES |
| q062 | Unanswerable | False | True | 1.0 | 1.0 | 1.0 | YES |
| q063 | Unanswerable | False | True | 1.0 | 1.0 | 1.0 | YES |
| q064 | Unanswerable | False | True | 1.0 | 1.0 | 1.0 | YES |
| q065 | Unanswerable | False | True | 1.0 | 1.0 | 1.0 | YES |

**Judge-Human Agreement Rate**: 12/20 (60.0%)

## Detailed Question Judgments

### Question [q001]: Which Git command triggers the post-rewrite hook, but git filter-branch does not?
- **Reference Answer**: git commit --amend and git rebase
- **Pipeline Answer**:
  > The `git commit --amend` and `git rebase` commands trigger the post‑rewrite hook, but `git filter-branch` does not [hooks.006-other-client-hooks].
- **Correctness**: 1.0 (Reason: Candidate lists both git commit --amend and git rebase as triggering post-rewrite, matching the reference answer exactly.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (1/1 valid citations)

### Question [q002]: Which Git version introduced the git restore command as an alternative to git reset?
- **Reference Answer**: Git version 2.23.0
- **Pipeline Answer**:
  > `git restore` was introduced in Git version 2.23.0 as an alternative to `git reset` [undoing.004-undoing-things-with-git-restore].
- **Correctness**: 1.0 (Reason: Candidate correctly states the version as 2.23.0, matching the reference answer.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (1/1 valid citations)

### Question [q003]: Which version of Git was used to write the book, and what is the recommendation regarding version compatibility?
- **Reference Answer**: The book was written using Git version 2. Since Git preserves backwards compatibility, any recent version should work fine, though some commands might act slightly differently in ancient versions.
- **Pipeline Answer**:
  > Not found in the book.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q004]: Which environment variable, if set, disables the use of the system-wide configuration file in Git?
- **Reference Answer**: GITCONFIGNOSYSTEM
- **Pipeline Answer**:
  > Not found in the book.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q005]: What is the primary question to consider when a topic branch is ready to be integrated into a mainline branch?
- **Reference Answer**: The question is how to do it and what overall workflow to use to maintain the project.
- **Pipeline Answer**:
  > The primary question to consider when all the work in a topic branch is ready to be integrated into a more mainline branch is how to do it [maintaining.008-integrating-contributed-work]. Additionally, you must consider what overall workflow you want to use to maintain your project [maintaining.008-integrating-contributed-work].
- **Correctness**: 1.0 (Reason: Candidate repeats both core points from reference: how to integrate and which overall workflow to maintain the project.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (2/2 valid citations)

### Question [q006]: What is the purpose of the -M switch when using the git format-patch command?
- **Reference Answer**: The -M switch tells Git to look for renames.
- **Pipeline Answer**:
  > The `-M` switch for `git format-patch` tells Git to look for renames. [contributing.006-public-project-over-email]
- **Correctness**: 1.0 (Reason: Candidate answer exactly matches the reference, correctly stating that -M tells Git to look for renames.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (1/1 valid citations)

### Question [q007]: What is the primary limitation of using the SSH protocol for open source projects?
- **Reference Answer**: SSH does not support anonymous access, requiring users to have SSH access to the machine even for read-only operations.
- **Pipeline Answer**:
  > The SSH protocol cannot provide anonymous read‑only access; users must have SSH credentials on the server, which makes it unsuitable for open‑source projects where people may simply want to clone the repository without prior authentication [protocols.012-the-cons].
- **Correctness**: 1.0 (Reason: Candidate accurately restates the primary limitation: SSH lacks anonymous read‑only access, requiring credentials, matching the reference answer.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (1/1 valid citations)

### Question [q008]: What are the two specific actions performed by the `git checkout` command when switching back to a different branch?
- **Reference Answer**: It moves the HEAD pointer to point to the target branch and reverts the files in the working directory to the snapshot that branch points to.
- **Pipeline Answer**:
  > Not found in the book.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q009]: What command-line option should be passed to `git svn clone` to prevent the inclusion of `git-svn-id` metadata in commit messages?
- **Reference Answer**: --no-metadata
- **Pipeline Answer**:
  > The git‑svn tools let you work with a Subversion repository from Git, though they behave like a limited version of Git and can introduce translation issues that may confuse you and your collaborators【client-svn.016-git-svn-summary】
- **Correctness**: 0.0 (Reason: Candidate answer does not provide the required '--no-metadata' option; it gives unrelated information.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 0.0 (0/0 valid citations)

### Question [q010]: What is the command to delete a tag from a remote server using the more intuitive method described in the text?
- **Reference Answer**: git push origin --delete <tagname>
- **Pipeline Answer**:
  > To delete a tag from a remote server using the more intuitive method, run:
  > 
  > ```bash
  > git push origin --delete <tagname>
  > ```
  > [tagging.008-deleting-tags]
- **Correctness**: 1.0 (Reason: Candidate answer exactly matches the reference command, providing the same intuitive method with correct syntax.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (1/1 valid citations)

### Question [q011]: Why is it necessary to set the environment variable GITSSLNO_VERIFY=true when using the Git Fusion virtual machine image for testing?
- **Reference Answer**: The image includes a certificate for a domain that does not match the virtual machine's IP address, causing Git to reject the HTTPS connection.
- **Pipeline Answer**:
  > - The Git Fusion virtual‑machine image includes a certificate that is issued for a domain name, which does not match the VM’s IP address, so Git will reject the HTTPS connection when cloning or otherwise accessing the repository over HTTPS [client-p4.003-setting-up].  
  > - Setting the environment variable `GITSSLNO_VERIFY=true` tells Git not to verify SSL certificates, allowing the HTTPS connection to succeed for testing purposes [environment.006-networking].
- **Correctness**: 1.0 (Reason: Candidate correctly states the certificate domain/IP mismatch causing Git to reject HTTPS and explains that setting GITSSLNO_VERIFY=true disables verification, matching the reference.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (2/2 valid citations)

### Question [q012]: What is the primary function of Git submodules?
- **Reference Answer**: Submodules allow you to keep a Git repository as a subdirectory of another Git repository, letting you clone another repository into your project while keeping commits separate.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q013]: What happens to remote-tracking branches and configuration settings when a remote is removed using `git remote remove`?
- **Reference Answer**: All remote-tracking branches and configuration settings associated with that remote are also deleted.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q014]: How do you configure Git to keep your local version of a specific file (e.g., database.xml) during a merge instead of creating a conflict?
- **Reference Answer**: Set the attribute `database.xml merge=ours` and define a dummy driver with `git config --global merge.ours.driver true`.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q015]: What are the three specific actions supported by Git credential helpers, and what is the behavior of the 'get' action if the helper does not have stored credentials?
- **Reference Answer**: The three actions are 'get' (request for username/password), 'store' (save credentials), and 'erase' (purge credentials). If the helper does not know anything useful for the 'get' action, it can simply exit with no output.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget.
- **Correctness**: 0.0 (Reason: False refusal: abstained on an answerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q061]: How do I resolve a GitHub Actions billing failure?
- **Reference Answer**: Not found in the book.
- **Pipeline Answer**:
  > Not found in the book.
- **Correctness**: 1.0 (Reason: Correctly abstained on unanswerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q062]: How do I configure Git LFS storage limits on GitHub?
- **Reference Answer**: Not found in the book.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget. Please try a simpler question.
- **Correctness**: 1.0 (Reason: Correctly abstained on unanswerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q063]: What is the GitHub Copilot monthly price?
- **Reference Answer**: Not found in the book.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget. Please try a simpler question.
- **Correctness**: 1.0 (Reason: Correctly abstained on unanswerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q064]: How do I create a GitLab merge request approval rule?
- **Reference Answer**: Not found in the book.
- **Pipeline Answer**:
  > Not found in the book.
- **Correctness**: 1.0 (Reason: Correctly abstained on unanswerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)

### Question [q065]: How do I recover a deleted GitHub repository?
- **Reference Answer**: Not found in the book.
- **Pipeline Answer**:
  > Request timed out or exceeded token budget. Please try a simpler question.
- **Correctness**: 1.0 (Reason: Correctly abstained on unanswerable question.)
- **Faithfulness**: 1.0 (0/0 claims supported)
- **Citation Accuracy**: 1.0 (0/0 valid citations)
