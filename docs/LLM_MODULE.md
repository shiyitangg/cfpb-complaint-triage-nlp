# LLM module: structured extraction and reliability evaluation

This module was added **after the 2023 test set was opened**. It is not part of the triage
results in the README, it does not change the main model, and it never touches 2023 data.
Its numbers are reported separately.

The question it answers is narrow: when a small hosted LLM extracts a few facts from a
complaint narrative into a fixed JSON schema, how often is the output valid, grounded in the
text, and correct against hand labels, and what does it cost?

Status: plan written before any data was sampled or any API call was made.

## 1. Data

| Set | Source | Size | Use |
|---|---|---|---|
| dev | 2022 validation split, main setting (no near duplicate of train) | 50 | prompt iteration; no hand labels |
| gold | 50 from the 2022 validation split + 50 from 2024 | 100 | hand-labeled once, evaluated once after the prompt is frozen |

- 2023 (the opened test year) is not used for anything in this module.
- 2024 rows come from `scripts/09a_llm_match_2024.py`, which matches a fixed 5% random sample
  of 2024 archive narratives to the official export (selected by a hash of the Complaint ID).
  Matching all of 2024 with step 1b ran out of memory on the laptop, and about 50 items are
  needed. The original matched data and tables are not touched. The 2024 outcome-completeness
  figures are estimated on this sample.
- 2024 rows go through the same cleaning as the main dataset: same label rule, same
  50-character minimum after removing masks, one row per Complaint ID. 2024 rows whose
  normalized text exactly matches any row in the modeling data are dropped.
- Before sampling, the script reports the share of 2024 narratives whose outcome is
  In progress, Untimely response or missing. If that share is large, 2024 is not used and
  the gold set is drawn from 2022 only.
- Sampling is stratified by product within each window, with square-root allocation: a
  product with pool size n gets a share proportional to sqrt(n). Proportional allocation
  would give credit reporting more than half the sample and leave fee disputes almost
  empty.
- Narratives longer than 3,000 characters are excluded from both sets so that labeling stays
  within about two hours. The script reports how much of each pool this removes. Results do
  not cover long narratives.
- dev and gold are disjoint. Random seed 20261025.

### Sampling record (run 2026-10-25)

- 2024 sample (step 09a, 5% share): 815,075 archive narratives dated 2024, 40,825 sampled,
  40,770 matched to the official export (match rate 0.998653, no date mismatches).
- Outcome completeness on the matched sample: resolved 40,732 (99.91%), Untimely response 37,
  missing 1, In progress 0. 2024 is used.
- Pools after cleaning and exact dedup: 2022 validation (main setting) 230,679 rows, 220,152
  within 3,000 characters (4.56% removed); 2024 sample 25,625 rows, 24,049 within 3,000
  characters (6.15% removed).
- Drawn: dev 50 (2022), gold 100 (50 from 2022, 50 from 2024), 14 product names across both
  windows (CFPB renamed several products in 2023, so names differ by window). Manifest
  SHA-256 258e7104f2361fedf287bb92c3e62ac34aace239b2524f08b374a3f98449cef1.
- Dry run on 20 dev items: median prompt 2,934 characters, worst-case cost 0.0129 USD for
  one attempt each.
- Smoke run, extract_v1, first 20 dev items (2026-10-25): 20 calls, all schema-valid with no
  retry; mean 1,785 input and 159 output tokens; 0.258 USD per 1,000 narratives; latency
  median 1.53 s, p95 2.10 s. Evidence for non-null values: 47 exact, 1 loose, 2 with no
  quote. Amounts matched the regex baseline on all 20 items.
- Full dev set, extract_v1, 50 items: all schema-valid, no retry; 0.258 USD per 1,000; latency
  median 1.57 s, p95 2.04 s; evidence for non-null values 111 exact, 1 loose, 3 with no quote
  (exact rate 96.5%); model null rate 32% / 34% / 4% for prior contact / refund request /
  fee dispute; amounts matched the regex baseline on all 50 items. Ledger total 0.0129 USD.
- Dev set, extract_v2 (rules version 2), 50 items: all schema-valid, no retry; 0.283 USD per
  1,000 (longer prompt, mean 2,050 input tokens); latency median 1.53 s, p95 1.99 s; evidence
  for non-null values 105 exact, 1 loose, 1 not verbatim, 1 with no quote (exact rate 97.2%);
  model null rate 36% / 46% / 2%; amounts matched the regex baseline on all 50. Agreement
  with extract_v1: 0.96 / 0.74 / 0.94 (prior contact / refund request / fee dispute).
  Ledger total 0.0271 USD.
- Reading the extract_v2 outputs against rules version 2 (2026-10-26), remaining problems on
  dev: refund request set to false with a sentence that states no wish (dev-011, 014, 024,
  042, 045); prior contact set to false for "never heard of this company" (dev-033) and true
  for a follow-up on an earlier CFPB complaint (dev-039); a dispute with the card issuer about
  a fee not counted as a refund request (dev-010); one evidence quote copied from the prompt
  instead of the narrative (dev-046), which the verbatim check caught. extract_v3 tightens
  the wording for these cases; the rules themselves are unchanged.
- Dev set, extract_v3, 50 items: all schema-valid, no retry; 0.296 USD per 1,000 (mean 2,209
  input tokens); latency median 1.47 s, p95 1.85 s; evidence for non-null values 94 exact,
  2 not verbatim (exact rate 97.9%); model null rate 36% / 66% / 6%; amounts matched the
  regex baseline on all 50. Agreement with extract_v2: 0.94 / 0.78 / 0.92. Ledger total
  0.0419 USD.
- Reading the extract_v3 outputs (2026-10-26): v3 fixed the five refund cases and dev-033, but
  introduced new errors of about the same number: refund request dropped to null for a filed
  claim and a reported unauthorized charge (dev-007, 021) and set to true for a request to
  unwind a car loan (dev-036); prior contact set to false for "I have no CITI accounts"
  (dev-035) and for a consumer who did report the charge to their bank (dev-021); fee dispute
  left null on three items that are clearly about something else (dev-006, 015, 039). The
  dev-039 and dev-046 problems remained in both versions. With 50 items and no way to fix
  sampling, a trade of this size cannot be told apart from run-to-run variation.
- Decision: freeze extract_v2. v3 did not show a net gain, it is longer, and further rounds
  on the same 50 items would mostly fit those items. v2 vs v3 agreement on dev is kept as the
  prompt-sensitivity result.

## 2. Fields

The model sees the cleaned narrative only: no product, issue, company or date.

| Field | Type | Checked against |
|---|---|---|
| `amounts_mentioned` | list of strings, each a dollar amount as written | regex baseline, automatic |
| `prior_contact_with_company` | `{value: true/false/null, evidence: string/null}` | hand labels |
| `requested_refund_or_reimbursement` | same | hand labels |
| `fee_or_charge_dispute` | same | hand labels; also compared with the CFPB Issue field |

`evidence` must be copied verbatim from the narrative. `null` means the narrative does not
say. The model is told that null is the right answer when the text is silent, and the null
rate is reported per field.

### Regex baseline for amounts

CFPB writes amounts as `{$1200.00}`; cleaning drops the braces. The baseline is every match
of a dollar sign followed by a number, parsed to a value rounded to cents. Amounts written
without a dollar sign ("500 dollars") are outside the baseline, so disagreements are read
by hand before being called model errors.

## 3. Annotation rules

Label from the narrative text only. Do not look at any model output before or during
labeling. Values are `true`, `false` or `null`. Every cell must be filled.

Rules version 2, revised on 2026-10-26 after reading the extract_v1 outputs on the 50 dev
items and before any gold item was labeled. Changes from version 1: `false` now needs an
explicit statement for the first two fields; disputes filed to reverse a charge and requests
to release held funds count as refund requests; accounts opened by identity theft are not
fee disputes. The prompt follows the same rules.

**prior_contact_with_company**
- `true`: the consumer says they, or someone acting for them, already contacted a business
  involved in the problem about this problem: called, emailed, wrote, chatted, visited a
  branch, filed a dispute or claim with it. A dispute sent to a credit bureau or to the
  furnisher of the disputed item counts.
- `false`: only when the consumer explicitly says they have not contacted or disputed with
  the business. "I never did business with this company" is not a statement about contact.
- `null`: the narrative does not say either way.
- Contacting only a regulator, lawyer or other third party is not contact with the company.
  A previous CFPB complaint does not count. Citing laws or listing accounts as "disputed" in
  this complaint does not count.

**requested_refund_or_reimbursement**
- `true`: the consumer asks for money to be returned or paid to them: a refund,
  reimbursement, compensation or damages; reversal or credit of a charge or fee already
  applied, including by filing a dispute or claim to get an unauthorized or wrong charge
  reversed; or release of their own money that the company is holding (a frozen balance,
  funds that cannot be withdrawn).
- `false`: the consumer states what they want and none of it is money (for example, removal
  of an item from a credit report, stopping calls, closing an account, changing loan terms).
  There must be a sentence stating the wish.
- `null`: no desired outcome is stated. Describing harm or what went wrong is not a stated
  outcome.
- Saying "I don't owe this debt" or asking to cancel a debt is not a refund request.

**fee_or_charge_dispute**
- `true`: the complaint disputes a specific fee, charge, interest charge or transaction
  amount: overdraft or late fees, unauthorized or duplicate charges, charges for something
  not received, added collection fees, wrong interest, a payment applied wrongly so the
  balance is wrong.
- `false`: the complaint is about something else (report accuracy, harassment, account
  access, customer service, loan terms) and no fee or charge is disputed.
- `null`: only when it truly cannot be told.
- Disputing whether a whole debt is owed is `false` unless a fee or charge on it is disputed.
  An account or loan opened in the consumer's name by identity theft is `false` unless
  specific charges on an account of their own are disputed.

Labels are entered with `scripts/09b_llm_label.py`, which shows one narrative at a time and
never shows model output. Gold outputs are not generated until both labeling passes are done.

Unclear cases are written in the `notes` column of the labeling CSV. They feed the failure
analysis, where label ambiguity is reported as its own category.

**Labeling record.** The first pass on 2026-10-26 used `false` for "not mentioned" and had no
`null` labels, which does not follow the rules above. It was set aside before any model saw
the gold set (kept locally as `data/llm/gold_labels_pass0_binary.csv`) and all 100 items were
labeled again the same day. Second pass counts: prior_contact 88 true / 0 false / 12 null;
refund 18 / 19 / 63; fee 27 / 71 / 2; 17 items have notes.

**Self-agreement.** At least three days after labeling, 20 gold items are relabeled in a new
order without looking at the first labels. Agreement with the first pass is reported per
field.

## 4. Model and calls

- Provider and model are recorded in `reports/llm/frozen_prompt.json`, with the date of the
  calls. The model's own reported name from each response is cached too.
- The API no longer accepts `temperature` (requests that send it are rejected from API
  version 2026-10-21), so outputs cannot be pinned to greedy decoding. Stability is measured
  instead: the frozen prompt is run twice on the gold set and agreement between runs is
  reported. Thinking is turned off; it is not needed for this task and would add cost.
- JSON outputs are constrained to the schema by the API and validated again in code. If an
  output fails validation, the call is retried once with the validation errors appended. If
  it fails again, the item counts as a failure.
- Every response is cached on disk, keyed by model, prompt version, prompt hash, run number
  and text hash. A rerun with the same key reads the cache and costs nothing.
- Budget: a spend ledger in `data/llm/spend.json` adds up the cost of every call across
  runs. Before each call the worst-case cost (estimated input plus the full output limit) is
  checked against the cap. The total cap is 10 USD. A run also stops at `--max-calls`.
- The API key lives only in an environment variable. `.env` is git-ignored and
  `.env.example` lists the variable name only.
- Before the first call, the provider's data-use policy is read and the date is recorded
  here.

## 5. Order of work and freeze rule

1. Smoke run on 20 dev items. Measure tokens, cost per 1,000 narratives and latency.
2. Iterate the prompt on the 50 dev items. Signals: schema validity, evidence pass rate,
   agreement with the regex baseline, and reading the outputs. Each version is a new file
   (`prompts/extract_v1.md`, `extract_v2.md`, ...). Old versions stay in the repo.
3. Freeze: `scripts/10_llm_extract.py --freeze` writes `reports/llm/frozen_prompt.json`
   with the prompt version, its SHA-256 and the model settings. From then on the prompt is
   not edited.
4. Shiyi labels the 100 gold items blind. Labeling can happen before or after the freeze,
   but labels are not compared with any output until step 6.
5. Run the frozen prompt on gold twice (run 1 and run 2). The extract script refuses to run
   on gold if the prompt does not match the frozen hash.
6. `scripts/11_llm_eval.py --set gold` writes `reports/llm_eval_opened.lock` (timestamp and
   frozen-config hash) on its first run and refuses to run again without `--force`. Forced
   runs are appended to a log, as with the test set.

Prompt sensitivity is measured on the dev set: agreement between the frozen version (v2) and
the versions before and after it (v1, v3) on the same 50 items. Older versions are never scored on gold.

## 6. Metrics

On gold, against hand labels, for each boolean field:
- precision, recall and F1 for `true`, treating `false` and `null` as not-true;
- the 3 x 3 confusion matrix of true / false / null;
- null rate of the model and of the labels;
- 95% bootstrap intervals for F1 (2,000 resamples). With 100 items these are wide and are
  reported as such.

On every set:
- schema-valid rate after at most one retry, and the share that needed the retry;
- evidence check: share of non-null values whose evidence is an exact substring of the
  narrative after collapsing whitespace, a looser case-insensitive match, or neither. A
  non-null value with no evidence counts as unsupported;
- amounts: exact set agreement with the regex baseline per item, and micro precision and
  recall of amount values against the baseline;
- `fee_or_charge_dispute` against the CFPB Issue field, as a cross-tab;
- run-to-run agreement per field (gold, run 1 vs run 2);
- tokens, cost per 1,000 narratives, median and 95th-percentile latency.

Failure analysis: at least 20 errors on gold, each given a category in
`reports/llm/gold_errors.csv` (for example: label ambiguity, missed statement, over-reading,
wrong field, evidence not verbatim).

## 7. What is committed

- Code, prompts, this file, result tables under `reports/llm/`, and gold labels as
  Complaint ID plus three values (no text).
- Not committed: narratives, the labeling CSV with text, cached responses and spend ledger
  (all under `data/`).

## 8. Data-use policy check

- Provider: Anthropic (Claude API), model claude-haiku-5-5
- Policy read on: 2026-10-25 (Anthropic Commercial Terms of Service, effective June 17, 2025)
- Notes: read after the 20-item smoke run had started, before the rest of the dev set.

## 9. Results

Gold was run twice with the frozen extract_v2 on 2026-10-31 and scored once (see
`reports/llm_eval_opened.lock`). Tables: `reports/llm/gold_extract_v2_run1_*.csv` and
`gold_labels.csv`; error categories: `gold_errors.csv`. The README section "LLM module" has the
full tables. In short:

- F1 for `true`: prior contact 0.80 (95% CI 0.72 to 0.87), refund 0.71 (0.48 to 0.87), fee
  0.77 (0.62 to 0.88).
- Schema-valid 200 of 200 calls with no retry; 96.3% of non-null evidence quotes are exact
  substrings; amounts match the regex baseline on 99 of 100 items.
- Run-to-run agreement 0.97 / 0.94 / 0.97; self-agreement on 20 relabeled items 1.00 / 0.85 /
  0.90.
- $0.29 per 1,000 narratives, median latency 1.22 s, total module spend $0.10.
- Failure analysis on 30 of 93 errors, taken in a fixed shuffled order by
  `scripts/12_llm_error_review.py`: refund over-reading 15, refund rule misapplied 6, refund
  missed 1, prior contact missed 6, fee rule misapplied 2.

## 10. Limitations

- 100 gold items labeled by one person, so intervals are wide and labels are subjective.
- One model and one prompt family.
- No narratives over 3,000 characters.
- Sampling is not proportional to product volume, so pooled metrics do not describe the
  full complaint stream.
- No production or online validation.
- The evidence check confirms a quote exists, not that it supports the value.
- The extracted fields are not used by the triage model, and no claim is made that they would
  improve it.
- Only 30 of 93 errors were categorized, and none was tagged as label ambiguity even though
  self-agreement on refund was 0.85.
