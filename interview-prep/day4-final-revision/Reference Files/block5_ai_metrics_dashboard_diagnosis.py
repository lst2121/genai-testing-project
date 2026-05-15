"""
================================================================================
DAY 4 - BLOCK 5: AI METRICS AND DASHBOARD DIAGNOSIS
================================================================================
Goal:
Revise AI/GenAI metrics by diagnosing dashboard failures:
Is it a retrieval issue, generation issue, tool-call issue, schema issue,
guardrail issue, analytics issue, or hallucination?
================================================================================
"""


# =============================================================================
# 1. CORE METRICS
# =============================================================================
"""
Context Precision:
    relevant retrieved chunks / total retrieved chunks
    Low = noisy retrieval.

Context Recall:
    required facts retrieved / total required facts
    Low = missing context.

Faithfulness:
    supported answer claims / total answer claims
    Low = answer has unsupported claims.

Groundedness:
    final answer is backed by context/tool result.
    Low = model is not using evidence.

Answer Relevancy:
    answer addresses the user question.
    Low = off-topic answer.

Hallucination Rate:
    hallucinated answers / total evaluated answers.

Tool Selection Accuracy:
    correct tool selections / total tool-needed cases.

Argument Accuracy:
    correct tool arguments / total tool calls.

Tool Grounding Rate:
    final answers grounded in tool output / total tool answers.

Guardrail Recall:
    unsafe prompts blocked / total unsafe prompts.

Guardrail Precision:
    truly unsafe blocked prompts / all blocked prompts.
"""


# =============================================================================
# 2. DIAGNOSIS PATTERNS
# =============================================================================
"""
Low context recall:
    Retrieval missed required facts.
    Likely issue:
        chunking, embeddings, top_k, filters, missing docs.

Low context precision:
    Retrieval returned irrelevant chunks.
    Likely issue:
        ranking, metadata filters, bad chunks, broad query.

Good context precision/recall but low faithfulness:
    Retrieval is good, generation hallucinated.
    Likely issue:
        prompt, model behavior, weak grounding instruction.

High relevancy but low faithfulness:
    Answer sounds relevant but unsupported.
    Likely hallucination.

Low answer relevancy but high faithfulness:
    Answer is grounded but answers wrong question.
    Likely issue:
        query understanding, prompt, intent classification.

Correct tool but wrong args:
    Tool selection is fine; argument extraction failed.
    Likely issue:
        tool schema, tool description, examples, prompt.

Wrong tool selected:
    Tool selection issue.
    Likely issue:
        ambiguous tool descriptions, weak intent routing.

Schema validation failure:
    Model generated invalid fields or app schema is too strict/inconsistent.

Tool success but wrong final answer:
    Final response synthesis hallucinated.

Tool failure but final answer says success:
    Severe grounding/hallucination issue.

Safe prompt blocked:
    Guardrail false positive / over-blocking.

Unsafe prompt allowed:
    Guardrail false negative / safety bypass.

Dashboard metric wrong:
    Analytics issue:
        missing event, duplicate event, aggregation bug, filter bug, timezone bug.
"""


# =============================================================================
# 3. LANGSMITH VS RAGAS
# =============================================================================
"""
LangSmith:
    Trace/debug individual runs.

Use it to inspect:
    prompt
    retrieved context
    tool call
    arguments
    intermediate steps
    final answer
    latency/tokens

RAGAS:
    Dataset-level RAG evaluation.

Use it to measure:
    faithfulness
    answer relevancy
    context precision
    context recall

Interview line:
    LangSmith tells me why a specific run failed. RAGAS tells me whether quality
    is improving or degrading across a dataset.
"""


# =============================================================================
# 4. CONFUSION MATRIX REFRESHER
# =============================================================================
"""
Positive class example:
    blocked/unsafe

TP:
    unsafe prompt correctly blocked

FP:
    safe prompt incorrectly blocked

FN:
    unsafe prompt allowed

TN:
    safe prompt allowed

Precision:
    TP / (TP + FP)

Recall:
    TP / (TP + FN)

F1:
    2 * precision * recall / (precision + recall)
"""


# =============================================================================
# 5. DASHBOARD SCENARIOS
# =============================================================================
"""
Scenario 1:
    context_precision = 0.30
    context_recall = 0.90
    faithfulness = 0.50

Diagnosis:
    Retriever finds required info but includes too much noise. Generation may be
    confused by irrelevant chunks.


Scenario 2:
    context_precision = 0.90
    context_recall = 0.30
    faithfulness = 0.40

Diagnosis:
    Retrieved chunks are relevant but incomplete. Missing facts cause
    hallucination.


Scenario 3:
    context_precision = 0.90
    context_recall = 0.90
    faithfulness = 0.30

Diagnosis:
    Retrieval is good. Generation is not grounded.


Scenario 4:
    tool_selection_accuracy = 1.0
    argument_accuracy = 0.60

Diagnosis:
    Model chooses right tools but extracts wrong arguments.


Scenario 5:
    tool result says customer not found.
    final answer says customer is Premium.

Diagnosis:
    Tool grounding failure / hallucination.


Scenario 6:
    dashboard total runs = 100
    raw event count = 90

Diagnosis:
    Duplicate counting, wrong source, or aggregation bug.
"""


# =============================================================================
# INTERVIEW ANSWER
# =============================================================================
"""
Question:
    How do you debug an AI quality dashboard?

Answer:

    "I first split the issue by layer. If context recall is low, retrieval is
    missing required facts. If context precision is low, retrieval is noisy. If
    retrieval metrics are good but faithfulness is low, generation is
    hallucinating. If tool selection is correct but argument accuracy is low,
    argument extraction or schema design is the issue. If tool output is correct
    but final answer is wrong, it is a grounding problem. For guardrails, I look
    at false positives and false negatives. For dashboard count mismatches, I
    validate raw events, ingestion, aggregation, filters, and timezone logic."
"""


# =============================================================================
# OFFLINE PRACTICE TASKS
# =============================================================================
"""
Practice:

1. Given metrics, write diagnosis.
2. Given confusion matrix counts, calculate precision/recall/F1.
3. Given tool expected/actual rows, calculate tool and argument accuracy.
4. Given dashboard count mismatch, list possible analytics causes.
5. Explain LangSmith vs RAGAS in one minute.
"""


# =============================================================================
# 6. DETAILED DEFINITIONS WITH INTERVIEW LANGUAGE
# =============================================================================
"""
Hallucination:
    The model produces information that is not supported by retrieved context,
    tool output, database/API result, or known ground truth.

    RAG example:
        Context says refunds take 7 days.
        Answer says refunds take 24 hours.

    Tool example:
        customer_lookup returns "Customer not found".
        Final answer says customer is Premium and active.


Groundedness:
    Whether the final answer is backed by the evidence available to the system.
    For RAG, evidence is retrieved context. For agents, evidence is tool output.

    Good grounded answer:
        Tool result: result=105
        Final answer: "The result is 105."

    Bad groundedness:
        Tool result: result=105
        Final answer: "The result is 75."


Faithfulness:
    Whether claims in the answer can be inferred from the provided context.

    Formula idea:
        faithfulness = supported_claims / total_claims

    Difference from groundedness:
        They are close. Faithfulness is often RAG/context focused. Groundedness
        is broader and can include tool/API/database evidence.


Answer Relevancy:
    Whether the answer addresses the user's question.

    Important:
        A response can be relevant but not faithful.
        Example: user asks customer plan, answer gives a plan, but invented it.


Context Precision:
    Of retrieved chunks, how many were useful?

    Low precision:
        Too much noise.


Context Recall:
    Of required facts, how many were retrieved?

    Low recall:
        Missing important information.
"""


# =============================================================================
# 7. FAILURE CLASSIFICATION TABLE
# =============================================================================
"""
Use this table when reading dashboards.

| Symptom | Likely Layer | Explanation |
|---|---|---|
| Low context recall | Retrieval | Required facts were not retrieved |
| Low context precision | Retrieval | Retrieved chunks contain noise |
| Good retrieval, low faithfulness | Generation | Model ignored or distorted context |
| Good tool result, wrong answer | Final synthesis | Model misreported tool output |
| Correct tool, wrong args | Tool argument generation | Model extracted wrong fields |
| Wrong tool | Tool selection/routing | Intent/tool selection failed |
| Schema validation failed | Schema/tool contract | Missing/extra/wrong typed args |
| Tool timeout/error | Tool execution | Backend/tool failed |
| Safe prompt blocked | Guardrail FP | Over-blocking |
| Unsafe prompt allowed | Guardrail FN | Safety bypass |
| Dashboard count mismatch | Analytics | Event/ingestion/aggregation/filter issue |
| Trace missing tool span | Observability | Instrumentation issue |
"""


# =============================================================================
# 8. DASHBOARD CASE STUDIES
# =============================================================================
"""
Case Study 1: Retrieval Issue
-----------------------------
Dashboard:
    context_precision = 0.82
    context_recall = 0.25
    faithfulness = 0.40
    answer_relevancy = 0.78

Diagnosis:
    The retrieved chunks are mostly relevant, but required facts are missing.
    The model likely hallucinated because it did not get enough information.

Where to inspect:
    - top_k
    - chunk size
    - chunk overlap
    - metadata filters
    - whether source document is indexed
    - embedding model/search query

Fix:
    Improve retriever coverage before changing generation prompt.


Case Study 2: Noisy Context
---------------------------
Dashboard:
    context_precision = 0.30
    context_recall = 0.90
    faithfulness = 0.55

Diagnosis:
    Retriever found required facts, but also brought irrelevant context. The
    model may mix unrelated facts into the answer.

Where to inspect:
    - ranking
    - reranker
    - metadata filters
    - chunk quality
    - query rewriting


Case Study 3: Generation Hallucination
--------------------------------------
Dashboard:
    context_precision = 0.90
    context_recall = 0.90
    faithfulness = 0.30
    answer_relevancy = 0.90

Diagnosis:
    Retrieval is good. The model has enough evidence but is not following it.
    This is generation hallucination.

Where to inspect:
    - system prompt
    - final answer prompt
    - temperature
    - citation requirement
    - instruction to answer only from context
    - LangSmith trace final prompt


Case Study 4: Tool Argument Issue
---------------------------------
Dashboard:
    tool_selection_accuracy = 0.95
    argument_accuracy = 0.55
    schema_validation_failures = high

Diagnosis:
    The model knows which tool to call but struggles to produce valid arguments.

Where to inspect:
    - tool schema descriptions
    - required fields
    - enum values
    - examples in prompt
    - overly permissive or overly strict schema

Fix:
    Improve schema descriptions, add examples, use enums, validate before
    execution, and return clear schema errors.


Case Study 5: Tool Grounding Issue
----------------------------------
Trace:
    tool_name = customer_lookup
    tool_result = {"success": false, "error": "Customer not found"}
    final_answer = "Customer is Premium and active"

Diagnosis:
    Tool execution behaved correctly, but final answer hallucinated after tool
    failure. This is a final response grounding failure.

Fix:
    Update final response prompt:
        "If tool returns error, explain the error. Do not invent missing data."


Case Study 6: Guardrail Quality Issue
-------------------------------------
Eval:
    safe greeting -> blocked
    unsafe tool bypass -> allowed

Diagnosis:
    There are both false positives and false negatives.
    False positive hurts UX.
    False negative is security risk.

Where to inspect:
    - safety classifier prompt
    - policy examples
    - threshold
    - obfuscated prompt handling
    - application-level permission fallback


Case Study 7: Analytics Dashboard Issue
---------------------------------------
Dashboard:
    total agent runs = 105

Raw event store:
    unique run_id count = 100

Diagnosis:
    Possible duplicate events or dashboard counts event rows instead of unique
    runs.

Where to inspect:
    - dedup key
    - event_id/run_id uniqueness
    - retry behavior in event producer
    - aggregation SQL
    - time filter
"""


# =============================================================================
# 9. LANGSMITH TRACE DEBUGGING CHECKLIST
# =============================================================================
"""
When a run fails, inspect in LangSmith:

1. User input
    Was the query clear, unsafe, or ambiguous?

2. System/developer prompt
    Did instructions conflict?

3. Retrieved context
    Was required information present?

4. Tool call
    Was the correct tool selected?

5. Tool arguments
    Were required fields present and correct?

6. Tool output
    Did the tool succeed or fail?

7. Final answer
    Did answer follow context/tool output?

8. Latency/tokens
    Any cost/performance regression?

9. Errors
    Any schema, API, timeout, or auth error?

Interview line:
    "LangSmith helps me locate the failing layer in a single run, while RAGAS
    helps me quantify quality across many rows."
"""


# =============================================================================
# 10. RAGAS DATASET AND PYTEST GATE
# =============================================================================
"""
RAGAS input shape:

    question / user_input
    answer / response
    contexts / retrieved_contexts
    ground_truth / reference

Common RAGAS metrics:

    faithfulness
    answer_relevancy
    context_precision
    context_recall

Pytest gate idea:

    def test_faithfulness_above_threshold(eval_df):
        failures = eval_df[eval_df["faithfulness"] < 0.70]
        assert failures.empty, failures[["question", "faithfulness"]]

Important:
    Do per-row checks, not only average. Averages hide dangerous failures.
"""


# =============================================================================
# 11. HALLUCINATION TYPES
# =============================================================================
"""
1. Factual hallucination
    Invents facts not in evidence.

2. Source hallucination
    Claims a source/citation supports something it does not.

3. Tool hallucination
    Claims a tool succeeded or returned data when it failed.

4. Entity hallucination
    Invents customer/product/ticket/order details.

5. Numerical hallucination
    Wrong calculation, count, percentage, or aggregation.

6. Policy hallucination
    Invents a company rule or SOP.

7. Temporal hallucination
    Uses outdated or wrong date/time.

8. Permission hallucination
    Claims user can perform an action they are not allowed to perform.
"""


# =============================================================================
# 12. RAPID INTERVIEW DRILLS
# =============================================================================
"""
Drill 1:
    context_precision low, context_recall high.

Answer:
    Retrieval has noise. Improve ranking/filtering/reranking.


Drill 2:
    context_precision high, context_recall low.

Answer:
    Retrieval is missing facts. Improve chunking/top_k/indexing/filters.


Drill 3:
    retrieval metrics high, faithfulness low.

Answer:
    Generation hallucination. Inspect prompt/model/final answer instruction.


Drill 4:
    tool selected correctly, schema error high.

Answer:
    Argument generation/schema issue. Improve schema descriptions/examples.


Drill 5:
    tool result correct, final answer wrong.

Answer:
    Tool grounding/final synthesis issue.


Drill 6:
    dashboard value wrong, raw events correct.

Answer:
    Aggregation/API/UI filter issue.


Drill 7:
    dashboard raw events missing.

Answer:
    Event emission or ingestion issue.
"""
