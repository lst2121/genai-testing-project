def calculate_context_precision(relevant_retrieved: int, total_retrieved: int) -> float:
    return relevant_retrieved / total_retrieved


def calculate_context_recall(retrieved_required: int, total_required: int) -> float:
    return retrieved_required / total_required


def calculate_success_rate(events: list[dict]) -> float:
    successful = sum(1 for event in events if event["success"])
    return successful / len(events)


def calculate_average_latency(events: list[dict]) -> float:
    return sum(event["latency_ms"] for event in events) / len(events)


def test_context_precision_formula():
    assert calculate_context_precision(3, 4) == 0.75


def test_context_recall_formula():
    assert calculate_context_recall(2, 3) == 2 / 3


def test_success_rate_from_events():
    events = [
        {"run_id": "run-1", "success": True, "latency_ms": 100},
        {"run_id": "run-2", "success": False, "latency_ms": 200},
        {"run_id": "run-3", "success": True, "latency_ms": 300},
    ]
    assert calculate_success_rate(events) == 2 / 3


def test_average_latency_from_events():
    events = [
        {"run_id": "run-1", "success": True, "latency_ms": 100},
        {"run_id": "run-2", "success": False, "latency_ms": 200},
    ]
    assert calculate_average_latency(events) == 150


# Practice next:
# 1. Add duplicate run_id detection.
# 2. Add missing run_id validation.
# 3. Add guardrail precision/recall calculation.
# 4. Add tool argument accuracy calculation.
