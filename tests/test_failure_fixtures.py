from evals.fixtures import FIXTURE_REGISTRY, ControlledFailingTraceManager, FixtureModelClient
from src.nebius_client import DeterministicModelClient


def test_repeated_step_and_trace_fixtures_are_controlled():
    assert FIXTURE_REGISTRY["duplicate-model-calls"].expected_error_code == "repeated_call"
    assert FIXTURE_REGISTRY["step-limit"].expected_error_code == "step_limit"
    model = FixtureModelClient(DeterministicModelClient(), "step-limit")
    assert model.fixture_id == "step-limit"
    tracer = ControlledFailingTraceManager()
    with tracer.span("test", "chain", {}):
        pass
    assert tracer.failure_count == 1


def test_failure_fixtures_require_no_latency_sleep():
    assert FIXTURE_REGISTRY["weather-timeout"].latency_ms == 0
    assert FIXTURE_REGISTRY["restaurant-timeout"].latency_ms == 0
