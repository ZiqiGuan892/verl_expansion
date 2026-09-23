"""Native vLLM server extension with no new generation behavior."""

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    """Replica wraps this class in Ray; every native method remains inherited."""
