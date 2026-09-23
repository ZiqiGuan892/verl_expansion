"""Native vLLM server extension with no new generation behavior."""

import inspect
import os

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    """Replica wraps this class in Ray; every native method remains inherited."""

    async def shutdown_engine(self) -> dict:
        """Gracefully stop the vLLM EngineCore child before the Ray actor is killed.

        D2/D3 create-and-cleanup tests intentionally launch a second vLLM
        server on devices borrowed from a donor.  Killing the Ray HTTP actor
        directly can leave vLLM's multiprocessing EngineCore alive for a
        short time (the native actor does not expose a lifecycle hook).  A
        subsequent borrowed launch can then fail while the old child still
        owns device memory or IPC resources.  The method is deliberately
        best-effort and version tolerant: vLLM 0.23 exposes ``shutdown`` on
        ``AsyncLLM``; older compatible releases may only expose
        ``shutdown_background_loop``.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            # AsyncLLM can fail while its EngineCore child is starting, before
            # vLLM assigns the client to ``self.engine``.  Reap only children
            # of this HTTP actor; this cannot affect another Ray actor/job.
            return self._terminate_engine_children("engine_not_initialized")

        errors = []
        for method_name in ("shutdown", "shutdown_background_loop"):
            method = getattr(engine, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
                self.engine = None
                return {"shutdown": True, "method": method_name}
            except Exception as exc:  # cleanup must still try the Ray kill
                errors.append(f"{method_name}: {exc}")

        if errors:
            cleanup = self._terminate_engine_children("shutdown_failed")
            raise RuntimeError("; ".join(errors) + f"; child_cleanup={cleanup}")
        return self._terminate_engine_children("no_shutdown_method")

    @staticmethod
    def _terminate_engine_children(reason: str) -> dict:
        """Terminate vLLM EngineCore descendants owned by this HTTP actor."""
        try:
            import psutil
        except ImportError:
            return {"shutdown": False, "reason": reason, "children": [], "psutil": False}

        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        child_pids = [child.pid for child in children]
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        return {"shutdown": bool(child_pids), "reason": reason, "children": child_pids, "psutil": True}
