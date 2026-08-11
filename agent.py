"""
title: Agent
description: One model thinks and calls tools across multiple rounds; a second model inherits that exact context and finishes the response.
version: 0.1.0
"""

import os
import html
import json
import asyncio
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable, AsyncGenerator


def _get_model_options(__user__=None) -> list:
    """Target for THINKING_MODEL/FINAL_MODEL's 'select' input, if your Open
    WebUI version supports json_schema_extra select inputs on Pipe valves
    (if it doesn't, these fields just fall back to a plain text box -- type
    the model name manually, same as before). Runs synchronously and
    briefly blocks; only fires when the valves config page is opened, not
    per chat request. Reads OLLAMA_BASE_URL from an env var (mirrored there
    by pipe() on every call) since a classmethod has no access to an
    instance's self.valves.
    """
    base_url = os.environ.get("AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return [{"label": n, "value": n} for n in names]
    except Exception:
        return []


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
        THINKING_MODEL: str = Field(
            default="Qwen3-5-4B-heretic-i1-IQ4-XS:latest",
            description="Model that reasons and calls tools.",
            json_schema_extra={
                "input": {"type": "select", "options": "get_model_options"}
            },
        )
        FINAL_MODEL: str = Field(
            default="gemma-4-E4B-it-qat-q4-0-heretic-Q4-0-PURE:latest",
            description="Model that continues from the thinking model's context and produces the response.",
            json_schema_extra={
                "input": {"type": "select", "options": "get_model_options"}
            },
        )
        THINKING_SYSTEM_PROMPT: str = Field(
            default="",
            description=(
                "System prompt for the thinking/tool-calling model. Leave blank "
                "to use whatever system message (if any) is already in the "
                "conversation, unmodified."
            ),
        )
        FINAL_SYSTEM_PROMPT: str = Field(
            default="",
            description=(
                "System prompt for the final response model. Leave blank to "
                "use whatever system message (if any) is already in the "
                "conversation, unmodified."
            ),
        )
        NUM_CTX: int = Field(default=131072)
        MAX_ROUNDS: int = Field(
            default=5,
            description="Max tool-call rounds the thinking model gets before we cut over to the final model.",
        )
        ABORT_THINKING_ON_CONTENT: bool = Field(
            default=True,
            description=(
                "If the thinking model starts emitting plain content (not a tool "
                "call) and no tool_calls have appeared yet in that message, close "
                "the connection immediately instead of letting it finish generating. "
                "The content is discarded either way (it's never shown to the user "
                "or passed to the final model) — this just saves compute on output "
                "that's going to be thrown away. Disable if your thinking model "
                "sometimes emits commentary content BEFORE a tool call in the same "
                "turn — aborting would cut that call off."
            ),
        )
        PARALLEL_TOOL_CALLS: bool = Field(
            default=True,
            description=(
                "If a round contains multiple tool calls, run them concurrently "
                "with asyncio.gather instead of one at a time. Emitted tool "
                "blocks and appended tool-result messages still appear in the "
                "same order the model requested them, regardless of which "
                "finishes first. Disable if your tools aren't safe to run "
                "concurrently (e.g. they share mutable state or hit a "
                "rate-limited API that can't take concurrent requests)."
            ),
        )
        REPORT_FULL_TURN_STATS: bool = Field(
            default=True,
            description=(
                "If True, the generation stats shown to the client (tokens/sec, "
                "durations, etc.) are summed across every model call in the turn "
                "-- every thinking round plus the final model -- so they reflect "
                "the true total compute/time spent. If False, only the final "
                "model's own stats are reported, i.e. just the numbers for the "
                "text actually shown as the answer."
            ),
        )

        @classmethod
        def get_model_options(cls, __user__=None):
            return _get_model_options(__user__)

    def __init__(self):
        self.valves = self.Valves()
        self.type = "pipe"
        self.id = "two_model_relay"
        self.name = "Two-Model Relay"
        # Scratch slot used to hand the fully-accumulated message back out
        # of _stream_and_forward, since async generators can't "return" a
        # value alongside their yields.
        self._last_message = {}

    def pipes(self):
        return [{"id": "two-model-relay", "name": "Agent"}]

    def _clean(self, messages):
        cleaned = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role and content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _with_system_prompt(self, working_messages, system_prompt):
        """Returns a copy of working_messages with `system_prompt` as the
        leading system message, replacing any existing system message.

        Does NOT mutate working_messages — each model call gets its own
        view of the system message without polluting the shared context
        that the other model reads. If system_prompt is empty/blank, the
        original list is returned unchanged (whatever system message, if
        any, came in from the caller is left as-is).
        """
        if not system_prompt:
            return working_messages
        filtered = [m for m in working_messages if m.get("role") != "system"]
        return [{"role": "system", "content": system_prompt}] + filtered

    def _chunk(
        self,
        content: str = "",
        finish_reason: Optional[str] = None,
        usage: Optional[dict] = None,
    ) -> dict:
        delta = {"content": content} if content else {}
        payload = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
        if usage:
            payload["usage"] = usage
        return payload

    def _tool_call_block(self, call_id: str, name: str, arguments: dict, result) -> str:
        args_attr = html.escape(json.dumps(arguments, ensure_ascii=False))
        result_json = (
            html.escape(json.dumps(result, ensure_ascii=False))
            if not isinstance(result, str)
            else html.escape(result)
        )
        return (
            f'<details type="tool_calls" done="true" id="{html.escape(call_id)}" '
            f'name="{html.escape(name)}" arguments="{args_attr}">\n'
            f"<summary>Tool Executed</summary>\n"
            f"{result_json}\n"
            f"</details>\n\n"
        )

    async def _emit_status(
        self, __event_emitter__, description: str, done: bool = False
    ):
        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def _run_tool_call(self, call, tool_funcs, call_index: int):
        """Runs a single tool call and returns (call_id, fname, fargs, result).

        Exceptions are caught and turned into a "Tool error: ..." string
        result, same as the previous sequential behavior, so a single
        failing call doesn't blow up the whole gather().
        """
        fname = call["function"]["name"]
        fargs = call["function"].get("arguments", {})
        call_id = call.get("id") or f"call_{fname}_{call_index}"

        if fname in tool_funcs:
            try:
                func = tool_funcs[fname]
                result = await func(**fargs) if _is_async(func) else func(**fargs)
            except Exception as e:
                result = f"Tool error: {e}"
        else:
            result = f"Tool '{fname}' not available"

        return call_id, fname, fargs, result

    # Stats that are genuinely cumulative across multiple Ollama calls:
    # eval_count/eval_duration are tokens generated and time spent
    # generating on THAT call, so summing them across rounds gives a true
    # "total tokens generated / total time generating this turn". Same for
    # total_duration and load_duration (real wall-clock time spent, model
    # load cost if the model wasn't already resident).
    _SUM_FIELDS = {"total_duration", "load_duration", "eval_count", "eval_duration"}

    # Stats that are NOT cumulative: prompt_eval_count/prompt_eval_duration
    # describe the size/time of the *entire prompt* sent on that call. Since
    # working_messages grows every round, each round's prompt re-includes
    # everything from prior rounds -- summing these across rounds would
    # double- (or triple-, quadruple-...) count the same early messages.
    # The max across calls approximates the true "largest single context
    # ingested this turn" instead.
    _MAX_FIELDS = {"prompt_eval_count", "prompt_eval_duration"}

    def _merge_usage(self, into: dict, new: dict):
        """Merges Ollama's /api/chat done-stats from `new` into `into`.

        Fields in _SUM_FIELDS are summed (they represent genuinely
        additional work/time on each call). Fields in _MAX_FIELDS take the
        max instead of summing, since they describe the size of the whole
        prompt on that call rather than incremental work -- summing them
        would badly overstate input token counts once more than one round
        has happened. Any other numeric field not in either set is left
        alone (unrecognized fields aren't meaningful to combine either
        way, so they're dropped rather than silently mis-aggregated).
        """
        if not new:
            return
        for k, v in new.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if k in self._SUM_FIELDS:
                into[k] = into.get(k, 0) + v
            elif k in self._MAX_FIELDS:
                into[k] = max(into.get(k, 0), v)

    def _finalize_usage(self, usage: dict) -> dict:
        """Expands the raw accumulated Ollama done-stats into the full
        display shape Open WebUI's own native Ollama integration produces
        (see convert_response_ollama_to_openai in
        backend/open_webui/utils/response.py). That function is what
        computes response_token/s, prompt_token/s, approximate_total, and
        the OpenAI-style prompt_tokens/completion_tokens/total_tokens
        aliases for a raw Ollama call -- but it only runs for native Ollama
        model requests, not for pipe-supplied usage, so without this a
        pipe's stats popup would be missing those fields entirely (only
        the raw total_duration/prompt_eval_count/eval_count/... fields
        would show, as seen in testing).

        Formulas are copied as-is from Open WebUI's implementation:
        tokens/sec is eval_count divided by eval_duration converted from
        nanoseconds to seconds (written as `/ (dur / 10_000_000) * 100`,
        which is algebraically the same thing), and approximate_total
        floors total_duration to whole seconds and formats it "HhMmSs".
        """
        eval_count = usage.get("eval_count", 0) or 0
        eval_duration = usage.get("eval_duration", 0) or 0
        prompt_eval_count = usage.get("prompt_eval_count", 0) or 0
        prompt_eval_duration = usage.get("prompt_eval_duration", 0) or 0
        total_duration = usage.get("total_duration", 0) or 0

        response_tok_s = (
            round((eval_count / (eval_duration / 10_000_000)) * 100, 2)
            if eval_duration > 0
            else "N/A"
        )
        prompt_tok_s = (
            round((prompt_eval_count / (prompt_eval_duration / 10_000_000)) * 100, 2)
            if prompt_eval_duration > 0
            else "N/A"
        )
        total_s = total_duration // 1_000_000_000
        approximate_total = (
            f"{total_s // 3600}h{(total_s % 3600) // 60}m{total_s % 60}s"
        )

        return {
            "response_token/s": response_tok_s,
            "prompt_token/s": prompt_tok_s,
            "total_duration": total_duration,
            "load_duration": usage.get("load_duration", 0) or 0,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "approximate_total": approximate_total,
            # OpenAI-style aliases (Chat Completions naming), same fields
            # a native Ollama call's usage block carries alongside the
            # raw ones above.
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }

    async def _stream_ollama(self, client, model, messages, tools=None):
        """Yields raw NDJSON chunk dicts from Ollama's streaming /api/chat."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": self.valves.NUM_CTX},
        }
        if tools:
            payload["tools"] = tools

        async with client.stream(
            "POST", f"{self.valves.OLLAMA_BASE_URL}/api/chat", json=payload
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise RuntimeError(
                    f"Ollama /api/chat error {resp.status_code} (model={model}): {body}"
                )
            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)

    async def _stream_and_forward(
        self,
        client,
        model,
        working_messages,
        tools=None,
        suppress_content: bool = False,
        abort_on_bare_content: bool = False,
    ):
        """Streams a model's output, forwarding tokens live as they arrive,
        and stashes the fully-accumulated message (content, thinking,
        tool_calls, stats) on self._last_message once done.

        Wraps streamed reasoning tokens in <think>...</think> so the UI
        renders them the same way the old buffered version did. Thinking
        is ALWAYS streamed to the user, regardless of suppress_content —
        the two are tracked independently.

        suppress_content: if True, plain `content` tokens (as opposed to
        `thinking`) are accumulated internally (so abort_on_bare_content
        and downstream logic still work) but never streamed to the user
        and never shown in any form — not even in a collapsible box. Use
        this for the thinking model: its prose answer isn't the real
        response and shouldn't be surfaced at all.

        abort_on_bare_content: if True, and content starts arriving with
        no tool_calls seen yet in this message, close the connection
        immediately rather than let the model finish. This tells Ollama
        to stop generating (it detects the client disconnect), saving
        compute on output that's going to be discarded. Only safe when
        the model doesn't interleave commentary content before a tool
        call within the same turn.

        Ollama's final NDJSON line for a call (the one with "done": true)
        carries generation stats -- total_duration, load_duration,
        prompt_eval_count, prompt_eval_duration, eval_count, eval_duration,
        etc. Those are captured here (everything on that line except
        "message") and returned via self._last_message["stats"] so the
        caller can forward them to the client as a standard "usage" block.
        """
        acc_thinking = ""
        acc_content = ""
        acc_tool_calls = None
        thinking_open = False
        aborted = False
        stats = {}

        gen = self._stream_ollama(client, model, working_messages, tools=tools)
        try:
            async for raw in gen:
                piece = raw.get("message", {})
                p_thinking = piece.get("thinking", "") or ""
                p_content = piece.get("content", "") or ""
                p_tool_calls = piece.get("tool_calls")

                if p_thinking:
                    if not thinking_open:
                        yield self._chunk("<think>\n")
                        thinking_open = True
                    acc_thinking += p_thinking
                    yield self._chunk(p_thinking)

                # Capture tool_calls before evaluating the abort condition so
                # a chunk carrying both content and tool_calls together never
                # triggers a false abort.
                if p_tool_calls:
                    acc_tool_calls = p_tool_calls

                if p_content:
                    if thinking_open:
                        yield self._chunk("\n</think>\n\n")
                        thinking_open = False
                    acc_content += p_content
                    if not suppress_content:
                        yield self._chunk(p_content)

                    if abort_on_bare_content and acc_tool_calls is None:
                        aborted = True
                        break

                if raw.get("done"):
                    stats = {k: v for k, v in raw.items() if k != "message"}
                    break
        finally:
            # Explicitly close the underlying generator so the httpx stream
            # context manager exits now, not whenever GC gets to it. This is
            # what actually makes Ollama stop generating on abort.
            await gen.aclose()

        if thinking_open:
            yield self._chunk("\n</think>\n\n")

        self._last_message = {
            "role": "assistant",
            "content": acc_content,
            "thinking": acc_thinking,
            "tool_calls": acc_tool_calls,
            "aborted": aborted,
            "stats": stats,
        }

    async def pipe(
        self,
        body: dict,
        __tools__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> AsyncGenerator[dict, None]:
        messages = body.get("messages", [])

        os.environ["AGENT_OLLAMA_BASE_URL"] = self.valves.OLLAMA_BASE_URL

        tool_specs = []
        tool_funcs = {}
        if __tools__:
            for key, tool in __tools__.items():
                tool_specs.append({"type": "function", "function": tool["spec"]})
                tool_funcs[tool["spec"]["name"]] = tool["callable"]

        # This is the actual shared context. The thinking model's tool-call
        # turns, tool results, and reasoning get appended here as real
        # message turns, not summarized — the final model reads this list
        # directly. Its raw prose "answer" (content) is never appended.
        working_messages = list(self._clean(messages))
        total_tool_calls = 0

        # Accumulates generation stats (tokens/sec, durations, etc.) across
        # every model call in the turn, so they can be reported to the
        # client the same way a standard single-model response would be.
        usage: dict = {}

        async with httpx.AsyncClient(timeout=300) as client:
            round_num = 0
            while round_num < self.valves.MAX_ROUNDS:
                round_num += 1
                await self._emit_status(
                    __event_emitter__,
                    f"{self.valves.THINKING_MODEL} thinking (round {round_num}/{self.valves.MAX_ROUNDS})...",
                )

                thinking_messages = self._with_system_prompt(
                    working_messages, self.valves.THINKING_SYSTEM_PROMPT
                )

                async for piece in self._stream_and_forward(
                    client,
                    self.valves.THINKING_MODEL,
                    thinking_messages,
                    tools=tool_specs if tool_specs else None,
                    suppress_content=True,
                    abort_on_bare_content=self.valves.ABORT_THINKING_ON_CONTENT,
                ):
                    yield piece

                msg = self._last_message
                content = msg.get("content", "") or ""
                thinking = msg.get("thinking", "") or ""
                calls = msg.get("tool_calls")

                if self.valves.REPORT_FULL_TURN_STATS:
                    self._merge_usage(usage, msg.get("stats", {}))

                if not calls:
                    # Thinking model settled on a direct answer instead of
                    # calling more tools. Its content is fully discarded —
                    # never shown to the user, never passed to the final
                    # model — but its reasoning for this round is folded
                    # into the shared context so the final model still
                    # inherits the complete chain of thought.
                    if thinking:
                        working_messages.append(
                            {"role": "assistant", "content": "", "thinking": thinking}
                        )
                    break

                await self._emit_status(
                    __event_emitter__,
                    f"{self.valves.THINKING_MODEL} requested {len(calls)} tool call(s) in round {round_num}...",
                )

                working_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "thinking": thinking,
                        "tool_calls": calls,
                    }
                )

                # Run every tool call requested in this round. When
                # PARALLEL_TOOL_CALLS is on and there's more than one call,
                # they're launched concurrently via asyncio.gather. Either
                # way, results come back in a list aligned with `calls`, so
                # the emitted tool blocks and appended tool-result messages
                # below stay in the model's original request order — the
                # transcript the thinking/final models see is identical to
                # what a sequential run would have produced.
                call_indices = [total_tool_calls + i for i in range(1, len(calls) + 1)]
                total_tool_calls += len(calls)

                if self.valves.PARALLEL_TOOL_CALLS and len(calls) > 1:
                    await self._emit_status(
                        __event_emitter__,
                        f"Running {len(calls)} tool call(s) in parallel (round {round_num})...",
                    )
                    results = await asyncio.gather(
                        *(
                            self._run_tool_call(call, tool_funcs, idx)
                            for call, idx in zip(calls, call_indices)
                        )
                    )
                else:
                    results = []
                    for i, (call, idx) in enumerate(zip(calls, call_indices), start=1):
                        fname = call["function"]["name"]
                        await self._emit_status(
                            __event_emitter__,
                            f"Running tool {i}/{len(calls)} (round {round_num}): {fname}...",
                        )
                        results.append(await self._run_tool_call(call, tool_funcs, idx))

                for call_id, fname, fargs, result in results:
                    yield self._chunk(
                        self._tool_call_block(call_id, fname, fargs, result)
                    )

                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fname,
                            "content": (
                                result
                                if isinstance(result, str)
                                else json.dumps(result)
                            ),
                        }
                    )
                # loop continues -> thinking model gets another round with results in context
            else:
                await self._emit_status(
                    __event_emitter__,
                    f"Hit max rounds ({self.valves.MAX_ROUNDS}) — cutting over to final model.",
                )

            # Context transplant: the final model just continues this exact
            # conversation. No injected instructions, no summary of what
            # happened — it sees the same tool_calls/tool/thinking turns and
            # picks up from there (aside from its own system prompt, if
            # configured, swapped in below). Streamed live, same as the
            # thinking model. suppress_content is left False here: this is
            # the real answer.
            await self._emit_status(
                __event_emitter__, f"{self.valves.FINAL_MODEL} continuing..."
            )

            final_messages = self._with_system_prompt(
                working_messages, self.valves.FINAL_SYSTEM_PROMPT
            )

            async for piece in self._stream_and_forward(
                client, self.valves.FINAL_MODEL, final_messages
            ):
                yield piece

            # The final model's own stats are always included, regardless
            # of REPORT_FULL_TURN_STATS -- that flag only controls whether
            # the thinking rounds' stats are folded in as well.
            self._merge_usage(usage, self._last_message.get("stats", {}))

        await self._emit_status(__event_emitter__, "Done", done=True)
        yield self._chunk(
            "",
            finish_reason="stop",
            usage=self._finalize_usage(usage) if usage else None,
        )


def _is_async(func):
    import inspect

    return inspect.iscoroutinefunction(func)
