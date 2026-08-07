"""
title: Agent
description: One model thinks and calls tools across multiple rounds; a second model inherits that exact context and finishes the response.
version: 0.1.0
"""

import html
import json
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable, AsyncGenerator


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
        THINKING_MODEL: str = Field(
            default="Qwen3-5-4B-heretic-i1-IQ4-XS:latest",
            description="Model that reasons and calls tools.",
        )
        FINAL_MODEL: str = Field(
            default="gemma-4-E4B-it-qat-q4-0-heretic-Q4-0-PURE:latest",
            description="Model that continues from the thinking model's context and produces the response.",
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
                "the stream immediately instead of letting it finish. Saves "
                "compute on a response that gets discarded anyway. Disable if "
                "your thinking model sometimes emits commentary content BEFORE "
                "a tool call in the same turn — aborting would cut that call off."
            ),
        )

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

    def _chunk(self, content: str, finish_reason: Optional[str] = None) -> dict:
        delta = {"content": content} if content else {}
        return {"choices": [{"delta": delta, "finish_reason": finish_reason}]}

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
        wrap_content_as_discardable: bool = False,
        abort_on_bare_content: bool = False,
    ):
        """Streams a model's output, forwarding tokens live as they arrive,
        and stashes the fully-accumulated message (content, thinking,
        tool_calls) on self._last_message once done.

        Wraps streamed reasoning tokens in <think>...</think> so the UI
        renders them the same way the old buffered version did.

        wrap_content_as_discardable: if True, plain `content` tokens (as
        opposed to `thinking`) get wrapped in their own collapsible
        <details> block instead of streaming inline. Use this for the
        thinking model, whose bare content is never read by the final
        model and would otherwise look like part of the real answer.

        abort_on_bare_content: if True, and content starts arriving with
        no tool_calls seen yet in this message, close the connection
        immediately rather than let the model finish. This tells Ollama
        to stop generating (it detects the client disconnect), saving
        compute on output that's going to be discarded. Only safe when
        the model doesn't interleave commentary content before a tool
        call within the same turn.
        """
        acc_thinking = ""
        acc_content = ""
        acc_tool_calls = None
        thinking_open = False
        content_wrapper_open = False
        aborted = False

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
                    if wrap_content_as_discardable and not content_wrapper_open:
                        yield self._chunk(
                            '<details type="discarded_draft" done="false">\n'
                            "<summary>Thinking model's draft answer "
                            "(discarded — not seen by final model)</summary>\n\n"
                        )
                        content_wrapper_open = True
                    acc_content += p_content
                    yield self._chunk(p_content)

                    if abort_on_bare_content and acc_tool_calls is None:
                        aborted = True
                        break

                if raw.get("done"):
                    break
        finally:
            # Explicitly close the underlying generator so the httpx stream
            # context manager exits now, not whenever GC gets to it. This is
            # what actually makes Ollama stop generating on abort.
            await gen.aclose()

        if thinking_open:
            yield self._chunk("\n</think>\n\n")
        if content_wrapper_open:
            if aborted:
                yield self._chunk("\n\n*(cut short — generation aborted)*")
            yield self._chunk("\n\n</details>\n\n")

        self._last_message = {
            "role": "assistant",
            "content": acc_content,
            "thinking": acc_thinking,
            "tool_calls": acc_tool_calls,
        }

    async def pipe(
        self,
        body: dict,
        __tools__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> AsyncGenerator[dict, None]:
        messages = body.get("messages", [])

        tool_specs = []
        tool_funcs = {}
        if __tools__:
            for key, tool in __tools__.items():
                tool_specs.append({"type": "function", "function": tool["spec"]})
                tool_funcs[tool["spec"]["name"]] = tool["callable"]

        # This is the actual shared context. The thinking model's tool-call
        # turns and tool results get appended here as real message turns,
        # not summarized — the final model reads this list directly.
        working_messages = list(self._clean(messages))
        total_tool_calls = 0

        async with httpx.AsyncClient(timeout=300) as client:
            round_num = 0
            while round_num < self.valves.MAX_ROUNDS:
                round_num += 1
                await self._emit_status(
                    __event_emitter__,
                    f"{self.valves.THINKING_MODEL} thinking (round {round_num}/{self.valves.MAX_ROUNDS})...",
                )

                async for piece in self._stream_and_forward(
                    client,
                    self.valves.THINKING_MODEL,
                    working_messages,
                    tools=tool_specs if tool_specs else None,
                    wrap_content_as_discardable=True,
                    abort_on_bare_content=self.valves.ABORT_THINKING_ON_CONTENT,
                ):
                    yield piece

                msg = self._last_message
                content = msg.get("content", "") or ""
                calls = msg.get("tool_calls")

                if not calls:
                    # Nothing left to call — whatever content it produced here
                    # is discarded (it was already streamed to the user as
                    # "thinking-model output", but working_messages doesn't
                    # get it appended); the final model writes the actual reply.
                    break

                await self._emit_status(
                    __event_emitter__,
                    f"{self.valves.THINKING_MODEL} requested {len(calls)} tool call(s) in round {round_num}...",
                )

                working_messages.append(
                    {"role": "assistant", "content": content, "tool_calls": calls}
                )

                for i, call in enumerate(calls, start=1):
                    total_tool_calls += 1
                    fname = call["function"]["name"]
                    fargs = call["function"].get("arguments", {})
                    call_id = call.get("id") or f"call_{fname}_{total_tool_calls}"

                    await self._emit_status(
                        __event_emitter__,
                        f"Running tool {i}/{len(calls)} (round {round_num}): {fname}...",
                    )

                    if fname in tool_funcs:
                        try:
                            result = (
                                await tool_funcs[fname](**fargs)
                                if _is_async(tool_funcs[fname])
                                else tool_funcs[fname](**fargs)
                            )
                        except Exception as e:
                            result = f"Tool error: {e}"
                    else:
                        result = f"Tool '{fname}' not available"

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
            # happened — it sees the same tool_calls/tool turns and picks up
            # from there. Streamed live, same as the thinking model.
            await self._emit_status(
                __event_emitter__, f"{self.valves.FINAL_MODEL} continuing..."
            )
            async for piece in self._stream_and_forward(
                client, self.valves.FINAL_MODEL, working_messages
            ):
                yield piece

        await self._emit_status(__event_emitter__, "Done", done=True)
        yield self._chunk("", finish_reason="stop")


def _is_async(func):
    import inspect

    return inspect.iscoroutinefunction(func)
