from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import NoneType
from typing import TYPE_CHECKING, Any, Protocol, Union, get_args, get_origin, get_type_hints, overload

from vv_agent.tools.base import ToolContext
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.outputs import ToolOutput, ToolOutputError, ToolOutputFile, ToolOutputImage, ToolOutputJson, ToolOutputText
from vv_agent.types import ToolDirective, ToolExecutionResult, ToolResultStatus

if TYPE_CHECKING:
    from vv_agent.tools.executor import FunctionToolExecutor

ApprovalPredicate = Callable[[Any, Any], bool]
ToolErrorFormatter = Callable[[Exception], str]


class Tool(Protocol):
    name: str
    description: str
    params_json_schema: dict[str, Any]
    strict_json_schema: bool
    is_enabled: bool | Callable[[Any, Any], bool]
    needs_approval: bool | ApprovalPredicate

    def invoke(self, context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutput: ...


@dataclass(slots=True)
class FunctionTool:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke: Callable[[ToolContext | None, dict[str, Any]], ToolOutput]
    is_enabled: bool | Callable[[Any, Any], bool] = True
    needs_approval: bool | ApprovalPredicate = False
    strict_json_schema: bool = True
    timeout_seconds: float | None = None
    exposure: ToolExposure = ToolExposure.DIRECT
    failure_error_function: ToolErrorFormatter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.params_json_schema),
                "strict": self.strict_json_schema,
            },
        }

    def invoke(self, context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutput:
        try:
            if self.timeout_seconds is None:
                return self.on_invoke(context, arguments)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vv-agent-tool-{self.name}")
            future = executor.submit(self.on_invoke, context, arguments)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError:
                future.cancel()
                return ToolOutputError(
                    message=f"Tool {self.name} timed out after {self.timeout_seconds:g} seconds.",
                    error_code="tool_timeout",
                    retryable=True,
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            if _is_cancelled_error(exc):
                raise
            message = (
                self.failure_error_function(exc)
                if self.failure_error_function is not None
                else f"Tool execution failed ({self.name}): {exc}"
            )
            return ToolOutputError(message=message, error_code="tool_execution_failed")

    def to_executor(self) -> FunctionToolExecutor:
        from vv_agent.tools.executor import FunctionToolExecutor

        return FunctionToolExecutor(self)

    def to_tool_execution_result(
        self,
        output: ToolOutput,
        *,
        tool_call_id: str = "",
    ) -> ToolExecutionResult:
        if isinstance(output, ToolOutputText):
            return ToolExecutionResult(tool_call_id=tool_call_id, content=output.text, metadata=dict(output.metadata))
        if isinstance(output, ToolOutputJson):
            metadata = {"output_type": "json", **dict(output.metadata)}
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                content=json.dumps(output.data, ensure_ascii=False),
                metadata=metadata,
            )
        if isinstance(output, ToolOutputImage):
            metadata = {"output_type": "image", **dict(output.metadata)}
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                content=json.dumps(
                    {"url": output.url, "path": output.path, "mime_type": output.mime_type}, ensure_ascii=False
                ),
                image_url=output.url,
                image_path=output.path,
                metadata=metadata,
            )
        if isinstance(output, ToolOutputFile):
            metadata = {"output_type": "file", **dict(output.metadata)}
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                content=json.dumps({"path": output.path, "mime_type": output.mime_type}, ensure_ascii=False),
                metadata=metadata,
            )
        if isinstance(output, ToolOutputError):
            metadata = {
                "output_type": "error",
                "retryable": output.retryable,
                **dict(output.metadata),
            }
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                status="error",
                status_code=ToolResultStatus.ERROR,
                error_code=output.error_code,
                content=json.dumps(
                    {
                        "ok": False,
                        "error": output.message,
                        "error_code": output.error_code,
                        "retryable": output.retryable,
                    },
                    ensure_ascii=False,
                ),
                metadata=metadata,
            )
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            content=str(output),
            directive=ToolDirective.CONTINUE,
        )


@overload
def function_tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    params_json_schema: dict[str, Any] | None = None,
    strict_json_schema: bool = True,
    is_enabled: bool | Callable[[Any, Any], bool] = True,
    needs_approval: bool | ApprovalPredicate = False,
    timeout_seconds: float | None = None,
    exposure: ToolExposure = ToolExposure.DIRECT,
    failure_error_function: ToolErrorFormatter | None = None,
) -> FunctionTool: ...


@overload
def function_tool(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    params_json_schema: dict[str, Any] | None = None,
    strict_json_schema: bool = True,
    is_enabled: bool | Callable[[Any, Any], bool] = True,
    needs_approval: bool | ApprovalPredicate = False,
    timeout_seconds: float | None = None,
    exposure: ToolExposure = ToolExposure.DIRECT,
    failure_error_function: ToolErrorFormatter | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def function_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    params_json_schema: dict[str, Any] | None = None,
    strict_json_schema: bool = True,
    is_enabled: bool | Callable[[Any, Any], bool] = True,
    needs_approval: bool | ApprovalPredicate = False,
    timeout_seconds: float | None = None,
    exposure: ToolExposure = ToolExposure.DIRECT,
    failure_error_function: ToolErrorFormatter | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    current_frame = inspect.currentframe()
    decorator_frame = current_frame.f_back if current_frame is not None else None
    decorator_localns = dict(decorator_frame.f_locals) if decorator_frame is not None else {}

    def decorate(target: Callable[..., Any]) -> FunctionTool:
        tool_name = name or getattr(target, "__name__", target.__class__.__name__)
        tool_description = description or inspect.getdoc(target) or ""
        current_decorate_frame = inspect.currentframe()
        caller = current_decorate_frame.f_back if current_decorate_frame is not None else None
        localns = dict(decorator_localns)
        if caller is not None:
            localns.update(caller.f_locals)
        schema, argument_builder, pass_context = _schema_and_argument_builder(
            target,
            explicit_schema=params_json_schema,
            localns=localns,
        )

        def invoke(context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutput:
            positional, keyword = argument_builder(arguments)
            result = target(context, *positional, **keyword) if pass_context else target(*positional, **keyword)
            return _coerce_tool_output(result)

        return FunctionTool(
            name=tool_name,
            description=tool_description,
            params_json_schema=schema,
            on_invoke=invoke,
            is_enabled=is_enabled,
            needs_approval=needs_approval,
            strict_json_schema=strict_json_schema,
            timeout_seconds=timeout_seconds,
            exposure=exposure,
            failure_error_function=failure_error_function,
        )

    if func is not None:
        return decorate(func)
    return decorate


def adapt_tool(tool: Tool) -> FunctionTool:
    if isinstance(tool, FunctionTool):
        return tool

    required_attributes = (
        "name",
        "description",
        "params_json_schema",
        "strict_json_schema",
        "is_enabled",
        "needs_approval",
        "invoke",
    )
    missing = [attribute for attribute in required_attributes if not hasattr(tool, attribute)]
    if missing:
        raise TypeError(
            f"Agent tool must be a FunctionTool, ToolExecutor, or implement the Tool protocol; missing: {', '.join(missing)}"
        )
    invoke = tool.invoke
    if not callable(invoke):
        raise TypeError("Tool protocol attribute 'invoke' must be callable")
    params_json_schema = tool.params_json_schema
    if not isinstance(params_json_schema, dict):
        raise TypeError("Tool protocol attribute 'params_json_schema' must be a dict")

    raw_exposure = getattr(tool, "exposure", ToolExposure.DIRECT)
    try:
        exposure = ToolExposure(raw_exposure)
    except ValueError as exc:
        raise TypeError(f"Unsupported tool exposure: {raw_exposure!r}") from exc
    metadata = getattr(tool, "metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Tool protocol attribute 'metadata' must be a dict when provided")

    def on_invoke(context: ToolContext | None, arguments: dict[str, Any]) -> ToolOutput:
        return _coerce_tool_output(invoke(context, arguments))

    return FunctionTool(
        name=str(tool.name),
        description=str(tool.description),
        params_json_schema=dict(params_json_schema),
        on_invoke=on_invoke,
        is_enabled=tool.is_enabled,
        needs_approval=tool.needs_approval,
        strict_json_schema=bool(tool.strict_json_schema),
        timeout_seconds=getattr(tool, "timeout_seconds", None),
        exposure=exposure,
        failure_error_function=getattr(tool, "failure_error_function", None),
        metadata=dict(metadata),
    )


def _schema_and_argument_builder(
    func: Callable[..., Any],
    *,
    explicit_schema: dict[str, Any] | None,
    localns: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Callable[[dict[str, Any]], tuple[list[Any], dict[str, Any]]], bool]:
    signature = inspect.signature(func)
    hints = _safe_type_hints(func, localns=localns)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    pass_context = False
    if parameters:
        first = parameters[0]
        annotation = hints.get(first.name, first.annotation)
        if _is_tool_context_type(annotation):
            pass_context = True
            parameters = parameters[1:]

    if explicit_schema is not None:
        return dict(explicit_schema), lambda arguments: ([], dict(arguments)), pass_context

    if len(parameters) == 1:
        parameter = parameters[0]
        annotation = hints.get(parameter.name, parameter.annotation)
        if _is_dataclass_type(annotation):
            schema = _schema_from_dataclass(annotation, localns=localns)

            def build(arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
                return [annotation(**_apply_dataclass_defaults(annotation, arguments))], {}

            return schema, build, pass_context
        if _is_typed_dict_type(annotation):
            schema = _schema_from_typed_dict(annotation, localns=localns)
            return schema, lambda arguments: ([dict(arguments)], {}), pass_context
        if _is_pydantic_model_type(annotation):
            schema = dict(annotation.model_json_schema())

            def build(arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
                return [annotation.model_validate(arguments)], {}

            return _normalize_object_schema(schema), build, pass_context

    schema = _schema_from_signature(parameters, hints)

    def build(arguments: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        keyword: dict[str, Any] = {}
        for parameter in parameters:
            if parameter.name in arguments:
                keyword[parameter.name] = arguments[parameter.name]
            elif parameter.default is not inspect.Parameter.empty:
                keyword[parameter.name] = parameter.default
        return [], keyword

    return schema, build, pass_context


def _coerce_tool_output(value: Any) -> ToolOutput:
    if isinstance(value, ToolOutputText | ToolOutputJson | ToolOutputImage | ToolOutputFile | ToolOutputError):
        return value
    if isinstance(value, str):
        return ToolOutputText(text=value)
    if isinstance(value, dict | list | tuple | int | float | bool) or value is None:
        return ToolOutputJson(data=value)
    return ToolOutputText(text=str(value))


def _is_cancelled_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "CancelledError" and exc.__class__.__module__ == "vv_agent.runtime.cancellation"


def _schema_from_signature(parameters: list[inspect.Parameter], hints: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in parameters:
        annotation = hints.get(parameter.name, parameter.annotation)
        schema = _schema_for_type(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            schema["default"] = parameter.default
        properties[parameter.name] = schema
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _schema_from_dataclass(cls: type[Any], *, localns: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = _safe_type_hints(cls, localns=localns)
    for item in fields(cls):
        annotation = hints.get(item.name, item.type)
        schema = _schema_for_type(annotation)
        default = _field_default(item)
        if default is MISSING:
            required.append(item.name)
        else:
            schema["default"] = default
        properties[item.name] = schema
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _schema_from_typed_dict(cls: type[Any], *, localns: dict[str, Any] | None = None) -> dict[str, Any]:
    hints = _safe_type_hints(cls, localns=localns)
    required_keys = set(getattr(cls, "__required_keys__", set(hints)))
    properties = {name: _schema_for_type(annotation) for name, annotation in hints.items()}
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in hints if name in required_keys],
        "additionalProperties": False,
    }


def _schema_for_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, getattr(__import__("types"), "UnionType", object)}:
        non_none = [arg for arg in args if arg is not NoneType]
        if len(non_none) == 1:
            return _schema_for_type(non_none[0])
    if annotation is str or annotation is Path:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if origin in {list, tuple, set}:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _schema_for_type(item_type)}
    if origin is dict:
        return {"type": "object"}
    if _is_dataclass_type(annotation):
        return _schema_from_dataclass(annotation)
    return {"type": "string"}


def _is_tool_context_type(annotation: Any) -> bool:
    return annotation is ToolContext


def _safe_type_hints(target: Any, *, localns: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return get_type_hints(target, globalns=getattr(target, "__globals__", None), localns=localns)
    except Exception:
        return dict(getattr(target, "__annotations__", {}))


def _apply_dataclass_defaults(cls: type[Any], arguments: dict[str, Any]) -> dict[str, Any]:
    values = dict(arguments)
    for item in fields(cls):
        if item.name in values:
            continue
        default = _field_default(item)
        if default is not MISSING:
            values[item.name] = default
    return values


def _field_default(item: Any) -> Any:
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:
        return item.default_factory()
    return MISSING


def _is_dataclass_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and is_dataclass(annotation)


def _is_typed_dict_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and hasattr(annotation, "__required_keys__") and hasattr(annotation, "__annotations__")


def _is_pydantic_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and callable(getattr(annotation, "model_json_schema", None)) and callable(
        getattr(annotation, "model_validate", None)
    )


def _normalize_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object":
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        schema.setdefault("additionalProperties", False)
    return schema
