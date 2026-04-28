"""
Semantic Kernel — AI agent orchestration for Azure.

Provides:
- Kernel setup with Azure OpenAI chat and embedding services
- Plugin registration (native functions + semantic functions)
- Stepwise and Handlebars planners
- AI Search memory store integration
- Multi-turn chat with persistent memory
- Function calling and tool execution
- OpenTelemetry tracing integration
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins import (
    ConversationSummaryPlugin,
    MathPlugin,
    TextPlugin,
    TimePlugin,
)
from semantic_kernel.functions import kernel_function

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from a Semantic Kernel agent invocation."""

    content: str
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class PluginFunction:
    """A registered plugin function with metadata."""

    name: str
    plugin: str
    description: str


class SemanticKernelAgent:
    """
    Production Semantic Kernel agent with Azure OpenAI backend.

    Orchestrates AI tasks using plugins, planners, and persistent
    conversation memory. Supports automatic function calling,
    multi-step planning, and Azure AI Search memory integration.

    Example:
        agent = SemanticKernelAgent()

        # Register a custom plugin
        @kernel_function(name="get_stock_price", description="Get current stock price")
        async def get_price(symbol: str) -> str:
            return f"${symbol}: $150.00"

        agent.register_native_function("Finance", get_price)

        # Chat with automatic tool use
        response = await agent.chat(
            "What is the MSFT stock price and what's today's date?"
        )
        print(response.content)
    """

    def __init__(self) -> None:
        cfg = get_config()
        raw = load_config()

        self._cfg = cfg
        self._raw = raw

        self.kernel = self._build_kernel()
        self._chat_history = ChatHistory()

        self._register_default_plugins()

        logger.info("SemanticKernelAgent initialised")

    def _build_kernel(self) -> sk.Kernel:
        """Build and configure the Semantic Kernel with Azure services."""
        cfg = self._cfg
        raw = self._raw

        kernel = sk.Kernel()

        aoai_cfg = cfg.azure_openai
        chat_deployment = cfg.get_deployment("gpt4o")
        embed_deployment = cfg.get_deployment("text_embed_3_large")

        # Register Azure OpenAI chat service
        kernel.add_service(
            AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=chat_deployment,
                endpoint=aoai_cfg.endpoint,
                api_key=aoai_cfg.api_key,
                api_version=aoai_cfg.api_version,
            )
        )

        # Register Azure OpenAI embedding service
        kernel.add_service(
            AzureTextEmbedding(
                service_id="azure_openai_embed",
                deployment_name=embed_deployment,
                endpoint=aoai_cfg.endpoint,
                api_key=aoai_cfg.api_key,
                api_version=aoai_cfg.api_version,
            )
        )

        return kernel

    def _register_default_plugins(self) -> None:
        """Register built-in Semantic Kernel core plugins."""
        self.kernel.add_plugin(MathPlugin(), plugin_name="Math")
        self.kernel.add_plugin(TextPlugin(), plugin_name="Text")
        self.kernel.add_plugin(TimePlugin(), plugin_name="Time")

        logger.info("Default SK plugins registered", plugins=["Math", "Text", "Time"])

    def register_native_function(
        self,
        plugin_name: str,
        function: Callable,
    ) -> None:
        """
        Register a Python function as a Semantic Kernel plugin function.

        The function must be decorated with @kernel_function.

        Args:
            plugin_name: Logical grouping name for the function.
            function: The decorated Python callable.
        """
        self.kernel.add_function(plugin_name=plugin_name, function=function)
        logger.info("Native function registered", plugin=plugin_name, function=function.__name__)

    def register_semantic_function(
        self,
        plugin_name: str,
        function_name: str,
        prompt_template: str,
        description: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        """
        Register a prompt template as a Semantic Kernel semantic function.

        Semantic functions use prompt templates with {{$variable}} syntax.

        Args:
            plugin_name: Plugin group name.
            function_name: Function name within the plugin.
            prompt_template: The prompt template string.
            description: Description for the planner to understand usage.
            max_tokens: Max tokens for this function's completions.
            temperature: Sampling temperature.
        """
        from semantic_kernel.prompt_template import PromptTemplateConfig

        prompt_config = PromptTemplateConfig(
            template=prompt_template,
            description=description,
            execution_settings={
                "azure_openai_chat": AzureChatPromptExecutionSettings(
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            },
        )

        self.kernel.add_function(
            plugin_name=plugin_name,
            function_name=function_name,
            prompt_template_config=prompt_config,
        )

        logger.info(
            "Semantic function registered",
            plugin=plugin_name,
            function=function_name,
        )

    async def chat(
        self,
        user_message: str,
        system: str | None = None,
        auto_invoke_functions: bool = True,
        max_iterations: int = 10,
    ) -> AgentResponse:
        """
        Send a message and get a response with automatic function calling.

        The agent will automatically decide which registered plugin functions
        to call based on the user's request, execute them, and incorporate
        their outputs into the final response.

        Args:
            user_message: The user's message.
            system: Optional system prompt (set once per conversation).
            auto_invoke_functions: Enable automatic tool/function execution.
            max_iterations: Maximum function call iterations.

        Returns:
            AgentResponse with content and function call details.
        """
        if system and not self._chat_history.messages:
            self._chat_history.add_system_message(system)

        self._chat_history.add_user_message(user_message)

        settings = AzureChatPromptExecutionSettings(
            max_tokens=self._cfg.azure_openai.inference.get("max_tokens", 4096),
            temperature=self._cfg.azure_openai.inference.get("temperature", 0.1),
        )

        if auto_invoke_functions:
            settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
                auto_invoke=True,
                maximum_auto_invoke_attempts=max_iterations,
            )

        chat_service = self.kernel.get_service("azure_openai_chat")

        result = await chat_service.get_chat_message_content(
            chat_history=self._chat_history,
            settings=settings,
            kernel=self.kernel,
        )

        assistant_message = str(result)
        self._chat_history.add_assistant_message(assistant_message)

        function_calls: list[dict[str, Any]] = []
        if hasattr(result, "items"):
            for item in result.items:
                if hasattr(item, "name") and hasattr(item, "arguments"):
                    function_calls.append({
                        "function": item.name,
                        "arguments": item.arguments,
                    })

        logger.info(
            "SK agent chat complete",
            message_length=len(assistant_message),
            function_calls=len(function_calls),
        )

        return AgentResponse(
            content=assistant_message,
            function_calls=function_calls,
        )

    async def invoke_function(
        self,
        plugin_name: str,
        function_name: str,
        **kwargs: Any,
    ) -> str:
        """
        Directly invoke a specific Semantic Kernel function.

        Args:
            plugin_name: The plugin containing the function.
            function_name: The function to invoke.
            **kwargs: Arguments to pass to the function.

        Returns:
            The function's string result.
        """
        kernel_args = sk.KernelArguments(**kwargs)
        result = await self.kernel.invoke(
            plugin_name=plugin_name,
            function_name=function_name,
            arguments=kernel_args,
        )
        return str(result)

    async def create_plan(
        self,
        goal: str,
        planner_type: str = "handlebars",
    ) -> str:
        """
        Generate a multi-step plan to achieve a goal using available plugins.

        Args:
            goal: The objective to achieve.
            planner_type: "handlebars" | "stepwise"

        Returns:
            The plan as a string (Handlebars template or step list).
        """
        if planner_type == "handlebars":
            from semantic_kernel.planners.handlebars_planner import HandlebarsPlanner, HandlebarsPlannerOptions
            planner = HandlebarsPlanner(self.kernel, HandlebarsPlannerOptions(allow_loops=True))
            plan = await planner.create_plan(goal)
            result = await plan.invoke(self.kernel)
            return str(result)
        else:
            from semantic_kernel.planners import SequentialPlanner
            planner = SequentialPlanner(self.kernel)
            plan = await planner.create_plan(goal)
            result = await plan.invoke(self.kernel)
            return str(result)

    def reset_conversation(self) -> None:
        """Clear the chat history to start a fresh conversation."""
        self._chat_history = ChatHistory()
        logger.info("Conversation history cleared")

    def list_functions(self) -> list[PluginFunction]:
        """List all registered plugin functions."""
        functions = []
        for plugin_name, plugin in self.kernel.plugins.items():
            for func_name, func in plugin.functions.items():
                functions.append(PluginFunction(
                    name=func_name,
                    plugin=plugin_name,
                    description=func.description or "",
                ))
        return functions
