"""
UI Components for LLM Cross-Talk Analyzer - Phase 2 Enhanced Version
Includes: Convergence chart, diff viewer, export controls, model selection
"""

import reflex as rx
from app.state import ComparisonState
import plotly.graph_objects as go


def header() -> rx.Component:
    """Renders the application header with enhanced controls."""
    return rx.el.header(
        rx.el.div(
            rx.el.div(
                rx.icon("git-compare-arrows", size=32, class_name="text-indigo-500"),
                rx.el.h1(
                    "LLM Cross-Talk Analyzer",
                    class_name="text-2xl font-bold text-gray-800 tracking-tight",
                ),
                class_name="flex items-center gap-4",
            ),
            rx.el.div(
                rx.el.button(
                    rx.icon("settings", class_name="mr-2 h-5 w-5"),
                    "Settings",
                    on_click=ComparisonState.toggle_settings,
                    class_name="flex items-center px-4 py-2 bg-gray-100 text-gray-700 font-semibold rounded-lg hover:bg-gray-200 transition-colors",
                ),
                rx.cond(
                    ComparisonState.has_responses,
                    rx.el.button(
                        rx.icon("git-compare", class_name="mr-2 h-5 w-5"),
                        "View Diffs",
                        on_click=ComparisonState.toggle_diff_viewer,
                        class_name="flex items-center px-4 py-2 bg-blue-100 text-blue-700 font-semibold rounded-lg hover:bg-blue-200 transition-colors",
                    ),
                ),
                rx.cond(
                    ComparisonState.has_responses,
                    rx.el.button(
                        rx.icon("trash-2", class_name="mr-2 h-5 w-5"),
                        "Clear All",
                        on_click=ComparisonState.clear_all,
                        class_name="flex items-center px-4 py-2 bg-red-100 text-red-700 font-semibold rounded-lg hover:bg-red-200 transition-colors",
                    ),
                ),
                class_name="flex items-center gap-3",
            ),
            class_name="flex items-center justify-between w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
        ),
        class_name="w-full py-4 bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-10",
    )


def settings_panel() -> rx.Component:
    """Renders the collapsible settings panel with model selection."""
    return rx.cond(
        ComparisonState.show_settings,
        rx.el.div(
            rx.el.div(
                rx.el.h3(
                    "Configuration Settings",
                    class_name="text-xl font-bold text-gray-800 mb-6",
                ),
                rx.el.div(
                    rx.el.h4(
                        "Model Selection",
                        class_name="text-lg font-semibold text-gray-700 mb-4",
                    ),
                    rx.el.div(
                        rx.el.div(
                            rx.el.label(
                                "OpenAI Model",
                                class_name="text-sm font-semibold text-gray-700 mb-2 block",
                            ),
                            rx.select(
                                ComparisonState.openai_models_options,
                                value=ComparisonState.openai_model,
                                on_change=ComparisonState.set_openai_model,
                                class_name="w-full",
                            ),
                            class_name="flex-1",
                        ),
                        rx.el.div(
                            rx.el.label(
                                "Claude Model",
                                class_name="text-sm font-semibold text-gray-700 mb-2 block",
                            ),
                            rx.select(
                                ComparisonState.claude_models_options,
                                value=ComparisonState.claude_model,
                                on_change=ComparisonState.set_claude_model,
                                class_name="w-full",
                            ),
                            class_name="flex-1",
                        ),
                        class_name="grid md:grid-cols-2 gap-4 mb-4",
                    ),
                    rx.el.div(
                        rx.el.label(
                            "OpenAI Reasoning Effort",
                            class_name="text-sm font-semibold text-gray-700 mb-2 block",
                        ),
                        rx.select(
                            ComparisonState.reasoning_efforts_options,
                            value=ComparisonState.reasoning_effort,
                            on_change=ComparisonState.set_reasoning_effort,
                            class_name="w-full md:w-1/2",
                        ),
                        rx.el.p(
                            "Higher effort = more thorough reasoning (GPT-5 models only)",
                            class_name="text-xs text-gray-500 mt-1",
                        ),
                        class_name="mt-4",
                    ),
                    rx.el.div(
                        rx.el.label(
                            "Claude Extended Thinking",
                            class_name="text-sm font-semibold text-gray-700 mb-2 block",
                        ),
                        rx.el.div(
                            rx.switch(
                                checked=ComparisonState.extended_thinking_enabled,
                                on_change=ComparisonState.toggle_extended_thinking,
                            ),
                            rx.el.span(
                                rx.cond(
                                    ComparisonState.extended_thinking_enabled,
                                    "Enabled",
                                    "Disabled",
                                ),
                                class_name="ml-3 text-sm text-gray-700 font-medium",
                            ),
                            class_name="flex items-center",
                        ),
                        rx.cond(
                            ComparisonState.extended_thinking_enabled,
                            rx.el.div(
                                rx.el.label(
                                    "Thinking Budget (tokens)",
                                    class_name="text-xs font-semibold text-gray-600 mb-1 block mt-3",
                                ),
                                rx.select(
                                    ComparisonState.thinking_budget_options_str,
                                    value=ComparisonState.thinking_budget_tokens.to_string(),
                                    on_change=ComparisonState.set_thinking_budget,
                                    class_name="w-full md:w-1/2",
                                ),
                                class_name="mt-2",
                            ),
                        ),
                        rx.el.p(
                            "Shows Claude's reasoning process before the response",
                            class_name="text-xs text-gray-500 mt-1",
                        ),
                        class_name="mt-4",
                    ),
                    class_name="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.el.div(
                    rx.el.h4(
                        "Convergence Parameters",
                        class_name="text-lg font-semibold text-gray-700 mb-4",
                    ),
                    rx.el.div(
                        rx.el.label(
                            "Convergence Threshold",
                            class_name="text-sm font-semibold text-gray-700 mb-2 block",
                        ),
                        rx.el.div(
                            rx.el.span(
                                f"{ComparisonState.convergence_threshold * 100:.0f}%",
                                class_name="text-lg font-bold text-indigo-600",
                            ),
                            class_name="mb-2",
                        ),
                        rx.el.input(
                            type="range",
                            key=ComparisonState.convergence_threshold,
                            default_value=ComparisonState.convergence_threshold,
                            min=0.85,
                            max=0.99,
                            step=0.01,
                            on_change=ComparisonState.update_convergence_threshold.throttle(
                                50
                            ),
                            class_name="w-full",
                        ),
                        rx.el.p(
                            "Minimum similarity (85-99%) required for convergence",
                            class_name="text-xs text-gray-500 mt-1",
                        ),
                        class_name="mb-4",
                    ),
                    rx.el.div(
                        rx.el.label(
                            "Maximum Iterations",
                            class_name="text-sm font-semibold text-gray-700 mb-2 block",
                        ),
                        rx.el.div(
                            rx.el.span(
                                ComparisonState.max_iterations.to_string(),
                                class_name="text-lg font-bold text-indigo-600",
                            ),
                            class_name="mb-2",
                        ),
                        rx.el.input(
                            type="range",
                            key=ComparisonState.max_iterations,
                            default_value=ComparisonState.max_iterations,
                            min=5,
                            max=20,
                            step=1,
                            on_change=ComparisonState.update_max_iterations.throttle(
                                50
                            ),
                            class_name="w-full",
                        ),
                        rx.el.p(
                            "Maximum iterations before stopping (5-20)",
                            class_name="text-xs text-gray-500 mt-1",
                        ),
                        class_name="mb-4",
                    ),
                    rx.el.div(
                        rx.el.label(
                            "Temperature",
                            class_name="text-sm font-semibold text-gray-700 mb-2 block",
                        ),
                        rx.el.div(
                            rx.el.span(
                                f"{ComparisonState.temperature:.2f}",
                                class_name="text-lg font-bold text-indigo-600",
                            ),
                            class_name="mb-2",
                        ),
                        rx.el.input(
                            type="range",
                            key=ComparisonState.temperature,
                            default_value=ComparisonState.temperature,
                            min=0.0,
                            max=1.0,
                            step=0.1,
                            on_change=ComparisonState.update_temperature.throttle(50),
                            class_name="w-full",
                        ),
                        rx.el.p(
                            "Response creativity: 0.0 (deterministic) to 1.0 (creative)",
                            class_name="text-xs text-gray-500 mt-1",
                        ),
                        class_name="mb-4",
                    ),
                    class_name="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200",
                ),
                rx.el.div(
                    rx.el.h4(
                        "Current Status",
                        class_name="text-sm font-semibold text-gray-700 mb-2",
                    ),
                    rx.el.div(
                        rx.el.div(
                            rx.icon("target", class_name="mr-2 h-4 w-4 text-gray-600"),
                            rx.el.span(
                                ComparisonState.convergence_status,
                                class_name="text-sm text-gray-700",
                            ),
                            class_name="flex items-center",
                        ),
                        rx.el.div(
                            rx.icon("layers", class_name="mr-2 h-4 w-4 text-gray-600"),
                            rx.el.span(
                                f"Iteration {ComparisonState.current_iteration_count} of {ComparisonState.max_iterations}",
                                class_name="text-sm text-gray-700",
                            ),
                            class_name="flex items-center mt-1",
                        ),
                        class_name="p-3 bg-gray-50 rounded-lg border border-gray-200",
                    ),
                ),
                class_name="w-full max-w-6xl p-6 bg-white rounded-2xl shadow-lg border border-gray-100",
            ),
            class_name="w-full flex justify-center mb-6",
        ),
    )


def convergence_chart() -> rx.Component:
    """Renders the convergence progress chart using Plotly."""
    return rx.cond(
        ComparisonState.has_responses & (ComparisonState.current_iteration_count > 1),
        rx.el.div(
            rx.el.h3(
                "Convergence Progress",
                class_name="text-xl font-bold text-gray-800 mb-4",
            ),
            rx.plotly(data=ComparisonState.convergence_fig, class_name="w-full h-80"),
            class_name="w-full max-w-6xl p-6 bg-white rounded-2xl shadow-lg border border-gray-100 mb-8",
        ),
    )


def diff_viewer() -> rx.Component:
    """Renders the iteration diff viewer."""
    return rx.cond(
        ComparisonState.show_diff_viewer
        & (ComparisonState.current_iteration_count > 1),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.el.h3(
                        "Iteration Diff Viewer",
                        class_name="text-xl font-bold text-gray-800",
                    ),
                    rx.el.button(
                        rx.icon("x", class_name="h-5 w-5"),
                        on_click=ComparisonState.toggle_diff_viewer,
                        class_name="text-gray-500 hover:text-gray-700 transition-colors",
                    ),
                    class_name="flex items-center justify-between mb-4",
                ),
                rx.el.div(
                    rx.el.label(
                        "Compare Iteration:",
                        class_name="text-sm font-semibold text-gray-700 mr-3",
                    ),
                    rx.select(
                        ComparisonState.diff_iteration_options,
                        value=ComparisonState.selected_iteration_for_diff.to_string(),
                        on_change=ComparisonState.select_iteration_for_diff,
                        class_name="w-32",
                    ),
                    rx.el.p(
                        "Shows changes from previous iteration",
                        class_name="text-xs text-gray-500 ml-3",
                    ),
                    class_name="flex items-center mb-6",
                ),
                rx.el.div(
                    rx.el.div(
                        rx.el.h4(
                            "OpenAI Changes",
                            class_name="text-lg font-semibold text-gray-800 mb-3 flex items-center",
                        ),
                        rx.el.div(
                            rx.html(ComparisonState.openai_diff_html),
                            class_name="p-4 bg-gray-50 rounded-lg border border-gray-200 overflow-auto max-h-96 font-mono text-sm",
                        ),
                        class_name="mb-6",
                    ),
                    rx.el.div(
                        rx.el.h4(
                            "Claude Changes",
                            class_name="text-lg font-semibold text-gray-800 mb-3 flex items-center",
                        ),
                        rx.el.div(
                            rx.html(ComparisonState.claude_diff_html),
                            class_name="p-4 bg-gray-50 rounded-lg border border-gray-200 overflow-auto max-h-96 font-mono text-sm",
                        ),
                    ),
                ),
                class_name="w-full max-w-6xl p-6 bg-white rounded-2xl shadow-lg border border-gray-100",
            ),
            class_name="w-full flex justify-center mb-8",
        ),
    )


def export_controls() -> rx.Component:
    """Renders export buttons for JSON and Markdown."""
    return rx.cond(
        ComparisonState.has_responses,
        rx.el.div(
            rx.el.h3(
                "Export Results", class_name="text-lg font-bold text-gray-800 mb-4"
            ),
            rx.el.div(
                rx.el.button(
                    rx.icon("download", class_name="mr-2 h-5 w-5"),
                    "Export JSON",
                    on_click=ComparisonState.export_json,
                    class_name="flex items-center px-6 py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 transition-all",
                ),
                rx.el.button(
                    rx.icon("file-text", class_name="mr-2 h-5 w-5"),
                    "Export Markdown",
                    on_click=ComparisonState.export_markdown,
                    class_name="flex items-center px-6 py-3 bg-purple-600 text-white font-semibold rounded-xl shadow-md hover:bg-purple-700 transition-all",
                ),
                class_name="flex items-center gap-4",
            ),
            class_name="w-full max-w-4xl p-6 bg-white rounded-2xl shadow-lg border border-gray-100 mb-8",
        ),
    )


def prompt_input_area() -> rx.Component:
    """Renders the prompt input form with mode selector."""
    return rx.el.div(
        rx.cond(~ComparisonState.is_loading, mode_selector()),
        rx.el.form(
            rx.el.textarea(
                name="prompt",
                placeholder="Enter your prompt here to begin the analysis...",
                class_name="w-full p-4 rounded-xl border-2 border-gray-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-shadow duration-200 resize-none text-base font-medium",
                rows=4,
            ),
            rx.el.div(
                rx.el.button(
                    rx.icon("play", class_name="mr-2"),
                    "Generate",
                    type="submit",
                    class_name="flex items-center justify-center px-6 py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed",
                    disabled=ComparisonState.is_loading,
                ),
                class_name="flex items-center justify-end mt-3",
            ),
            on_submit=ComparisonState.get_initial_responses,
            reset_on_submit=True,
            class_name="w-full",
        ),
        class_name="w-full max-w-4xl p-6 bg-white rounded-2xl shadow-lg border border-gray-100 flex flex-col gap-4",
    )


def mode_selector() -> rx.Component:
    """Renders the Manual/Automated mode toggle."""
    return rx.el.div(
        rx.el.button(
            rx.icon("user", class_name="mr-2 h-4 w-4"),
            "Manual",
            on_click=lambda: ComparisonState.set_mode("manual"),
            class_name=rx.cond(
                ComparisonState.mode == "manual",
                "flex items-center px-4 py-2 rounded-l-lg bg-indigo-600 text-white font-semibold z-10 shadow-inner",
                "flex items-center px-4 py-2 rounded-l-lg bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors",
            ),
        ),
        rx.el.button(
            rx.icon("bot", class_name="mr-2 h-4 w-4"),
            "Automated",
            on_click=lambda: ComparisonState.set_mode("automated"),
            class_name=rx.cond(
                ComparisonState.mode == "automated",
                "flex items-center px-4 py-2 rounded-r-lg bg-indigo-600 text-white font-semibold z-10 shadow-inner",
                "flex items-center px-4 py-2 rounded-r-lg bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors",
            ),
        ),
        class_name="flex items-center shadow-md rounded-xl border border-gray-300",
    )


def response_area() -> rx.Component:
    """Displays the side-by-side responses or a loading spinner."""
    return rx.el.div(
        rx.cond(
            ComparisonState.is_loading,
            rx.el.div(
                rx.cond(
                    ~ComparisonState.has_responses,
                    rx.el.div(
                        skeleton_card("OpenAI"),
                        skeleton_card("Claude"),
                        class_name="grid md:grid-cols-2 gap-8 w-full",
                    ),
                    rx.el.div(
                        rx.icon(
                            "loader-circle",
                            class_name="animate-spin h-12 w-12 text-indigo-500",
                        ),
                        rx.el.h3(
                            "Generating responses from OpenAI and Claude...",
                            class_name="text-lg font-semibold text-gray-600 mt-4",
                        ),
                        class_name="flex flex-col items-center justify-center p-12 bg-gray-50/50 border-2 border-dashed border-gray-200 rounded-2xl text-center w-full min-h-[300px]",
                    ),
                )
            ),
            rx.cond(
                ComparisonState.has_responses,
                rx.el.div(
                    convergence_chart(),
                    diff_viewer(),
                    rx.foreach(
                        ComparisonState.history, lambda item: iteration_view(item)
                    ),
                    rx.cond(
                        ComparisonState.mode == "manual",
                        iterate_button_section(),
                        automated_controls(),
                    ),
                    export_controls(),
                    class_name="flex flex-col items-center gap-10 w-full",
                ),
                placeholder_card(),
            ),
        ),
        class_name="w-full max-w-7xl mt-8",
    )


def iteration_view(item: rx.Var[dict]) -> rx.Component:
    """Displays a single iteration of responses with enhanced metrics."""
    return rx.el.div(
        rx.el.div(
            response_card(
                "OpenAI",
                item["openai_response"],
                "openai",
                item["iteration"].to_string(),
            ),
            response_card(
                "Claude",
                item["claude_response"],
                "claude",
                item["iteration"].to_string(),
            ),
            class_name="grid md:grid-cols-2 gap-8 w-full",
        ),
        rx.el.div(
            similarity_badge(item["similarity"]),
            rx.cond(item["iteration"] > 1, change_rate_badge(item["change_rate"])),
            class_name="flex items-center gap-4",
        ),
        class_name="flex flex-col items-center gap-4 w-full",
    )


def similarity_badge(similarity: rx.Var[float]) -> rx.Component:
    """Displays the similarity score as a badge with color coding."""
    percentage = (similarity * 100).to_string() + "%"
    return rx.el.div(
        rx.icon("check-check", class_name="mr-2"),
        "Similarity: ",
        percentage,
        class_name=rx.cond(
            similarity >= 0.95,
            "flex items-center px-4 py-2 bg-green-100 text-green-800 font-bold rounded-full border-2 border-green-300 shadow-sm",
            rx.cond(
                similarity >= 0.85,
                "flex items-center px-4 py-2 bg-blue-100 text-blue-800 font-semibold rounded-full border border-blue-200",
                "flex items-center px-4 py-2 bg-yellow-100 text-yellow-800 font-semibold rounded-full border border-yellow-200",
            ),
        ),
    )


def change_rate_badge(change_rate: rx.Var[float]) -> rx.Component:
    """Displays the change rate from previous iteration."""
    percentage = (change_rate * 100).to_string() + "%"
    return rx.el.div(
        rx.icon("activity", class_name="mr-2 h-4 w-4"),
        "Change: ",
        percentage,
        class_name="flex items-center px-3 py-1.5 bg-purple-100 text-purple-700 text-sm font-semibold rounded-full border border-purple-200",
    )


def iterate_button_section() -> rx.Component:
    """Renders the 'Iterate' button when in manual mode."""
    return rx.cond(
        (ComparisonState.mode == "manual") & ComparisonState.has_responses,
        rx.el.div(
            rx.cond(
                ~ComparisonState.converged,
                rx.el.button(
                    rx.cond(
                        ComparisonState.is_iterating,
                        rx.el.span(
                            rx.icon("loader-circle", class_name="animate-spin mr-2"),
                            "Iterating...",
                            class_name="flex items-center",
                        ),
                        rx.el.span(
                            rx.icon("git-compare-arrows", class_name="mr-2"),
                            "Iterate",
                            class_name="flex items-center",
                        ),
                    ),
                    on_click=ComparisonState.iterate_manual_mode,
                    disabled=ComparisonState.is_iterating
                    | (
                        ComparisonState.current_iteration_count
                        >= ComparisonState.max_iterations
                    ),
                    class_name="px-8 py-3 bg-green-600 text-white font-semibold rounded-xl shadow-md hover:bg-green-700 hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed w-full max-w-xs",
                ),
                rx.el.div(
                    rx.icon("check_check", class_name="mr-3 text-green-600"),
                    rx.el.span(
                        f"✓ Converged! Final similarity: {ComparisonState.last_similarity_score:.1f}%",
                        class_name="text-lg font-semibold text-green-700",
                    ),
                    class_name="flex items-center px-6 py-3 bg-green-50 rounded-xl border-2 border-green-200",
                ),
            ),
            class_name="flex flex-col items-center mt-4",
        ),
        None,
    )


def automated_controls() -> rx.Component:
    """Renders controls for the automated mode with enhanced feedback."""
    return rx.cond(
        ComparisonState.has_responses,
        rx.el.div(
            rx.cond(
                ComparisonState.automated_running,
                rx.el.div(
                    rx.icon("loader-circle", class_name="animate-spin mr-3 h-6 w-6"),
                    rx.el.div(
                        rx.el.span(
                            "Automated analysis in progress...",
                            class_name="font-semibold",
                        ),
                        rx.el.div(
                            rx.el.span("Iteration: "),
                            rx.el.span(
                                ComparisonState.current_iteration_count.to_string(),
                                class_name="font-bold",
                            ),
                            rx.el.span(" / "),
                            rx.el.span(
                                ComparisonState.max_iterations.to_string(),
                                class_name="font-bold",
                            ),
                            rx.el.span(" | Similarity: "),
                            rx.el.span(
                                ComparisonState.last_similarity_score.to_string() + "%",
                                class_name="font-bold text-indigo-600",
                            ),
                            class_name="text-sm text-gray-600 mt-1",
                        ),
                        class_name="flex flex-col",
                    ),
                    class_name="flex items-center text-lg text-gray-700",
                ),
                rx.cond(
                    ComparisonState.converged,
                    rx.el.div(
                        rx.icon(
                            "check_check", class_name="mr-3 text-green-600 h-6 w-6"
                        ),
                        rx.el.div(
                            rx.el.span(
                                "✓ Convergence reached!", class_name="font-bold"
                            ),
                            rx.el.div(
                                rx.el.span("Final similarity: "),
                                rx.el.span(
                                    ComparisonState.last_similarity_score.to_string()
                                    + "%",
                                    class_name="font-bold text-green-600",
                                ),
                                rx.el.span(" in "),
                                rx.el.span(
                                    ComparisonState.current_iteration_count.to_string(),
                                    class_name="font-bold",
                                ),
                                rx.el.span(" iterations"),
                                class_name="text-sm text-gray-600 mt-1",
                            ),
                            class_name="flex flex-col",
                        ),
                        class_name="flex items-center text-lg text-green-700",
                    ),
                    rx.cond(
                        ComparisonState.current_iteration_count
                        >= ComparisonState.max_iterations,
                        rx.el.div(
                            rx.icon(
                                "badge_alert", class_name="mr-3 text-yellow-600 h-6 w-6"
                            ),
                            rx.el.div(
                                rx.el.span(
                                    "Maximum iterations reached", class_name="font-bold"
                                ),
                                rx.el.div(
                                    rx.el.span("Final similarity: "),
                                    rx.el.span(
                                        ComparisonState.last_similarity_score.to_string()
                                        + "%",
                                        class_name="font-bold text-yellow-600",
                                    ),
                                    class_name="text-sm text-gray-600 mt-1",
                                ),
                                class_name="flex flex-col",
                            ),
                            class_name="flex items-center text-lg text-yellow-700",
                        ),
                        rx.el.div(
                            rx.icon("info", class_name="mr-3 text-indigo-600 h-6 w-6"),
                            rx.el.span(
                                "Automated analysis paused", class_name="font-semibold"
                            ),
                            class_name="flex items-center text-lg text-gray-700",
                        ),
                    ),
                ),
            ),
            rx.cond(
                ComparisonState.automated_running,
                rx.el.button(
                    rx.icon("circle-stop", class_name="mr-2"),
                    "Stop",
                    on_click=ComparisonState.stop_automated_cycle,
                    class_name="flex items-center px-6 py-2 bg-red-600 text-white font-semibold rounded-xl shadow-md hover:bg-red-700 transition-all",
                ),
                rx.el.button(
                    rx.icon("play", class_name="mr-2"),
                    "Run Automation",
                    on_click=ComparisonState.run_automated_cycle,
                    disabled=ComparisonState.converged
                    | (
                        ComparisonState.current_iteration_count
                        >= ComparisonState.max_iterations
                    ),
                    class_name="flex items-center px-6 py-2 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed",
                ),
            ),
            class_name="flex flex-col items-center justify-center gap-4 p-6 bg-white rounded-2xl shadow-lg border border-gray-200 w-full max-w-2xl mt-4",
        ),
        None,
    )


def response_card(
    model_name: str, content: rx.Var[str], logo_type: str, iteration: rx.Var[str]
) -> rx.Component:
    """A card for displaying a model's response with enhanced styling."""
    logo_src = f"https://api.dicebear.com/9.x/bottts-neutral/svg?seed={logo_type}"
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.el.img(src=logo_src, class_name="h-8 w-8"),
                rx.el.h3(
                    f"{model_name} Response",
                    class_name="text-lg font-semibold text-gray-800",
                ),
                class_name="flex items-center gap-3",
            ),
            rx.el.div(
                rx.icon("history", class_name="mr-1.5 h-4 w-4"),
                "Iteration ",
                iteration,
                class_name="flex items-center px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-bold rounded-full",
            ),
            class_name="flex items-center justify-between p-4 border-b border-gray-200",
        ),
        rx.el.div(
            rx.markdown(
                content,
                class_name="prose prose-sm max-w-none text-gray-700 font-medium leading-relaxed",
            ),
            class_name="p-6 h-full overflow-auto max-h-96",
        ),
        class_name="bg-white rounded-2xl shadow-md border border-gray-100 flex flex-col h-full hover:shadow-xl transition-shadow duration-300",
    )


def skeleton_card(model_name: str) -> rx.Component:
    """A loading skeleton card."""
    return rx.el.div(
        rx.el.div(
            rx.el.div(class_name="h-8 w-8 bg-gray-200 rounded-full"),
            rx.el.div(class_name="h-6 w-32 bg-gray-200 rounded-md"),
            class_name="flex items-center gap-3",
        ),
        rx.el.div(
            rx.el.div(class_name="h-4 w-24 bg-gray-200 rounded-md"),
            class_name="p-4 border-t border-gray-200",
        ),
        rx.el.div(
            rx.el.div(class_name="h-4 bg-gray-200 rounded w-5/6 mb-3"),
            rx.el.div(class_name="h-4 bg-gray-200 rounded w-full mb-3"),
            rx.el.div(class_name="h-4 bg-gray-200 rounded w-4/6 mb-3"),
            rx.el.div(class_name="h-4 bg-gray-200 rounded w-full mb-3"),
            rx.el.div(class_name="h-4 bg-gray-200 rounded w-3/4"),
            class_name="p-6",
        ),
        class_name="bg-white rounded-2xl shadow-md border border-gray-100 animate-pulse",
    )


def placeholder_card() -> rx.Component:
    """Card shown before any prompt is submitted."""
    return rx.el.div(
        rx.icon("sparkles", size=48, class_name="text-gray-400 mb-4"),
        rx.el.h3(
            "Awaiting Your Prompt",
            class_name="text-xl font-semibold text-gray-600 mb-2",
        ),
        rx.el.p(
            "Enter a prompt above to start the comparison.",
            class_name="text-gray-500 font-medium",
        ),
        class_name="flex flex-col items-center justify-center p-12 bg-gray-50 border-2 border-dashed border-gray-200 rounded-2xl text-center w-full",
    )