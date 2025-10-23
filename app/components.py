import reflex as rx
from app.state import ComparisonState


def header() -> rx.Component:
    """Renders the application header."""
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
            class_name="flex items-center justify-between w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
        ),
        class_name="w-full py-4 bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-10",
    )


def prompt_input_area() -> rx.Component:
    """Renders the prompt input form."""
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
    """Displays the side-by-side responses or loading skeletons."""
    return rx.el.div(
        rx.cond(
            ComparisonState.is_loading,
            rx.el.div(
                skeleton_card("OpenAI"),
                skeleton_card("Claude"),
                class_name="grid md:grid-cols-2 gap-8 w-full",
            ),
            rx.cond(
                ComparisonState.has_responses,
                rx.el.div(
                    rx.foreach(
                        ComparisonState.history, lambda item: iteration_view(item)
                    ),
                    rx.cond(
                        ComparisonState.mode == "manual",
                        iterate_button_section(),
                        automated_controls(),
                    ),
                    class_name="flex flex-col items-center gap-10 w-full",
                ),
                placeholder_card(),
            ),
        ),
        class_name="w-full max-w-7xl mt-8",
    )


def iteration_view(item: rx.Var[dict]) -> rx.Component:
    """Displays a single iteration of responses."""
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
        similarity_badge(item["similarity"]),
        class_name="flex flex-col items-center gap-4 w-full",
    )


def similarity_badge(similarity: rx.Var[float]) -> rx.Component:
    """Displays the similarity score as a badge."""
    percentage = (similarity * 100).to_string() + "%"
    return rx.el.div(
        rx.icon("check-check", class_name="mr-2"),
        "Similarity: ",
        percentage,
        class_name=rx.cond(
            similarity >= 0.95,
            "flex items-center px-4 py-2 bg-green-100 text-green-800 font-bold rounded-full border-2 border-green-300 shadow-sm",
            "flex items-center px-3 py-1.5 bg-blue-100 text-blue-800 font-semibold rounded-full border border-blue-200",
        ),
    )


def iterate_button_section() -> rx.Component:
    """Renders the 'Iterate' button when in manual mode."""
    return rx.cond(
        (ComparisonState.mode == "manual") & ComparisonState.has_responses,
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
            disabled=ComparisonState.is_iterating,
            class_name="px-8 py-3 bg-green-600 text-white font-semibold rounded-xl shadow-md hover:bg-green-700 hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed w-full max-w-xs mt-4",
        ),
        None,
    )


def automated_controls() -> rx.Component:
    """Renders controls for the automated mode."""
    return rx.cond(
        ComparisonState.has_responses,
        rx.el.div(
            rx.cond(
                ComparisonState.automated_running,
                rx.el.div(
                    rx.icon("loader-circle", class_name="animate-spin mr-3"),
                    rx.el.span("Automated analysis in progress... Iteration "),
                    rx.el.span(ComparisonState.history.length()),
                    class_name="flex items-center text-lg font-semibold text-gray-700",
                ),
                rx.cond(
                    ComparisonState.converged,
                    rx.el.div(
                        rx.icon("square_check", class_name="mr-3 text-green-600"),
                        rx.el.span("Convergence reached! Final similarity: "),
                        rx.el.span(
                            ComparisonState.last_similarity_score.to_string() + "%",
                            class_name="font-bold",
                        ),
                        class_name="flex items-center text-lg font-semibold text-green-700",
                    ),
                    rx.el.div(
                        rx.icon("info", class_name="mr-3 text-indigo-600"),
                        rx.el.span("Automated analysis paused."),
                        class_name="flex items-center text-lg font-semibold text-gray-700",
                    ),
                ),
            ),
            rx.cond(
                ComparisonState.automated_running,
                rx.el.button(
                    rx.icon("circle_stop", class_name="mr-2"),
                    "Stop",
                    on_click=ComparisonState.stop_automated_cycle,
                    class_name="flex items-center px-6 py-2 bg-red-600 text-white font-semibold rounded-xl shadow-md hover:bg-red-700 transition-all",
                ),
                rx.el.button(
                    rx.icon("play", class_name="mr-2"),
                    "Run Automation",
                    on_click=ComparisonState.run_automated_cycle,
                    disabled=ComparisonState.converged,
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
    """A card for displaying a model's response."""
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
            class_name="p-6 h-full",
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