"""
Main application entry point for LLM Cross-Talk Analyzer - Phase 2 Enhanced
"""
import reflex as rx
from app.state import ComparisonState
from app.components import (
    header,
    settings_panel,
    prompt_input_area,
    response_area,
)


def index() -> rx.Component:
    """The main page of the application with all Phase 2 enhancements."""
    return rx.el.main(
        header(),
        rx.el.div(
            settings_panel(),
            prompt_input_area(),
            response_area(),
            class_name="flex flex-col items-center w-full px-4 sm:px-6 lg:px-8 py-8 gap-8",
        ),
        class_name="font-['Raleway'] bg-gray-50 min-h-screen",
    )


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="indigo", radius="large"),
    head_components=[
        rx.el.link(rel="preconnect", href="https://fonts.googleapis.com"),
        rx.el.link(rel="preconnect", href="https://fonts.gstatic.com", cross_origin=""),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700&display=swap",
            rel="stylesheet",
        ),
    ],
)
app.add_page(index, title="LLM Cross-Talk Analyzer - Phase 2")
