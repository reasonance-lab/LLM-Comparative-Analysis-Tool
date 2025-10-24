"""
State management for LLM Cross-Talk Analyzer - Phase 2 Enhanced Version
Includes: Export functionality, chart data, model selection, diff tracking
"""

import reflex as rx
import os
import asyncio
import openai
import anthropic
from typing import Literal, TypedDict, cast
import logging
from collections import Counter
import math
import json
from datetime import datetime
import difflib
import plotly.graph_objects as go
import base64
import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UploadedFile(TypedDict):
    """Type definition for an uploaded file."""

    name: str
    size: int
    media_type: str
    data: str


class ModelResponse(TypedDict):
    """Type definition for a single iteration response."""

    iteration: int
    openai_response: str
    claude_response: str
    similarity: float
    change_rate: float


class ComparisonState(rx.State):
    """Manages the state for the LLM comparison application with enhanced features."""

    prompt: str = ""
    is_loading: bool = False
    is_iterating: bool = False
    automated_running: bool = False
    converged: bool = False
    mode: Literal["manual", "automated"] = "manual"
    history: list[ModelResponse] = []
    convergence_threshold: float = 0.9
    max_iterations: int = 10
    temperature: float = 0.7
    openai_model: str = "gpt-5-codex"
    claude_model: str = "claude-sonnet-4-5-20250929"
    openai_models: list[str] = [
        "gpt-5-codex",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
    ]
    claude_models: list[str] = [
        "claude-sonnet-4-5-20250929",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    reasoning_effort: str = "medium"
    reasoning_efforts: list[str] = ["medium", "high"]
    extended_thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000
    thinking_budget_options: list[int] = [8000, 10000, 16000]
    max_tokens_claude: int = 64000
    show_settings: bool = False
    show_diff_viewer: bool = False
    selected_iteration_for_diff: int = 0
    chart_data_ready: bool = False
    openai_diff_html: str = ""
    claude_diff_html: str = ""
    uploaded_files: list[UploadedFile] = []
    _openai_client: openai.OpenAI | None = None
    _anthropic_client: anthropic.Anthropic | None = None

    def _initialize_clients(self):
        """Initializes API clients if they don't exist."""
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=httpx.Timeout(600.0, read=600.0, write=10.0, connect=5.0),
            )
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

    @rx.var
    def has_uploaded_files(self) -> bool:
        """Check if any files have been uploaded."""
        return len(self.uploaded_files) > 0

    @rx.var
    def has_responses(self) -> bool:
        """Check if there is any history to display."""
        return len(self.history) > 0

    @rx.var
    def last_similarity_score(self) -> float:
        """Returns the similarity score of the last iteration as percentage."""
        if not self.history:
            return 0.0
        return self.history[-1]["similarity"] * 100

    @rx.var
    def current_iteration_count(self) -> int:
        """Returns the current iteration count."""
        return len(self.history)

    @rx.var
    def convergence_status(self) -> str:
        """Returns a human-readable convergence status."""
        if not self.history:
            return "No iterations yet"
        last_similarity = self.history[-1]["similarity"]
        threshold_pct = self.convergence_threshold * 100
        if last_similarity >= self.convergence_threshold:
            return f"✓ Converged at {last_similarity * 100:.1f}% similarity"
        elif len(self.history) >= self.max_iterations:
            return f"Max iterations reached ({self.max_iterations})"
        else:
            gap = (self.convergence_threshold - last_similarity) * 100
            return f"Need {gap:.1f}% more to reach {threshold_pct:.0f}% threshold"

    @rx.var
    def chart_iterations(self) -> list[int]:
        """Returns list of iteration numbers for chart."""
        return [item["iteration"] for item in self.history]

    @rx.var
    def chart_similarities(self) -> list[float]:
        """Returns list of similarity scores for chart."""
        return [item["similarity"] * 100 for item in self.history]

    @rx.var
    def chart_change_rates(self) -> list[float]:
        """Returns list of change rates for chart."""
        return [item["change_rate"] * 100 for item in self.history]

    @rx.var
    def convergence_fig(self) -> go.Figure:
        """Generate the convergence progress chart figure."""
        fig = go.Figure()
        if self.chart_iterations:
            fig.add_trace(
                go.Scatter(
                    x=self.chart_iterations,
                    y=self.chart_similarities,
                    mode="lines+markers",
                    name="Similarity",
                    line={"color": "#6366f1", "width": 3},
                    marker={"size": 8, "color": "#6366f1"},
                    hovertemplate="Iteration %{x}<br>Similarity: %{y:.1f}%<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.chart_iterations,
                    y=[self.convergence_threshold * 100] * len(self.chart_iterations),
                    mode="lines",
                    name="Threshold",
                    line={"color": "#22c55e", "width": 2, "dash": "dash"},
                    hovertemplate="Threshold: %{y:.0f}%<extra></extra>",
                )
            )
        fig.update_layout(
            title=None,
            xaxis={"title": "Iteration", "showgrid": True, "gridcolor": "#f3f4f6"},
            yaxis={
                "title": "Similarity (%)",
                "showgrid": True,
                "gridcolor": "#f3f4f6",
                "range": [0, 100],
            },
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"family": "Raleway, sans-serif"},
            hovermode="x unified",
            showlegend=True,
            legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
            margin={"l": 60, "r": 40, "t": 20, "b": 60},
        )
        return fig

    @rx.var
    def openai_models_options(self) -> list[str]:
        return self.openai_models

    @rx.var
    def claude_models_options(self) -> list[str]:
        return self.claude_models

    @rx.var
    def reasoning_efforts_options(self) -> list[str]:
        return self.reasoning_efforts

    @rx.var
    def thinking_budget_options_str(self) -> list[str]:
        return [str(opt) for opt in self.thinking_budget_options]

    @rx.var
    def diff_iteration_options(self) -> list[str]:
        if self.current_iteration_count < 2:
            return []
        return [str(i) for i in range(2, self.current_iteration_count + 1)]

    @rx.event
    def set_mode(self, new_mode: Literal["manual", "automated"]):
        """Set the evaluation mode."""
        self.mode = new_mode

    @rx.event
    def toggle_settings(self):
        """Toggle settings panel visibility."""
        self.show_settings = not self.show_settings

    @rx.event
    def toggle_diff_viewer(self):
        """Toggle diff viewer visibility."""
        self.show_diff_viewer = not self.show_diff_viewer
        if self.show_diff_viewer and len(self.history) > 1:
            self.selected_iteration_for_diff = len(self.history)
            self._generate_diffs()

    @rx.event
    def update_convergence_threshold(self, value: str):
        """Update convergence threshold from slider."""
        try:
            self.convergence_threshold = float(value)
        except (ValueError, TypeError) as e:
            logging.exception(f"Error updating convergence threshold: {e}")

    @rx.event
    def update_max_iterations(self, value: str):
        """Update max iterations from slider."""
        try:
            self.max_iterations = int(value)
        except (ValueError, TypeError) as e:
            logging.exception(f"Error updating max iterations: {e}")

    @rx.event
    def update_temperature(self, value: str):
        """Update temperature from slider."""
        try:
            self.temperature = float(value)
        except (ValueError, TypeError) as e:
            logging.exception(f"Error updating temperature: {e}")

    @rx.event
    def set_openai_model(self, model: str):
        """Set OpenAI model selection."""
        self.openai_model = model

    @rx.event
    def set_claude_model(self, model: str):
        """Set Claude model selection."""
        self.claude_model = model

    @rx.event
    def set_reasoning_effort(self, effort: str):
        """Set reasoning effort for OpenAI Responses API."""
        self.reasoning_effort = effort

    @rx.event
    def toggle_extended_thinking(self):
        """Toggle Claude extended thinking on/off."""
        self.extended_thinking_enabled = not self.extended_thinking_enabled

    @rx.event
    def set_thinking_budget(self, budget: str):
        """Set thinking budget tokens for Claude extended thinking."""
        try:
            self.thinking_budget_tokens = int(budget)
        except (ValueError, TypeError) as e:
            logging.exception(f"Error setting thinking budget: {e}")

    @rx.event
    def update_max_tokens_claude(self, value: str):
        """Update Claude's max_tokens from slider."""
        try:
            self.max_tokens_claude = int(value)
        except (ValueError, TypeError) as e:
            logging.exception(f"Error updating Claude max tokens: {e}")

    @rx.event
    def select_iteration_for_diff(self, iteration: str):
        """Select an iteration to view its diff."""
        try:
            self.selected_iteration_for_diff = int(iteration)
            self._generate_diffs()
        except (ValueError, TypeError) as e:
            logging.exception(f"Error selecting iteration for diff: {e}")

    def _generate_diffs(self):
        """Generate HTML diffs for selected iteration."""
        if not self.history or self.selected_iteration_for_diff < 2:
            self.openai_diff_html = "No previous iteration to compare"
            self.claude_diff_html = "No previous iteration to compare"
            return
        idx = self.selected_iteration_for_diff - 1
        if idx >= len(self.history):
            return
        current = self.history[idx]
        previous = self.history[idx - 1]
        openai_diff = difflib.unified_diff(
            previous["openai_response"].splitlines(keepends=True),
            current["openai_response"].splitlines(keepends=True),
            lineterm="",
            n=3,
        )
        self.openai_diff_html = self._format_diff_as_html(openai_diff)
        claude_diff = difflib.unified_diff(
            previous["claude_response"].splitlines(keepends=True),
            current["claude_response"].splitlines(keepends=True),
            lineterm="",
            n=3,
        )
        self.claude_diff_html = self._format_diff_as_html(claude_diff)

    def _format_diff_as_html(self, diff) -> str:
        """Format unified diff as styled text."""
        lines = list(diff)
        if not lines:
            return "No changes detected"
        formatted = []
        for line in lines:
            if line.startswith("+") and (not line.startswith("+++")):
                formatted.append(
                    f"<span style='color: #22c55e; background: #f0fdf4; display: block; padding: 2px 8px;'>+ {line[1:]}</span>"
                )
            elif line.startswith("-") and (not line.startswith("---")):
                formatted.append(
                    f"<span style='color: #ef4444; background: #fef2f2; display: block; padding: 2px 8px;'>- {line[1:]}</span>"
                )
            elif line.startswith("@@"):
                formatted.append(
                    f"<span style='color: #6366f1; background: #eef2ff; display: block; padding: 2px 8px; font-weight: bold;'>{line}</span>"
                )
            else:
                formatted.append(
                    f"<span style='color: #6b7280; display: block; padding: 2px 8px;'>{line}</span>"
                )
        return "".join(formatted)

    @rx.event
    def export_json(self):
        """Export current session data as JSON."""
        if not self.history:
            return rx.window_alert("No data to export")
        export_data = {
            "metadata": {
                "prompt": self.prompt,
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "convergence_threshold": self.convergence_threshold,
                "max_iterations": self.max_iterations,
                "temperature": self.temperature,
                "openai_model": self.openai_model,
                "claude_model": self.claude_model,
                "reasoning_effort": self.reasoning_effort,
                "extended_thinking_enabled": self.extended_thinking_enabled,
                "thinking_budget_tokens": self.thinking_budget_tokens,
                "converged": self.converged,
                "final_similarity": self.last_similarity_score,
                "total_iterations": len(self.history),
            },
            "history": self.history,
        }
        json_str = json.dumps(export_data, indent=2)
        return rx.download(
            data=json_str,
            filename=f"llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

    @rx.event
    def export_markdown(self):
        """Export current session as formatted Markdown."""
        if not self.history:
            return rx.window_alert("No data to export")
        md_lines = [
            "# LLM Cross-Talk Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            f"- **Mode:** {self.mode.capitalize()}",
            f"- **Convergence Threshold:** {self.convergence_threshold * 100:.0f}%",
            f"- **Max Iterations:** {self.max_iterations}",
            f"- **Temperature:** {self.temperature}",
            f"- **OpenAI Model:** {self.openai_model}",
            f"- **Reasoning Effort:** {self.reasoning_effort}",
            f"- **Claude Model:** {self.claude_model}",
            f"- **Extended Thinking:** {('Enabled' if self.extended_thinking_enabled else 'Disabled')}",
            f"- **Thinking Budget:** {self.thinking_budget_tokens} tokens"
            if self.extended_thinking_enabled
            else "",
            "",
            "## Results",
            f"- **Converged:** {('Yes' if self.converged else 'No')}",
            f"- **Final Similarity:** {self.last_similarity_score:.1f}%",
            f"- **Total Iterations:** {len(self.history)}",
            "",
            "## Original Prompt",
            "",
            self.prompt,
            "",
            "",
            "## Iteration History",
            "",
        ]
        for item in self.history:
            md_lines.extend(
                [
                    f"### Iteration {item['iteration']}",
                    f"**Similarity:** {item['similarity'] * 100:.1f}% | **Change Rate:** {item['change_rate'] * 100:.2f}%",
                    "",
                    "#### OpenAI Response",
                    "",
                    item["openai_response"],
                    "",
                    "",
                    "#### Claude Response",
                    "",
                    item["claude_response"],
                    "",
                    "",
                    "---",
                    "",
                ]
            )
        md_content = """
""".join(md_lines)
        return rx.download(
            data=md_content,
            filename=f"llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

    @rx.event
    def clear_all(self):
        """Clear all history and reset state."""
        self.history = []
        self.converged = False
        self.automated_running = False
        self.is_iterating = False
        self.prompt = ""
        self.show_diff_viewer = False
        self.chart_data_ready = False
        self.uploaded_files = []

    @rx.event(background=True)
    async def get_initial_responses(self, form_data: dict):
        """Fetches initial responses from both OpenAI and Claude."""
        async with self:
            self.prompt = form_data.get("prompt", "")
            if not self.prompt or not self.prompt.strip():
                if not self.uploaded_files:
                    yield rx.toast.error(
                        "Please enter a prompt or upload a file.", duration=3000
                    )
                    return
            self.is_loading = True
            self.is_iterating = False
            self.automated_running = False
            self.converged = False
            self.history = []
            self.chart_data_ready = False
            self._initialize_clients()
        try:
            openai_task = self._fetch_openai(self.prompt, self.uploaded_files)
            claude_task = self._fetch_claude(self.prompt, self.uploaded_files)
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
            if openai_res is None or claude_res is None:
                async with self:
                    self.is_loading = False
                if openai_res is None:
                    yield rx.toast.error("OpenAI failed to respond.", duration=5000)
                if claude_res is None:
                    yield rx.toast.error("Claude failed to respond.", duration=5000)
                return
            similarity = self._cosine_similarity(openai_res, claude_res)
            async with self:
                initial_responses: ModelResponse = {
                    "iteration": 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                    "change_rate": 0.0,
                }
                self.history.append(initial_responses)
                self.chart_data_ready = True
                if self.mode == "automated":
                    yield ComparisonState.run_automated_cycle
        except Exception as e:
            logging.exception(f"Error fetching initial responses: {e}")
            yield rx.toast.error(f"An unexpected error occurred: {e}", duration=5000)
        finally:
            async with self:
                self.is_loading = False

    @rx.event(background=True)
    async def iterate_manual_mode(self):
        """Performs one round of cross-evaluation in manual mode."""
        async with self:
            if not self.history:
                return
            self.is_iterating = True
            self._initialize_clients()
            last_iteration = self.history[-1]
            openai_previous = last_iteration["openai_response"]
            claude_previous = last_iteration["claude_response"]
            new_prompt_template = """Original User Request:
{original_prompt}

Your Previous Response:
---
{your_previous_response}
---

Alternative Response from Another Model:
---
{other_model_response}
---

Instructions:
- Carefully review BOTH responses against the original user request
- Identify the strengths and weaknesses of each approach
- If the alternative response has superior reasoning, insights, or completeness, integrate those elements
- Eliminate purely stylistic differences (formatting, phrasing variations that don't affect meaning)
- Synthesize the best aspects of both responses into a refined answer
- DO NOT add meta-commentary about the comparison process
- Output ONLY your refined, complete response to the original user request"""
            openai_new_prompt = new_prompt_template.format(
                original_prompt=self.prompt,
                your_previous_response=openai_previous,
                other_model_response=claude_previous,
            )
            claude_new_prompt = new_prompt_template.format(
                original_prompt=self.prompt,
                your_previous_response=claude_previous,
                other_model_response=openai_previous,
            )
        try:
            openai_task = self._fetch_openai(openai_new_prompt, self.uploaded_files)
            claude_task = self._fetch_claude(claude_new_prompt, self.uploaded_files)
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
            async with self:
                if openai_res is None or claude_res is None:
                    self.is_iterating = False
                    if openai_res is None:
                        yield rx.toast.error(
                            "OpenAI failed to respond during iteration.", duration=5000
                        )
                    if claude_res is None:
                        yield rx.toast.error(
                            "Claude failed to respond during iteration.", duration=5000
                        )
                    return
                similarity = self._cosine_similarity(openai_res, claude_res)
                prev_similarity = last_iteration["similarity"]
                change_rate = abs(similarity - prev_similarity)
                new_iteration: ModelResponse = {
                    "iteration": len(self.history) + 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                    "change_rate": change_rate,
                }
                self.history.append(new_iteration)
                if similarity >= self.convergence_threshold:
                    self.converged = True
                    yield rx.toast.success(
                        f"Convergence reached! Similarity: {similarity * 100:.1f}%",
                        duration=5000,
                    )
        except Exception as e:
            logging.exception(f"Error during iteration: {e}")
            yield rx.toast.error(
                f"An unexpected error occurred during iteration: {e}", duration=5000
            )
        finally:
            async with self:
                self.is_iterating = False

    @rx.event(background=True)
    async def run_automated_cycle(self):
        """Runs the automated cross-evaluation until convergence or max iterations."""
        async with self:
            if not self.history or self.automated_running:
                return
            self.automated_running = True
            self.converged = False
        oscillation_count = 0
        while True:
            async with self:
                if not self.automated_running:
                    break
                last_iteration = self.history[-1]
                current_iter_count = last_iteration["iteration"]
                similarity = last_iteration["similarity"]
                if similarity >= self.convergence_threshold:
                    self.converged = True
                    self.automated_running = False
                    yield rx.toast.success(
                        f"✓ Converged at {similarity * 100:.1f}% similarity!",
                        duration=5000,
                    )
                    break
                if current_iter_count >= self.max_iterations:
                    self.converged = False
                    self.automated_running = False
                    yield rx.toast.warning(
                        f"Max iterations ({self.max_iterations}) reached. Final similarity: {similarity * 100:.1f}%",
                        duration=5000,
                    )
                    break
                if len(self.history) >= 3:
                    recent_changes = [h["change_rate"] for h in self.history[-3:]]
                    avg_change = sum(recent_changes) / len(recent_changes)
                    if avg_change < 0.01:
                        oscillation_count += 1
                        if oscillation_count >= 2:
                            self.automated_running = False
                            yield rx.toast.warning(
                                f"Responses stabilized at {similarity * 100:.1f}% similarity without reaching threshold.",
                                duration=5000,
                            )
                            break
                    else:
                        oscillation_count = 0
                self._initialize_clients()
                openai_previous = last_iteration["openai_response"]
                claude_previous = last_iteration["claude_response"]
                new_prompt_template = """Original User Request:
{original_prompt}

Your Previous Response:
---
{your_previous_response}
---

Alternative Response from Another Model:
---
{other_model_response}
---

Instructions:
- Carefully review BOTH responses against the original user request
- Identify the strengths and weaknesses of each approach
- If the alternative response has superior reasoning, insights, or completeness, integrate those elements
- Eliminate purely stylistic differences (formatting, phrasing variations that don't affect meaning)
- Synthesize the best aspects of both responses into a refined answer
- DO NOT add meta-commentary about the comparison process
- Output ONLY your refined, complete response to the original user request"""
                openai_new_prompt = new_prompt_template.format(
                    original_prompt=self.prompt,
                    your_previous_response=openai_previous,
                    other_model_response=claude_previous,
                )
                claude_new_prompt = new_prompt_template.format(
                    original_prompt=self.prompt,
                    your_previous_response=claude_previous,
                    other_model_response=openai_previous,
                )
            try:
                openai_task = self._fetch_openai(openai_new_prompt, self.uploaded_files)
                claude_task = self._fetch_claude(claude_new_prompt, self.uploaded_files)
                openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
                if openai_res is None or claude_res is None:
                    async with self:
                        self.automated_running = False
                    if openai_res is None:
                        yield rx.toast.error(
                            "OpenAI failed during automated cycle.", duration=5000
                        )
                    if claude_res is None:
                        yield rx.toast.error(
                            "Claude failed during automated cycle.", duration=5000
                        )
                    break
                new_similarity = self._cosine_similarity(openai_res, claude_res)
                prev_similarity = last_iteration["similarity"]
                change_rate = abs(new_similarity - prev_similarity)
                async with self:
                    new_iteration: ModelResponse = {
                        "iteration": current_iter_count + 1,
                        "openai_response": openai_res,
                        "claude_response": claude_res,
                        "similarity": new_similarity,
                        "change_rate": change_rate,
                    }
                    self.history.append(new_iteration)
            except Exception as e:
                async with self:
                    self.automated_running = False
                logging.exception(f"Error during automated iteration: {e}")
                yield rx.toast.error(
                    f"Unexpected error in automated cycle: {e}", duration=5000
                )
                break
            await asyncio.sleep(0.5)
            yield
        async with self:
            self.automated_running = False

    @rx.event
    def stop_automated_cycle(self):
        """Stops the automated evaluation cycle."""
        self.automated_running = False

    def _is_response_complete(self, response_text: str) -> bool:
        """Check if the AI response appears to be complete."""
        stripped_text = response_text.strip()
        if "" in stripped_text:
            return stripped_text.count("") % 2 == 0
        return True

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        if not self._is_response_complete(text1) or not self._is_response_complete(
            text2
        ):
            return 0.0
        if (
            not text1
            or not text2
            or text1.startswith("Error:")
            or text2.startswith("Error:")
        ):
            return 0.0
        vec1 = Counter(text1.lower().split())
        vec2 = Counter(text2.lower().split())
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum((vec1[x] * vec2[x] for x in intersection))
        sum1 = sum((vec1[x] ** 2 for x in vec1.keys()))
        sum2 = sum((vec2[x] ** 2 for x in vec2.keys()))
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle file uploads."""
        if not files:
            return
        for file in files:
            try:
                upload_data = await file.read()
                base64_data = base64.b64encode(upload_data).decode("utf-8")
                self.uploaded_files.append(
                    {
                        "name": file.name,
                        "size": len(upload_data),
                        "media_type": file.content_type,
                        "data": base64_data,
                    }
                )
            except Exception as e:
                logging.exception(f"Error reading file {file.name}: {e}")
                yield rx.toast.error(f"Error processing {file.name}")

    @rx.event
    def remove_file(self, filename: str):
        """Remove a file from the uploaded list."""
        self.uploaded_files = [f for f in self.uploaded_files if f["name"] != filename]

    @rx.event
    def clear_files(self):
        """Clear all uploaded files."""
        self.uploaded_files = []

    async def _fetch_openai(
        self, current_prompt: str, files: list[UploadedFile]
    ) -> str | None:
        """Helper to fetch response from OpenAI using Responses API."""
        client = cast(openai.OpenAI, self._openai_client)
        try:
            system_message = "You are a helpful assistant. Your goal is to collaborate with another AI to converge on a single, optimal response. Focus on substance and accuracy over stylistic differences."
            content_parts = []
            if current_prompt:
                content_parts.append({"type": "input_text", "text": current_prompt})
            for file in files:
                if file["media_type"].startswith("image/"):
                    content_parts.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{file['media_type']};base64,{file['data']}",
                        }
                    )
                elif file["media_type"] == "application/pdf":
                    content_parts.append(
                        {
                            "type": "input_file",
                            "filename": file["name"],
                            "file_data": f"data:application/pdf;base64,{file['data']}",
                        }
                    )
            response = await asyncio.to_thread(
                client.responses.create,
                model=self.openai_model,
                input=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content_parts},
                ],
                reasoning={"effort": self.reasoning_effort, "summary": "auto"},
                store=True,
            )
            response_text = ""
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if item.type == "message":
                        for content_block in item.content:
                            if content_block.type == "output_text":
                                response_text += content_block.text
            response_text = response_text.strip()
            if not response_text:
                logging.warning("OpenAI returned an empty response.")
                return None
            return response_text
        except Exception as e:
            logging.exception(f"OpenAI API error: {e}")
            return None

    async def _fetch_claude(
        self, current_prompt: str, files: list[UploadedFile]
    ) -> str | None:
        """Helper to fetch response from Anthropic Claude with optional extended thinking."""
        client = cast(anthropic.Anthropic, self._anthropic_client)
        try:
            return await asyncio.to_thread(
                self._execute_claude_stream, current_prompt, files, client
            )
        except Exception as e:
            logging.exception(f"Anthropic API error: {e}")
            return None

    def _execute_claude_stream(
        self,
        current_prompt: str,
        files: list[UploadedFile],
        client: anthropic.Anthropic,
    ) -> str | None:
        """Synchronously executes the Claude stream and collects the response."""
        system_message = "You are a helpful assistant. Your goal is to collaborate with another AI to converge on a single, optimal response. Focus on substance and accuracy over stylistic differences."
        content_parts = []
        for file in files:
            if file["media_type"].startswith("image/"):
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": file["media_type"],
                            "data": file["data"],
                        },
                    }
                )
            elif file["media_type"] == "application/pdf":
                content_parts.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": file["data"],
                        },
                    }
                )
        if current_prompt:
            content_parts.append({"type": "text", "text": current_prompt})
        api_params = {
            "model": self.claude_model,
            "system": [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "max_tokens": self.max_tokens_claude,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": self.temperature,
        }
        response_text = ""
        thinking_text = ""
        try:
            if self.extended_thinking_enabled:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
                with client.messages.stream(**api_params) as stream:
                    for event in stream:
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                        ):
                            response_text += event.delta.text
                        elif event.type == "thinking_delta":
                            if hasattr(event.delta, "thinking"):
                                thinking_text += event.delta.thinking
            else:
                with client.messages.stream(**api_params) as stream:
                    for text in stream.text_stream:
                        response_text += text
            final_response = response_text.strip()
            if not final_response:
                logging.warning("Claude returned an empty response.")
                return None
            return final_response
        except Exception as e:
            logging.exception(f"Error during Claude stream execution: {e}")
            return None