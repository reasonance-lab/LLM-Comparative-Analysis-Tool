import reflex as rx
import os
import asyncio
import openai
import anthropic
from typing import Literal, TypedDict, cast
import logging
from collections import Counter
import math

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelResponse(TypedDict):
    iteration: int
    openai_response: str
    claude_response: str
    similarity: float


class ComparisonState(rx.State):
    """Manages the state for the LLM comparison application."""

    prompt: str = ""
    is_loading: bool = False
    is_iterating: bool = False
    automated_running: bool = False
    converged: bool = False
    mode: Literal["manual", "automated"] = "manual"
    history: list[ModelResponse] = []
    _openai_client: openai.OpenAI | None = None
    _anthropic_client: anthropic.Anthropic | None = None

    def _initialize_clients(self):
        """Initializes API clients if they don't exist."""
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

    @rx.var
    def has_responses(self) -> bool:
        """Check if there is any history to display."""
        return len(self.history) > 0

    @rx.var
    def last_similarity_score(self) -> float:
        """Returns the similarity score of the last iteration."""
        if not self.history:
            return 0.0
        return self.history[-1]["similarity"] * 100

    @rx.event
    def set_mode(self, new_mode: Literal["manual", "automated"]):
        """Set the evaluation mode."""
        self.mode = new_mode

    @rx.event(background=True)
    async def get_initial_responses(self, form_data: dict):
        """Fetches initial responses from both OpenAI and Claude."""
        async with self:
            self.prompt = form_data.get("prompt", "")
            if not self.prompt or not self.prompt.strip():
                yield rx.toast("Please enter a prompt.", duration=3000)
                return
            self.is_loading = True
            self.is_iterating = False
            self.automated_running = False
            self.converged = False
            self.history = []
            self._initialize_clients()
        openai_task = self._fetch_openai(self.prompt)
        claude_task = self._fetch_claude(self.prompt)
        try:
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
            similarity = self._cosine_similarity(openai_res, claude_res)
            async with self:
                initial_responses: ModelResponse = {
                    "iteration": 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                }
                self.history.append(initial_responses)
                if self.mode == "automated":
                    yield ComparisonState.run_automated_cycle
        except Exception as e:
            logging.exception(f"Error fetching initial responses: {e}")
            yield rx.toast(f"An API error occurred: {e}", duration=5000)
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
            new_prompt_template = f"Original User Prompt: {self.prompt}\n\nYour previous code:\n---\n{{your_previous_response}}\n---\n\nAlternative code approach:\n---\n{{other_model_response}}\n---\n\nInstructions:\n- Review both code solutions against the ORIGINAL USER PROMPT above\n- If the alternative approach has superior logic for solving the USER'S REQUIREMENT, adopt it\n- Eliminate all stylistic differences (comments, variable names, formatting)\n- Do NOT include any commentary about code quality or structure\n- Output ONLY the refined, complete Python code solution that best addresses the original prompt"
            openai_new_prompt = new_prompt_template.format(
                your_previous_response=openai_previous,
                other_model_response=claude_previous,
            )
            claude_new_prompt = new_prompt_template.format(
                your_previous_response=claude_previous,
                other_model_response=openai_previous,
            )
        openai_task = self._fetch_openai(openai_new_prompt)
        claude_task = self._fetch_claude(claude_new_prompt)
        try:
            openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
            async with self:
                similarity = self._cosine_similarity(openai_res, claude_res)
                new_iteration: ModelResponse = {
                    "iteration": len(self.history) + 1,
                    "openai_response": openai_res,
                    "claude_response": claude_res,
                    "similarity": similarity,
                }
                self.history.append(new_iteration)
        except Exception as e:
            logging.exception(f"Error during iteration: {e}")
            yield rx.toast(
                f"An API error occurred during iteration: {e}", duration=5000
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
        while True:
            async with self:
                if not self.automated_running:
                    break
                last_iteration = self.history[-1]
                current_iter_count = last_iteration["iteration"]
                similarity = last_iteration["similarity"]
                if similarity >= 0.9 or current_iter_count >= 10:
                    self.converged = True
                    self.automated_running = False
                    break
                self._initialize_clients()
                openai_previous = last_iteration["openai_response"]
                claude_previous = last_iteration["claude_response"]
                new_prompt_template = f"Original User Prompt: {self.prompt}\n\nYour previous code:\n---\n{{your_previous_response}}\n---\n\nAlternative code approach:\n---\n{{other_model_response}}\n---\n\nInstructions:\n- Review both code solutions against the ORIGINAL USER PROMPT above\n- If the alternative approach has superior logic for solving the USER'S REQUIREMENT, adopt it\n- Eliminate all stylistic differences (comments, variable names, formatting)\n- Do NOT include any commentary about code quality or structure\n- Output ONLY the refined, complete Python code solution that best addresses the original prompt"
                openai_new_prompt = new_prompt_template.format(
                    your_previous_response=openai_previous,
                    other_model_response=claude_previous,
                )
                claude_new_prompt = new_prompt_template.format(
                    your_previous_response=claude_previous,
                    other_model_response=openai_previous,
                )
            openai_task = self._fetch_openai(openai_new_prompt)
            claude_task = self._fetch_claude(claude_new_prompt)
            try:
                openai_res, claude_res = await asyncio.gather(openai_task, claude_task)
                new_similarity = self._cosine_similarity(openai_res, claude_res)
                async with self:
                    new_iteration: ModelResponse = {
                        "iteration": current_iter_count + 1,
                        "openai_response": openai_res,
                        "claude_response": claude_res,
                        "similarity": new_similarity,
                    }
                    self.history.append(new_iteration)
            except Exception as e:
                async with self:
                    self.automated_running = False
                logging.exception(f"Error during automated iteration: {e}")
                yield rx.toast(f"API error in automated cycle: {e}", duration=5000)
                break
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
            return stripped_text.endswith("")
        return True

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity, but only if both responses are complete."""
        if not self._is_response_complete(text1) or not self._is_response_complete(
            text2
        ):
            return 0.0
        vec1 = Counter(text1.split())
        vec2 = Counter(text2.split())
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum((vec1[x] * vec2[x] for x in intersection))
        sum1 = sum((vec1[x] ** 2 for x in vec1.keys()))
        sum2 = sum((vec2[x] ** 2 for x in vec2.keys()))
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    async def _fetch_openai(self, current_prompt: str) -> str:
        """Helper to fetch response from OpenAI."""
        client = cast(openai.OpenAI, self._openai_client)
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates Python code. Your goal is to collaborate with another AI to converge on a single, optimal solution. Focus on functional correctness and logical structure over stylistic differences.",
                    },
                    {"role": "user", "content": current_prompt},
                ],
                max_tokens=2048,
                temperature=0.7,
            )
            return response.choices[0].message.content or "No response from OpenAI."
        except Exception as e:
            logging.exception(f"OpenAI API error: {e}")
            return f"Error: Could not get response from OpenAI. Details: {e}"

    async def _fetch_claude(self, current_prompt: str) -> str:
        """Helper to fetch response from Anthropic Claude."""
        client = cast(anthropic.Anthropic, self._anthropic_client)
        try:
            message = await asyncio.to_thread(
                client.messages.create,
                model="claude-3-haiku-20240307",
                system="You are a helpful assistant that generates Python code. Your goal is to collaborate with another AI to converge on a single, optimal solution. Focus on functional correctness and logical structure over stylistic differences.",
                max_tokens=2048,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=0.7,
            )
            return message.content[0].text or "No response from Claude."
        except Exception as e:
            logging.exception(f"Anthropic API error: {e}")
            return f"Error: Could not get response from Claude. Details: {e}"