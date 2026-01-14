import os
import re
import requests
from bs4 import BeautifulSoup
from textwrap import dedent

# Agent Framework
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Evaluation Framework
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM

# --- TOOL DEFINITION ---
def search_duckduckgo(query: str, max_results: int = 5) -> list:
    """Search tool for finding scientist names."""
    url = "https://html.duckduckgo.com/html/"
    params = {'q': query, 'kl': 'us-en'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.post(url, data=params, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result')[:max_results]:
            title = result.find('a', class_='result__a')
            if title:
                results.append(title.get_text())
        return results
    except Exception:
        return []

# --- GENERATOR AGENT (Llama 3.2 via OpenRouter) ---
def get_generator_agent(api_key: str):
    """Returns the Agno Agent using Llama 3.2 via OpenRouter."""
    return Agent(
        name="Scientist Name Finder",
        model=OpenAIChat(
            id="meta-llama/llama-3.2-3b-instruct",  # Targeted Llama 3.2 model
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        ),
        tools=[search_duckduckgo],
        instructions=dedent("""\
            You find the names of top scientists in a given research field.
            IMPORTANT: Do not list the authors of the paper as the top scientist.

            For any research field provided, search and identify the 3 most influential scientists.
            Focus on:
            - Foundational contributors
            - Highly cited researchers
            - Award winners in the field

            Output ONLY a simple numbered list with just the names:
            1. Full Name One
            2. Full Name Two
            3. Full Name Three

            Do NOT add any other text, explanations, or details.
        """),
        markdown=False,
    )

# --- JUDGE CLASS (DeepSeek via OpenRouter) ---
class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(self, api_key):
        self.model_id = "deepseek/deepseek-chat"
        self.agent = Agent(
            model=OpenAIChat(
                id=self.model_id,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            ),
            description="You are a strict AI Judge. Output ONLY valid JSON.",
            markdown=False
        )

    def load_model(self):
        return self.agent.model

    def generate(self, prompt: str) -> str:
        # Force strict JSON output
        strict_prompt = prompt + "\n\nIMPORTANT: Return ONLY the JSON object. Do not say 'Here is the JSON'. Just the JSON."
        try:
            response = self.agent.run(strict_prompt)
            # Extract JSON cleanly
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            return match.group(0) if match else response.content
        except Exception as e:
            return str(e)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_id

# --- EVALUATION FUNCTION ---
def evaluate_relevance(api_key: str, topic: str, actual_output: str):
    """
    Runs the GEval metric using DeepSeek to check if names are valid.
    Returns (score, reason).
    """
    # 1. Initialize Judge
    judge_llm = DeepSeekJudge(api_key=api_key)

    # 2. Define Metric
    relevance_metric = GEval(
        name="Researcher Relevance",
        criteria=f"""
        1. Check if the names listed in the output are valid, real-world scientists associated with the field '{topic}'.
        2. If the names belong to celebrities, athletes (like Cricketers), or fictional characters, the score must be 0.
        3. The output must be a numbered list.
        """,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        model=judge_llm
    )

    # 3. Create Test Case
    test_case = LLMTestCase(
        input=f"Find top 3 researchers in {topic}",
        actual_output=actual_output
    )

    # 4. Measure
    relevance_metric.measure(test_case)
    
    return relevance_metric.score, relevance_metric.reason
