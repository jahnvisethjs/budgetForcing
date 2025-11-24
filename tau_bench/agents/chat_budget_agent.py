# Copyright Sierra

import json
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


class ChatBudgetForcingAgent(Agent):
    """
    ReAct-style agent with S1-inspired budget forcing.
    
    Uses vLLM via OpenAI-compatible API with token-based budget forcing:
    1. Initial thinking: Generate tool selection reasoning
    2. Budget forcing: Append "Wait" to force reconsideration
    3. Action parsing: Extract final tool call
    """
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        vllm_base_url: str = "http://localhost:8005/v1",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_tokens_thinking: int = 8000,
        num_ignore: int = 1,
        use_reasoning: bool = True,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize budget forcing agent.
        
        Args:
            tools_info: List of available tool definitions
            wiki: Domain knowledge/policy text
            vllm_base_url: vLLM server URL (OpenAI-compatible endpoint)
            model_name: Model name/path
            max_tokens_thinking: Maximum tokens for thinking phase
            num_ignore: Number of times to ignore end-of-thinking and append "Wait"
            use_reasoning: If True, use ReAct format; if False, use Act format
            temperature: Sampling temperature
        """
        # Reuse ReAct/Act instruction from chat_react_agent
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        
        # vLLM via OpenAI-compatible API
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",  # vLLM doesn't need a real key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        
        # Budget forcing parameters
        self.max_tokens_thinking = max_tokens_thinking
        self.num_ignore = num_ignore
        
        self.tools_info = tools_info

    def generate_next_step_with_budget_forcing(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """
        Generate next step with budget forcing.
        
        Implements 3-phase generation (S1 style):
        1. Initial thinking: Generate until model wants to stop (at "Action:")
        2. Budget forcing: If stopped at "Action:", append "Wait" and continue
        3. Parse action from final output
        
        Detection logic (matching S1's approach):
        - S1: Model hits stop token `<|im_start|><|im_end|>` → add "Wait"
        - TAU-Bench: Model outputs "Action:" → add "Wait"
        
        This forces the model to reconsider its tool choice before finalizing.
        
        Args:
            messages: Conversation history
            
        Returns:
            (message_dict, action, cost) tuple
        """
        # Convert messages to single prompt string for raw generation
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += content + "\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "  # Start assistant response
        
        # PHASE 1: Initial thinking
        # Stop when model outputs "Action:" (wants to take action)
        # This is analogous to S1's stop token for end-of-thinking
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens_thinking,
            stop=["Action:"],  # Stop when model wants to act
        )
        
        content = response.choices[0].text
        tokens_used = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason
        
        # PHASE 2: Budget forcing (S1 style)
        # If model stopped at "Action:" (wants to act), force it to reconsider
        # This matches S1's logic: when model hits stop token, add "Wait" if budget allows
        remaining_budget = self.max_tokens_thinking - tokens_used
        
        # Only force if model actually wanted to stop (not if it hit max_tokens)
        if finish_reason == "stop" and remaining_budget > 10:
            for i in range(self.num_ignore):
                if remaining_budget <= 10:
                    break
                
                # S1 approach: append output + "Wait" to prompt, then continue
                # Important: We append "Wait" BEFORE "Action:" to force reconsideration
                prompt_with_wait = prompt + content + "Wait"
                
                # Continue generation from extended prompt
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt_with_wait,
                    temperature=self.temperature,
                    max_tokens=remaining_budget,
                    stop=["Action:"],  # Still stop at "Action:"
                )
                
                # Append new content
                new_content = response.choices[0].text
                content = content + "Wait" + new_content
                
                # Update budget tracking
                tokens_used += response.usage.completion_tokens
                remaining_budget = self.max_tokens_thinking - tokens_used
                finish_reason = response.choices[0].finish_reason
                
                # If model hit max_tokens or no more budget, stop forcing
                if finish_reason != "stop" or remaining_budget <= 10:
                    break
        
        # PHASE 3: Parse action
        # At this point, content should end right before "Action:"
        # We need to let it complete the action
        if not content.strip().endswith("Action:"):
            # Complete the generation to get the full action
            final_prompt = prompt + content + "Action:"
            response = self.client.completions.create(
                model=self.model_name,
                prompt=final_prompt,
                temperature=self.temperature,
                max_tokens=500,  # Enough for action JSON
            )
            action_content = "Action:" + response.choices[0].text
        else:
            action_content = content
        
        # Extract action from content
        action_str = action_content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # Fallback: treat as respond action
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        
        # Create message dict with the full content (including final action)
        full_content = content if content.strip().endswith("Action:") else content + "\n" + action_content
        message = {
            "role": "assistant",
            "content": full_content,
        }
        
        # Approximate cost (vLLM is free, but keep for compatibility)
        cost = 0.0
        
        return message, action, cost

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve task using budget forcing agent.
        
        Args:
            env: Environment instance
            task_index: Task ID to solve
            max_num_steps: Maximum number of steps
            
        Returns:
            SolveResult with trajectory and reward
        """
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step_with_budget_forcing(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            
            if response.done:
                break
        
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


# Reuse instruction from chat_react_agent.py
REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
