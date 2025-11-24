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
        
        Implements S1-style budget forcing:
        1. Initial thinking: Generate with available token budget
        2. Budget forcing: Append "Wait" and force continuation num_ignore times
        3. Parse action from final output
        
        Args:
            messages: Conversation history
            
        Returns:
            (message_dict, action, cost) tuple
        """
        # Calculate available tokens to prevent context overflow
        # Model max: 32768, reserve some for the response
        model_max_context = 32768
        
        # Estimate input tokens (rough approximation: 1 token ≈ 4 chars)
        input_text = ""
        for msg in messages:
            input_text += msg.get("content", "")
        estimated_input_tokens = len(input_text) // 4
        
        # Calculate actual available tokens
        available_tokens = model_max_context - estimated_input_tokens - 500  # Reserve 500 for safety
        actual_max_tokens = min(self.max_tokens_thinking, max(100, available_tokens))
        
        # PHASE 1: Initial thinking
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=actual_max_tokens,
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.completion_tokens
        finish_reason = response.choices[0].finish_reason
        
        # Track first action for comparison
        try:
            first_action_str = content.split("Action:")[-1].strip()
            first_action_parsed = json.loads(first_action_str)
            first_action_name = first_action_parsed.get("name", "unknown")
        except:
            first_action_name = "unknown"
        
        print(f"\n{'='*60}")
        print(f"[BUDGET FORCING] Phase 1 Complete")
        print(f"  Tokens used: {tokens_used}/{actual_max_tokens}")
        print(f"  First action: {first_action_name}")
        print(f"{'='*60}")
        
        # PHASE 2: Budget forcing (S1 style)
        # S1 approach: Always force continuation num_ignore times
        # This gives model more thinking time before finalizing action
        remaining_budget = actual_max_tokens - tokens_used
        
        # Always force budget forcing if we have budget remaining
        # S1 doesn't try to detect "should we force" - it just forces num_ignore times
        if remaining_budget > 10:
            print(f"\n[BUDGET FORCING] Phase 2: Forcing reconsideration")
            print(f"  Remaining budget: {remaining_budget} tokens")
            print(f"  num_ignore iterations: {self.num_ignore}")
            
            for i in range(self.num_ignore):
                if remaining_budget <= 10:
                    break
                
                print(f"\n  --- Forcing iteration {i+1}/{self.num_ignore} ---")
                print(f"  Appending 'Wait,\\n' to prompt...")
                
                # S1 approach: append "Wait" to the assistant's message
                # This forces model to reconsider its reasoning
                messages_with_wait = messages + [
                    {"role": "assistant", "content": content + "Wait,\n"}
                ]
                
                # Continue generation
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_with_wait,
                    temperature=self.temperature,
                    max_tokens=remaining_budget,
                )
                
                # Append new content
                new_content = response.choices[0].message.content
                content = content + "Wait,\n" + new_content
                
                print(f"  New content generated: {len(new_content)} chars")
                print(f"  Total content now: {len(content)} chars")
                
                # Update budget tracking
                tokens_used += response.usage.completion_tokens
                remaining_budget = actual_max_tokens - tokens_used
                print(f"  Tokens used this iteration: {response.usage.completion_tokens}")
                print(f"  Total tokens: {tokens_used}/{actual_max_tokens}")
                print(f"  Remaining: {remaining_budget}")
                finish_reason = response.choices[0].finish_reason
                
                # Stop if no more budget
                if remaining_budget <= 10:
                    break
        
        # Track final action for comparison
        try:
            final_action_str = content.split("Action:")[-1].strip()
            final_action_parsed = json.loads(final_action_str)
            final_action_name = final_action_parsed.get("name", "unknown")
        except:
            final_action_name = "unknown"
        
        # Compare and log result
        if remaining_budget > 10 and self.num_ignore > 0:
            print(f"\n[BUDGET FORCING] Phase 2 Complete")
            if first_action_name != final_action_name:
                print(f"  ✅ ACTION CHANGED: {first_action_name} → {final_action_name}")
                print(f"  🎯 Model reconsidered and changed its mind!")
            else:
                print(f"  ⚠️  Same action: {first_action_name}")
                print(f"  Model stuck with original decision")
        else:
            print(f"\n[BUDGET FORCING] Phase 2 Skipped (insufficient budget)")
        
        print(f"{'='*60}\n")
        
        # PHASE 3: Parse action
        # Extract action from content (same logic as ChatReActAgent)
        action_str = content.split("Action:")[-1].strip()
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
        
        # Create message dict
        message = {
            "role": "assistant",
            "content": content,
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
