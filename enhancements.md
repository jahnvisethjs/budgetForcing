# Budget Forcing Enhancements for TAU-Bench

Based on our implementation analysis, here are **7 targeted enhancements** that directly address identified limitations and could improve accuracy/pass@k.

---

## 🎯 Enhancement 1: Adaptive `num_ignore` Based on Uncertainty

### Current Problem
Fixed `num_ignore=1` for all turns, even when model is confident or uncertain.

### Enhancement
Dynamically adjust forcing iterations based on model uncertainty.

### Implementation
```python
def get_adaptive_num_ignore(action_logprobs, base_num_ignore=1):
    """Increase forcing when model is uncertain."""
    # Get probability of chosen action
    action_prob = np.exp(action_logprobs)
    
    # High uncertainty (low prob) → more forcing
    if action_prob < 0.3:
        return base_num_ignore + 2  # 3 total
    elif action_prob < 0.6:
        return base_num_ignore + 1  # 2 total
    else:
        return base_num_ignore  # 1 (confident)

# In generate_next_step_with_budget_forcing
response = client.chat.completions.create(..., logprobs=True)
adaptive_num_ignore = get_adaptive_num_ignore(response.choices[0].logprobs)

for i in range(adaptive_num_ignore):
    # Budget forcing loop
```

### Expected Improvement
- **Efficiency:** Save 30-40% compute on confident decisions
- **Accuracy:** Spend more compute where needed most
- **Pass@k:** +5-10% by allocating compute optimally

---

## 🎯 Enhancement 2: Action-Specific Forcing

### Current Problem
Force reconsideration equally for all actions (find_user, get_order, exchange, respond).

### Enhancement
Apply different `num_ignore` values based on action consequence.

### Implementation
```python
CONSEQUENCE_LEVELS = {
    # High-consequence actions (modify database)
    "exchange_delivered_order_items": 3,
    "cancel_pending_order": 3,
    "modify_pending_order_items": 3,
    
    # Medium-consequence (queries)
    "get_order_details": 1,
    "get_product_details": 1,
    
    # Low-consequence (info gathering)
    "find_user_id_by_name_zip": 1,
    "respond": 2,  # Communication is important
}

def get_consequence_based_num_ignore(action_name):
    return CONSEQUENCE_LEVELS.get(action_name, 1)

# After Phase 1
first_action_name = extract_action_name(content)
num_ignore_adaptive = get_consequence_based_num_ignore(first_action_name)
```

### Expected Improvement
- **Safety:** Prevent hasty high-consequence actions
- **Accuracy:** Force deeper thinking before database modifications
- **Pass@k:** +8-12% by reducing critical errors

---

## 🎯 Enhancement 3: Multi-Sample Budget Forcing

### Current Problem
Single trajectory with budget forcing; if it reconsiders wrongly, we fail.

### Enhancement
Generate K trajectories with budget forcing, vote on best.

### Implementation
```python
def multi_sample_budget_forcing(env, K=3, num_ignore=1):
    """Generate K budget-forced trajectories and vote."""
    trajectories = []
    
    for i in range(K):
        # Use different temperatures for diversity
        temp = 0.6 + (i * 0.1)  # [0.6, 0.7, 0.8]
        
        agent = ChatBudgetAgent(
            temperature=temp,
            num_ignore=num_ignore
        )
        
        traj = agent.solve(env)
        trajectories.append(traj)
    
    # Vote on action sequences
    action_seqs = [tuple(t.actions) for t in trajectories]
    from collections import Counter
    most_common = Counter(action_seqs).most_common(1)[0][0]
    
    # Return trajectory with most common action sequence
    for traj in trajectories:
        if tuple(traj.actions) == most_common:
            return traj
```

### Expected Improvement
- **Robustness:** Avoid single-path failures
- **Accuracy:** Voting filters out reconsideration errors
- **Pass@k:** +15-20% (combines budget forcing + self-consistency)

### Cost
3× inference (manageable for research)

---

## 🎯 Enhancement 4: Hierarchical Budget Forcing

### Current Problem
Apply forcing at single level (action selection).

### Enhancement
Force at multiple granularities: thought → tool → arguments.

### Implementation
```python
# Phase 1: Force thought generation
content = generate_initial()
content += "Wait,\n"
thought_extended = generate_continuation()

# Phase 2: Force tool selection
content = thought_extended + "Action:\n"
content += "Wait,\n"
tool_reconsidered = generate_continuation()

# Phase 3: Force argument selection
content = tool_reconsidered + '{"name": "...", "arguments": '
content += "Wait,\n"
args_refined = generate_continuation()

final_content = thought_extended + tool_reconsidered + args_refined
```

### Expected Improvement
- **Granularity:** Force reconsideration at each decision point
- **Accuracy:** Fewer argument errors (common failure mode)
- **Pass@k:** +10-15%

### Cost
3× forcing iterations per turn (more expensive but targeted)

---

## 🎯 Enhancement 5: Contrastive Budget Forcing

### Current Problem
Budget forcing uses same model for initial + forced generation.

### Enhancement
Use smaller "amateur" model for initial, larger "expert" for forcing.

### Implementation
```python
# Phase 1: Amateur model (fast, cheap)
amateur_model = "Qwen/Qwen2.5-3B-Instruct"
initial_response = client.chat.completions.create(
    model=amateur_model,
    messages=messages,
    max_tokens=actual_max_tokens
)
content_amateur = initial_response.choices[0].message.content

# Phase 2: Expert corrects amateur
expert_model = "Qwen/Qwen2.5-7B-Instruct"
messages_with_wait = messages + [
    {"role": "assistant", "content": content_amateur + "Wait,\n"}
]

expert_response = client.chat.completions.create(
    model=expert_model,
    messages=messages_with_wait,
    max_tokens=remaining_budget
)
content_expert = content_amateur + "Wait,\n" + expert_response.choices[0].message.content
```

### Expected Improvement
- **Efficiency:** Amateur handles easy cases
- **Accuracy:** Expert corrects amateur mistakes
- **Pass@k:** +12-18% (expert wisdom with amateur speed)

### Cost
1× amateur + 1× expert ≈ 1.5× single expert cost

---

## 🎯 Enhancement 6: Verifier-Guided Budget Forcing

### Current Problem
No signal whether "Wait" improved the action or made it worse.

### Enhancement
Train a verifier to score actions; only accept forced action if score improves.

### Implementation
```python
# Train verifier on successful/failed trajectories
verifier = train_action_scorer(
    successful_actions=get_actions(reward > 0.8),
    failed_actions=get_actions(reward < 0.2)
)

# Phase 1: Initial action
initial_action = parse_action(initial_content)
initial_score = verifier.score(messages, initial_action)

# Phase 2: Budget forcing
forced_action = parse_action(forced_content)
forced_score = verifier.score(messages, forced_action)

# Phase 3: Accept better action
if forced_score > initial_score:
    final_action = forced_action
else:
    final_action = initial_action  # Revert to initial
```

### Expected Improvement
- **Quality control:** Only keep beneficial reconsiderations
- **Accuracy:** Avoid "thinking yourself into wrong answer"
- **Pass@k:** +20-25% (with good verifier)

### Cost
Training verifier once + inference per action

---

## 🎯 Enhancement 7: Iterative Refinement with Critique

### Current Problem
"Wait" doesn't explain what to reconsider.

### Enhancement
Generate explicit critique, then refine based on it.

### Implementation
```python
# Phase 1: Initial action
initial_content = generate_initial()
initial_action = parse_action(initial_content)

# Phase 2: Critique
critique_prompt = f"""
Review this action:
{initial_action}

Given the conversation context, identify:
1. What information is missing?
2. Is this the right tool?
3. Are the arguments correct?
4. What could go wrong?
"""

critique = generate_critique(messages + [critique_prompt])

# Phase 3: Refine based on critique
refinement_prompt = f"""
{initial_content}

Self-critique: {critique}

Based on this critique, revise your action:
"""

refined_content = generate_refined(messages + [refinement_prompt])
```

### Expected Improvement
- **Explainability:** See what model reconsidered
- **Accuracy:** Targeted improvement vs blind "Wait"
- **Pass@k:** +18-25% (structured reconsideration)

### Cost
2× inference (critique + refinement)

---

## 🔬 Recommended Testing Order

### Phase 1: Low-Hanging Fruit (1-2 weeks)
1. **Enhancement 2** (Action-specific forcing) - Easy to implement
2. **Enhancement 1** (Adaptive num_ignore) - Requires logprobs extraction
3. **Enhancement 7** (Critique-based refinement) - Most explainable

**Expected combined gain:** +20-30%

### Phase 2: Scaling Techniques (2-3 weeks)
4. **Enhancement 3** (Multi-sample) - Proven approach
5. **Enhancement 5** (Contrastive) - Requires loading two models

**Expected combined gain:** +30-40%

### Phase 3: Advanced (1-2 months)
6. **Enhancement 4** (Hierarchical) - Complex logic
7. **Enhancement 6** (Verifier-guided) - Requires training verifier

**Expected combined gain:** +40-50%

---

## 🎯 Quick Win: Start Here

**Implement Enhancement 2 (Action-Specific Forcing) TODAY:**

```python
# Add to chat_budget_agent.py
CONSEQUENCE_LEVELS = {
    "exchange_delivered_order_items": 3,
    "cancel_pending_order": 3,
    "modify_pending_order_items": 3,
    "respond": 2,
    # ... rest default to 1
}

# In generate_next_step_with_budget_forcing, line ~119
try:
    first_action_parsed = json.loads(first_action_str)
    first_action_name = first_action_parsed.get("name", "unknown")
    adaptive_num_ignore = CONSEQUENCE_LEVELS.get(first_action_name, self.num_ignore)
except:
    adaptive_num_ignore = self.num_ignore

# Use adaptive_num_ignore in the forcing loop instead of self.num_ignore
for i in range(adaptive_num_ignore):
    ...
```

**Run on tasks 0-10 and measure improvement!**

---

## 💡 Combination Strategy

The most powerful approach is to **combine multiple enhancements**:

### Recommended Combo: "Smart Budget Forcing"
1. **Enhancement 1** (Adaptive num_ignore based on uncertainty)
2. **Enhancement 2** (Action-specific forcing)
3. **Enhancement 7** (Critique-based refinement)

**Synergy:** Adaptive allocation + targeted forcing + explainable refinement

**Expected improvement:** +35-45% pass@k

**Implementation time:** 2-3 weeks

---

## 📊 Expected Results Summary

| Enhancement | Difficulty | Time | Δ Pass@k | Cost Increase |
|-------------|-----------|------|----------|---------------|
| 1. Adaptive num_ignore | Medium | 3 days | +5-10% | 0.6× (saves!) |
| 2. Action-specific | Easy | 1 day | +8-12% | 1× |
| 3. Multi-sample | Easy | 2 days | +15-20% | 3× |
| 4. Hierarchical | Hard | 1 week | +10-15% | 3× |
| 5. Contrastive | Medium | 4 days | +12-18% | 1.5× |
| 6. Verifier-guided | Hard | 3 weeks | +20-25% | 1.2× |
| 7. Critique-based | Medium | 3 days | +18-25% | 2× |

---

## 🚀 The Path Forward

Your budget forcing foundation is solid. These enhancements address the core limitation: **unconditional forcing wastes compute and doesn't guarantee improvement**.

By making forcing:
- **Adaptive** (Enhancement 1)
- **Targeted** (Enhancement 2)
- **Verified** (Enhancement 6)
- **Explainable** (Enhancement 7)

You can transform budget forcing from a blunt instrument into a **precision tool** that dramatically improves TAU-Bench performance!

**Start with Enhancement 2 today and measure results!** 🎯
