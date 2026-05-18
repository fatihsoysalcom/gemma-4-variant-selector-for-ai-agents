import math

# Define simulated Gemma 4 variants and their characteristics.
# These characteristics (model_size_gb, inference_speed_tokens_per_sec, capability_score)
# are simplified for demonstration purposes to illustrate trade-offs.
GEMMA_VARIANTS = {
    "Gemma-2B-IT": {
        "description": "2 Billion parameters, Instruction Tuned",
        "model_size_gb": 4,  # Approximate memory footprint for inference
        "inference_speed_tokens_per_sec": 150, # Tokens per second on a typical CPU/light GPU
        "capability_score": 7, # A score from 1-10, 10 being most capable
        "is_instruction_tuned": True
    },
    "Gemma-2B-Base": {
        "description": "2 Billion parameters, Base Model",
        "model_size_gb": 4,
        "inference_speed_tokens_per_sec": 160, # Base models can sometimes be slightly faster without IT overhead
        "capability_score": 6,
        "is_instruction_tuned": False
    },
    "Gemma-7B-IT": {
        "description": "7 Billion parameters, Instruction Tuned",
        "model_size_gb": 14,
        "inference_speed_tokens_per_sec": 50,
        "capability_score": 9,
        "is_instruction_tuned": True
    },
    "Gemma-7B-Base": {
        "description": "7 Billion parameters, Base Model",
        "model_size_gb": 14,
        "inference_speed_tokens_per_sec": 55,
        "capability_score": 8,
        "is_instruction_tuned": False
    }
}

# Define simulated MCP (Multi-Agent Communication Protocol) agent scenarios and their requirements.
# Each scenario specifies resource constraints and functional needs for an AI agent.
MCP_AGENT_SCENARIOS = {
    "Resource-constrained IoT Agent": {
        "max_memory_gb": 5,
        "min_inference_speed_tokens_per_sec": 100,
        "min_capability_score": 6,
        "requires_instruction_tuned": False, # Often fine-tuned for specific tasks, so base is fine
        "priority": "speed" # If multiple fit, prioritize speed
    },
    "Customer Support Chatbot": {
        "max_memory_gb": 15,
        "min_inference_speed_tokens_per_sec": 40,
        "min_capability_score": 8,
        "requires_instruction_tuned": True, # Essential for conversational agents
        "priority": "capability" # If multiple fit, prioritize capability
    },
    "Complex Data Analysis Agent": {
        "max_memory_gb": 20,
        "min_inference_speed_tokens_per_sec": 30,
        "min_capability_score": 9,
        "requires_instruction_tuned": False, # May be fine-tuned for analysis tasks
        "priority": "capability"
    },
    "Edge Device Assistant": {
        "max_memory_gb": 4.5,
        "min_inference_speed_tokens_per_sec": 120,
        "min_capability_score": 5,
        "requires_instruction_tuned": True,
        "priority": "speed"
    }
}

def recommend_gemma_variant(agent_name, requirements):
    """
    Recommends the most suitable Gemma 4 variant for a given MCP agent scenario
    based on its resource constraints and functional requirements.
    """
    print(f"\n--- Recommending for: {agent_name} ---")
    print(f"Requirements: Max Mem={requirements['max_memory_gb']}GB, Min Speed={requirements['min_inference_speed_tokens_per_sec']} t/s, Min Cap={requirements['min_capability_score']}, IT Required={requirements['requires_instruction_tuned']}")

    suitable_variants = []

    for variant_name, props in GEMMA_VARIANTS.items():
        # Check if the variant meets all hard requirements
        meets_memory = props["model_size_gb"] <= requirements["max_memory_gb"]
        meets_speed = props["inference_speed_tokens_per_sec"] >= requirements["min_inference_speed_tokens_per_sec"]
        meets_capability = props["capability_score"] >= requirements["min_capability_score"]
        # Instruction-tuned requirement: if required, model must be IT. If not required, any model is fine.
        meets_it_requirement = (props["is_instruction_tuned"] == requirements["requires_instruction_tuned"]) or \
                               (not requirements["requires_instruction_tuned"])

        if meets_memory and meets_speed and meets_capability and meets_it_requirement:
            suitable_variants.append((variant_name, props))
            print(f"  - {variant_name} is a SUITABLE candidate.")
        else:
            # Provide feedback on why a variant is not suitable
            reasons = []
            if not meets_memory: reasons.append(f"Memory ({props['model_size_gb']}GB > {requirements['max_memory_gb']}GB)")
            if not meets_speed: reasons.append(f"Speed ({props['inference_speed_tokens_per_sec']} t/s < {requirements['min_inference_speed_tokens_per_sec']} t/s)")
            if not meets_capability: reasons.append(f"Capability ({props['capability_score']} < {requirements['min_capability_score']})")
            if not meets_it_requirement: reasons.append("Instruction Tuned (required)")
            print(f"  - {variant_name} is NOT suitable: {' | '.join(reasons)}")

    if not suitable_variants:
        return "No suitable Gemma 4 variant found for these requirements."

    # If multiple variants are suitable, apply priority to select the best one
    if len(suitable_variants) > 1:
        if requirements["priority"] == "speed":
            # Prioritize the fastest model among suitable ones
            best_variant = max(suitable_variants, key=lambda x: x[1]["inference_speed_tokens_per_sec"])
            print(f"  Multiple suitable variants found. Prioritizing for SPEED.")
        elif requirements["priority"] == "capability":
            # Prioritize the most capable model among suitable ones
            best_variant = max(suitable_variants, key=lambda x: x[1]["capability_score"])
            print(f"  Multiple suitable variants found. Prioritizing for CAPABILITY.")
        else: # Default to smallest model size if no specific priority or unknown priority
            best_variant = min(suitable_variants, key=lambda x: x[1]["model_size_gb"])
            print(f"  Multiple suitable variants found. Defaulting to smallest model size.")
    else:
        best_variant = suitable_variants[0]

    return f"Recommended: {best_variant[0]} ({best_variant[1]['description']})\n" \
           f"  - Model Size: {best_variant[1]['model_size_gb']} GB\n" \
           f"  - Inference Speed: {best_variant[1]['inference_speed_tokens_per_sec']} tokens/sec\n" \
           f"  - Capability Score: {best_variant[1]['capability_score']}\n" \
           f"  - Instruction Tuned: {best_variant[1]['is_instruction_tuned']}"


if __name__ == "__main__":
    print("Simulating Gemma 4 variant selection for MCP Agents.")
    print("----------------------------------------------------")

    for agent_name, requirements in MCP_AGENT_SCENARIOS.items():
        recommendation = recommend_gemma_variant(agent_name, requirements)
        print(recommendation)
        print("-" * 60)
