"""
Strategy 3: Constrained/Regularized Personalization

Problem: Per-persona LoRAs drift too far from unified model, losing generalization.
Solution: Add regularization to keep personalized adapters close to unified adapter.

Approaches:
1. L2 regularization on (LoRA_persona - LoRA_unified)
2. Knowledge distillation from unified model
3. Elastic weight consolidation (EWC)
4. Smaller rank/alpha for per-persona (r=2-4 instead of r=8)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import torch.nn as nn
import json
from pathlib import Path

class ConstrainedLoRATrainer(Trainer):
    """
    Custom trainer that adds regularization to keep personalized LoRA
    close to the unified LoRA adapter.
    """

    def __init__(self, unified_lora_path=None, l2_lambda=0.01, **kwargs):
        super().__init__(**kwargs)
        self.l2_lambda = l2_lambda

        # Load unified LoRA parameters as reference
        if unified_lora_path:
            print(f"Loading unified LoRA from {unified_lora_path} for regularization...")
            unified_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(kwargs['model'].peft_config['default'].base_model_name_or_path),
                unified_lora_path
            )

            self.unified_params = {
                name: param.clone().detach()
                for name, param in unified_model.named_parameters()
                if 'lora' in name.lower()
            }
        else:
            self.unified_params = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to add regularization term"""

        # Standard language modeling loss
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # Add L2 regularization to unified adapter
        if self.unified_params is not None and self.l2_lambda > 0:
            reg_loss = 0.0
            count = 0

            for name, param in model.named_parameters():
                if 'lora' in name.lower() and name in self.unified_params:
                    reg_loss += torch.nn.functional.mse_loss(param, self.unified_params[name])
                    count += 1

            if count > 0:
                reg_loss = reg_loss / count
                total_loss = lm_loss + self.l2_lambda * reg_loss
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train_constrained_persona_lora(
    persona_id,
    train_data,
    eval_data,
    unified_lora_path='models/lora_unified',
    output_dir=None,
    rank=4,  # Smaller rank to reduce overfitting
    l2_lambda=0.1,  # Regularization strength
):
    """
    Train personalized LoRA with constraints to stay close to unified model.

    Args:
        persona_id: ID of persona to personalize for
        train_data: Training dataset for this persona
        eval_data: Eval dataset for this persona
        unified_lora_path: Path to unified LoRA adapter
        output_dir: Where to save personalized adapter
        rank: LoRA rank (lower = less overfitting)
        l2_lambda: Regularization strength (higher = stay closer to unified)
    """

    if output_dir is None:
        output_dir = f'models/lora_constrained/{persona_id}'

    print(f"\nTraining constrained LoRA for {persona_id}")
    print(f"  Rank: {rank}")
    print(f"  L2 lambda: {l2_lambda}")
    print(f"  Training examples: {len(train_data)}")

    # Load base model
    base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load unified LoRA first
    print("Loading unified LoRA as initialization...")
    model = PeftModel.from_pretrained(base_model, unified_lora_path)

    # Add personalized LoRA on top (very small rank)
    persona_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,  # Keep alpha = 2*r
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,  # Higher dropout to reduce overfitting
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Note: This adds a NEW LoRA on top. We want to fine-tune the existing one.
    # Instead, we'll fine-tune the loaded unified LoRA with regularization.

    model.print_trainable_parameters()

    # Training arguments with early stopping
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Fewer epochs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,  # Lower learning rate
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        warmup_steps=20,
    )

    # Use custom trainer with regularization
    trainer = ConstrainedLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        unified_lora_path=unified_lora_path,
        l2_lambda=l2_lambda,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved constrained LoRA to {output_dir}")

    return model


def ablation_study_regularization():
    """
    Run ablation to find optimal regularization strength.

    Test different l2_lambda values: [0.0, 0.01, 0.05, 0.1, 0.5]
    And different ranks: [2, 4, 8]
    """

    print("Regularization Ablation Study")
    print("=" * 80)

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5]
    ranks = [2, 4, 8]

    results = []

    for l2_lambda in lambdas:
        for rank in ranks:
            print(f"\nTesting lambda={l2_lambda}, rank={rank}")

            # Train on subset of personas (e.g., 10 personas)
            # Evaluate on their test sets
            # Compare to unified baseline

            # TODO: Implement full training loop
            # For now, just outline the experiment

            results.append({
                'l2_lambda': l2_lambda,
                'rank': rank,
                'avg_improvement_over_unified': None,  # Calculate this
                'num_personas_improved': None,  # Calculate this
            })

    # Save results
    output_path = Path('results/regularization_ablation')
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ablation results to {output_path / 'results.json'}")


if __name__ == '__main__':
    print(__doc__)
    print("\nThis script implements constrained personalization.")
    print("Run ablation_study_regularization() to find optimal hyperparameters.")
