# GRPO Analysis Summary: `parity_grpo_curriculum_boundary`

Run directory: `C:\Users\ravir\CWRU\NLP\projects\Chomskey\formal_rl_length_generalization\runs\parity_grpo_curriculum_boundary`
Figures directory: `C:\Users\ravir\CWRU\NLP\projects\Chomskey\formal_rl_length_generalization\runs\parity_grpo_curriculum_boundary\analysis_outputs\figures`

## Final training rollout snapshot
- Final logged step: 500
- reward: 1.2156
- process_match_accuracy: 0.5725
- output_match_accuracy: 0.5000
- exact_match_accuracy: 0.0000
- generated_tokens_mean: 49.6250
- cot_tokens_mean: 47.6250

## Best OOD checkpoint selection
- Best `ood_41_80` terminal: 0.4750 at step 100
- Best `ood_41_80` process: 0.5603 at step 500
- Best `ood_41_80` exact: 0.0000 at step 100

## Final GRPO vs SFT comparison
- ood_41_80 / process: SFT=0.5261, GRPO final=0.5603, delta=+0.0341
- ood_41_80 / terminal: SFT=0.4750, GRPO final=0.3750, delta=-0.1000
- ood_41_80 / exact: SFT=0.0000, GRPO final=0.0000, delta=+0.0000
- train_1_40 / process: SFT=0.9683, GRPO final=0.9705, delta=+0.0021
- train_1_40 / terminal: SFT=0.8750, GRPO final=0.8250, delta=-0.0500
- train_1_40 / exact: SFT=0.7750, GRPO final=0.7250, delta=-0.0500

## Saved figures
- `01_grpo_rollout_reward_accuracy.png`
- `02_grpo_loss_entropy.png`
- `03_rollout_generated_and_cot_tokens.png`
- `04_curriculum_schedule.png`
- `05_eval_process_by_length_bucket.png`
- `05_eval_terminal_by_length_bucket.png`
- `05_eval_exact_by_length_bucket.png`
- `06_final_sft_vs_grpo_process.png`
- `06_final_sft_vs_grpo_terminal.png`
- `06_final_sft_vs_grpo_exact.png`
