# Local Qwen backend validation

Validated on 2026-10-12. NutriCoach's provider menu offers OpenRouter, Groq and
local Qwen for chat and food photos. OpenRouter remains the default.
Hosted providers use separate credentials; local inference needs no API key.

Local model: `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Generation is greedy, with reasoning disabled by default and images bounded
to 1024 pixels before processor resizing. Local chat decodes the author's tool
format and uses the existing NutriCoach tools.

| Validation | Job / environment | Result |
|---|---|---|
| Current NutriCoach suite after geometry removal, including Streamlit checks; live hosted/GPU integration modules excluded | VM | 171 passed, 7 skipped; 50 focused tests also passed |
| Full NutriCoach suite, including real Qwen image, geometry and chat tool inference | H100 job `964118`, node `jzxh044` | 192 passed, 12 skipped; exit `0:0`; job elapsed 2:07 |
| Provider selection, credentials, tool parsing and precision switching | VM provider tests | 12 passed |
| Real 4-bit image inference under a 32 GiB CUDA allocation ceiling | H100 job `963878`, node `jzxh044` | 1 passed; exit `0:0`; job elapsed 5:41 |

The full-suite skips concern absent trained recipe checkpoints, unavailable
model registry resources, live hosted tests without credentials and the separate
opt-in 4-bit test. Groq requests were verified using mocked clients; no real
Groq evaluation was performed without a dedicated key.
The H100 suite rows describe the earlier checkout before geometry removal;
real GPU inference was not rerun for the deletion.

The 4-bit test measured **18.52 GiB peak reserved GPU memory** and
**52.12 GiB peak host RSS**. Its 305.6-second analyzer elapsed time includes
model initialization, so it is not steady-state latency. These measurements
validate the CUDA limit on H100; they do not validate a 32 GB system-RAM machine.
The model used the correct, slower PyTorch linear-attention fallback because
the optimized flash-linear-attention package is not installed.

The tested VM environment uses PyTorch `2.6.0+cu124`, Transformers `5.17.0`,
Tokenizers `0.23.2`, Safetensors `0.8.0`, Accelerate `1.12.0` and Bitsandbytes `0.50.2`.
Python syntax checks and Slurm shell syntax passed, and `git diff --check` was clean.

Job **`964397`** reached its eight-hour limit on 2026-10-12. Qwen one-shot
completed **500/500 dishes at 94.7 kcal MAE**. The geometry pipeline saved 449
attempts with 447 successes at 92.5 kcal MAE; 425 successes used a fallback.
On their 447 common successful dishes, geometry minus one-shot MAE is -4.0 kcal,
with a 95% dish-bootstrap interval of [-15.5, 8.4]. This does not establish
an improvement. The baseline is complete; the geometry result remains partial.

Predictions are saved under `results/nutrition5k_500_qwen_local/`.
[The Qwen results report](nutrition5k_500_qwen_results.md) combines the saved
predictions with RGB regression, Food-R1, FoodLMM and the provisional DPF
reproduction. RGB regression and Food-R1 beat the Qwen baseline on all 500 dishes.
Resume job **`1002790`** was cancelled at the user's request on 2026-10-12.
The geometry implementation, benchmark registration, asset preparation option
and dedicated tests were removed. Current reports contain the five remaining
methods; the geometry numbers above and original raw predictions are historical.
All saved inference-source and manifest hashes were checked before the resume;
new evaluations now require a new output directory because the code changed.
The earlier Dots reports remain separate.
