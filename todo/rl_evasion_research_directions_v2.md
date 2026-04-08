# RL-Based Detector Evasion: Enhanced Research Directions

*Follow-up to the first two reports. This revision drops the scheduled-retraining direction, merges and enhances the self-discovery and self-play directions into one unified core contribution, and enhances the multi-detector/drift direction. Every claim that could be checked is anchored to the literature. The honesty note from the previous report stands: I'll say when an idea is strong, when it rests on a misconception, and what the version that actually works looks like.*

---

## Executive framing

The first report covered the state of the art in RL-based text and image evasion (AuthorMist, StealthRL, StealthDiffusion). The second covered four directions you proposed. In re-reading the second report, I want to be clearer about a single organizing claim that should shape this entire project:

> The meaningful research question is not *"can we evade detector D"* — that's largely settled for the dominant detector family. The meaningful question is *"can we build a generator whose output distribution is close enough to the human distribution that no detector, including ones we never trained against, can reliably separate the two at a useful false-positive rate."*

This reframing matters because it changes what success looks like. Under the old framing, you publish "we beat GPTZero at 95%." Under this framing, you publish "we closed X% of the distributional gap in a feature space that generalizes across detector families, and here is the remaining gap and why it persists." The second is a better paper and a more durable contribution.

With that in mind, the project has two connected research threads — a *distributional* thread and an *adversarial* thread — that should be pursued together.

---

## Thread 1 — Closed-loop distribution matching with a persistent feature bank

*(Merges and enhances the previous "self-discovery of detectable patterns" direction and the "is SSL relevant" direction. These were never really separate — one is the goal, the other is a family of techniques for achieving it.)*

### The theoretical anchor: SPIN

The right starting point is SPIN (Self-Play Fine-Tuning, Chen et al., 2024, arXiv:2401.01335). SPIN iteratively trains an LLM to distinguish its own generated responses from human-written ones in an SFT dataset, and the authors prove that **the global optimum of the SPIN objective is achieved exactly when the model's output distribution matches the human data distribution** ([SPIN project page](https://uclaml.github.io/SPIN/); [GitHub](https://github.com/uclaml/SPIN)). That theoretical result is exactly what you want for a detector-evasion project framed as distribution matching rather than detector-score maximization. It also gives you a principled answer to the question "how will I know when to stop training?" — you stop when the internal discriminator can no longer separate the two distributions, which is by construction the point at which *any* external detector relying on the same feature subspace must also fail.

SPIN's practical mechanics are helpful: in each iteration, the current LLM plays both roles, the logistic-loss formulation reduces to a DPO-style update ([verl documentation](https://verl.readthedocs.io/en/latest/algo/spin.html)), and this has been replicated in LoRA-efficient form ([thomasgauthier/LLM-self-play](https://github.com/thomasgauthier/LLM-self-play)). So you can actually run this on a 7B model with a single GPU.

**What SPIN is missing for our purpose.** SPIN was designed to improve general quality, not to specifically close the human/machine distributional gap that detectors exploit. Its discriminator is an LLM finetuned with DPO loss, which implicitly captures whatever features the model can represent — but there is no explicit guarantee that those features cover the ones real detectors actually use (perplexity curvature à la DetectGPT/Fast-DetectGPT, token-level log-rank statistics à la Binoculars, stylometric features à la Soto et al.). The model might learn to be "more human-like" in some embedding space while still being detectable on burstiness or POS-bigram frequency. This is exactly the failure mode the May 2025 paper (arXiv:2505.14608) documented: optimizing against probability-curvature detectors leaves stylometric signatures intact.

### The enhancement: explicit multi-feature distribution matching

The proposal is to extend SPIN so the self-play game is played in **multiple feature spaces simultaneously**, chosen specifically to cover the feature families that real detectors use. The list isn't speculative — it comes directly from the detector literature.

**Stylometric features (covers StyloAI, Kumarage et al. 2023, Soto et al. 2024, Zaitsu & Jin 2023):**

- **Burstiness** — the variance of sentence-by-sentence perplexity. Every detector in the GPTZero lineage uses this, and it's been known since Ippolito et al. (2020) that LLM output has systematically lower burstiness than human text. The Counter Turing Test paper (Kumarage et al., EMNLP 2023, [arXiv:2310.05030](https://aclanthology.org/2023.emnlp-main.136.pdf)) formalizes this as a core detection signal.
- **Phraseology, punctuation, linguistic diversity** — the three-feature decomposition from Kumarage et al. 2023, which StyloAI (Opara, AIED 2024, [arXiv:2405.10129](https://arxiv.org/abs/2405.10129)) showed achieves 81–98% accuracy with just 31 stylometric features and a random forest.
- **Function-word unigrams, POS bigrams, phrase patterns** — Zaitsu & Jin (PLOS ONE 2023) report these as the most discriminative features for detecting GPT-4-generated text, with perfect discrimination on MDS dimensions. These are the features that would survive in a "language-agnostic" detector and they're trivially cheap to compute with spaCy.
- **Type-token ratio, lexical diversity, readability** — standard stylometry, included in StyloMetrix and in the Tarım & Onan 2025 comparison ([arXiv:2507.10475](https://arxiv.org/html/2507.10475)).

**Token-distribution features (covers DetectGPT, Fast-DetectGPT, Binoculars, GLTR, log-rank detectors):**

- **Perplexity curvature** — DetectGPT (Mitchell et al., 2023) and Fast-DetectGPT's core signal; model output sits in negative-curvature regions of the log-probability surface. You can compute this in closed form using a reference LM as scorer.
- **Log-rank statistics** — the Binoculars (Hans et al., 2024) signal. Two reference LMs, ratio of cross-perplexity to perplexity.

**Neural embedding features (covers RoBERTa classifiers, RADAR, MAGE-trained detectors):**

- **Sentence encoder distance** — compute MMD or Wasserstein-1 between E5 or SBERT embeddings of model output and human reference. This is what StealthRL already uses as a semantic preservation term ([arXiv:2602.08934](https://arxiv.org/pdf/2602.08934)) and it's the cleanest way to bring a learned representation space into the loop.
- **Author-representation embeddings** — Soto et al.'s style embeddings are the ones that defeated RL optimization in the 2505.14608 result, so including them as a matching target is the direct fix for that robustness gap.

### The composite training objective

Call it **MultiSPIN**. At iteration t, the generator $\pi_{\theta_t}$ produces candidate text y' given prompts x drawn from a human reference set. The loss combines:

$$
\mathcal{L}_{\text{MultiSPIN}} = \mathcal{L}_{\text{SPIN}}(y, y') + \lambda_1 \|\phi_{\text{stylo}}(y) - \phi_{\text{stylo}}(y')\|^2 + \lambda_2 \text{MMD}(\phi_{\text{emb}}(y), \phi_{\text{emb}}(y')) + \lambda_3 \text{MMD}(\phi_{\text{auth}}(y), \phi_{\text{auth}}(y')) + \lambda_4 \mathcal{L}_{\text{task}}
$$

where $\phi_{\text{stylo}}$ is a stylometric feature extractor (StyloMetrix or StyloAI's 31-feature set), $\phi_{\text{emb}}$ is a frozen E5 encoder, $\phi_{\text{auth}}$ is the Soto style encoder, and $\mathcal{L}_{\text{task}}$ is a task-quality term (see §Guardrails). The $\lambda$s are tuning knobs but I'd start with all of them at the same scale and grid search one ablation at a time.

The loss is not a detector loss. No external AI-text detector appears anywhere in the training signal. The model is being pulled toward the human distribution in a feature space that *covers* the features real detectors use, without the model ever seeing those detectors. If this works, the resulting model should transfer zero-shot to held-out detectors.

### The persistent feature bank

SPIN uses only the previous iteration's generator as the opponent. For our problem, that throws away information: every iteration identifies *which features the current model gives itself away on*, and that information should be carried forward. Concretely:

- After each training round, run the trained model's outputs through a battery of **probe classifiers** — small linear or MLP probes on top of the feature vectors above, trained to separate the current generator from human reference.
- Record which features/feature-combinations each probe uses (via feature importance for linear probes, or integrated gradients for MLPs). This is the "tells" for round t.
- Add the most discriminative probes to a persistent **feature bank**. On the next training round, the loss includes a penalty against being separable by any probe in the bank.
- This is mechanically close to **elastic weight consolidation** (Kirkpatrick et al., 2017) applied to a feature-matching objective rather than to weights, and it addresses catastrophic forgetting of previous robustness properties across training rounds without storing old model weights.

This turns the training procedure into a continual distribution-matching loop where each round permanently patches the previous round's tells. It's the technical realization of what you intuitively called "long-term memory." I'd avoid that phrase in the paper — it reads as memory-augmented models to most readers — and instead call it "cumulative robustness via a persistent discriminator bank" or similar.

### What this actually achieves versus what it doesn't

Honest inventory of what I expect this to work for and what I expect to remain hard.

**Expected wins.**
- Should close the burstiness/perplexity-variance gap (Ippolito 2020; Tarım & Onan 2025), which kills GPTZero-style detectors.
- Should close the perplexity-curvature gap that DetectGPT/Fast-DetectGPT rely on, because the explicit term in the loss matches the detector's own feature.
- Should, by construction, transfer to held-out detectors that use any of the feature families in the loss.
- Should degrade the stylometric baseline (function-word bigrams, POS bigrams) that currently achieves high accuracy on LLM text (Zaitsu & Jin 2023; StyloAI 2024).

**Expected failures.**
- **Semantic and discourse-level features will persist.** LLMs have characteristic discourse structures — over-hedging, tripartite list preference, stock transitions ("furthermore," "however," "in conclusion"), a specific rhythm of qualification — that are not captured by any of the feature extractors above because they live above the sentence level. A detector that explicitly models discourse structure (there aren't many good ones yet, but they exist, e.g., GPT-Who, Venkatraman et al., Findings NAACL 2024) will still work. This is a limitation to acknowledge up front rather than discover at paper-submission time.
- **Long-text detection will stay harder than short-text detection.** The 2024 PLOS ONE paper on Japanese stylometric detection shows that longer samples make detection *easier*, because statistics converge. Nothing in MultiSPIN changes that — it closes the per-sentence gap, not the aggregate gap.
- **Reference-corpus contamination.** If your "human" reference data is scraped post-2023, it's polluted with LLM output. This is the model collapse failure mode (Shumailov et al. 2024, ["The Curse of Recursion"](https://arxiv.org/abs/2305.17493); Nature 2024) — the curse is that training generator outputs against other generator outputs causes distributional tails to disappear. For our purpose the concrete risk is that you train the model to "match humans" against a reference that already looks partially AI, closing the wrong gap. **The fix is non-negotiable: use provenance-verified pre-2023 corpora** (CNN/DailyMail, pre-2023 Wikipedia dumps, PubMed abstracts from pre-2023, Project Gutenberg, Reddit pre-2021). Gerstgrasser et al. ([arXiv:2404.01413](https://arxiv.org/html/2404.01413v2), COLM 2024) showed that *accumulating* real data alongside synthetic avoids collapse, but that doesn't apply here because the entire training signal depends on the reference being authentically human.

### On the "is SSL relevant" question

I addressed this in the previous report and want to tighten it. The answer is: **SSL in the textbook sense (masked prediction, contrastive pretraining, BYOL) is not the right framing** — you have free labels ("human" vs "generator-t"), so representation learning from unlabeled data isn't the bottleneck. The right framings are:

1. **Self-play** — which is what SPIN and the MultiSPIN extension above use. The generator and discriminator are the same model at different iterations. This is a closed-loop adversarial training in the GAN tradition, not SSL.
2. **Unsupervised distribution matching** — the MMD / Wasserstein / stylometric-feature terms above. These are distributional objectives, not self-supervised pretext tasks.
3. **Density-ratio estimation** — you can train a small classifier $h(x) \approx p_{\text{human}}(x) / p_{\text{model}}(x)$ on the available labeled data and use $\nabla \log h$ as guidance. This is a reasonable alternative to SPIN's DPO-style inner loop and is known to have better gradient properties for distribution matching in some regimes. If SPIN training proves unstable, density-ratio estimation is the first fallback I'd try.

None of these are "self-supervised learning." Use the right names — it'll save you from a reviewer correctly complaining that you're mislabeling standard techniques.

---

## Thread 2 — Multi-detector robustness and drift adaptation (enhanced)

*(Refined version of the previous multi-detector direction. The "ensemble robustness" framing by itself is already done; the novel contribution is in drift adaptation, the defender-wins analysis, and meta-learning for fast re-adaptation.)*

### What's already solved, so you don't waste time on it

StealthRL ([arXiv:2602.08934](https://arxiv.org/pdf/2602.08934)) already trains against a three-detector ensemble (RoBERTa classifier, Fast-DetectGPT, and a zero-shot baseline) and reports zero-shot transfer to Binoculars as a held-out detector, with near-zero detection at TPR@1%FPR. AuthorMist ([arXiv:2503.08716](https://ar5iv.labs.arxiv.org/html/2503.08716)) trains against commercial API detectors. "Training against an ensemble of detectors" is not a contribution.

The MAGE benchmark (Li et al., ACL 2024, [arXiv:2305.13242](https://arxiv.org/abs/2305.13242), [GitHub](https://github.com/yafuly/MAGE)) is also important to note: it's the canonical testbed for cross-domain, cross-model detection in the wild, and it showed that standard detectors drop severely out-of-distribution — GLTR from 0.94 to 0.80 AUROC, DetectGPT from 0.92 to 0.57. **But** the RAID benchmark follow-up (Dugan et al., 2024) reported that when detectors are trained on the in-distribution generators, multiple teams achieved >99% TPR at 5% FPR. So the gap between "out-of-distribution" and "adversarially-trained in-distribution" detection is huge, which is the key fact that motivates the drift-adaptation question.

### The defender's advantage and RADAR

The single most important paper for this thread is **RADAR** (Hu, Chen, Ho, NeurIPS 2023, [arXiv:2307.03838](https://arxiv.org/abs/2307.03838), [IBM GitHub](https://github.com/IBM/RADAR)). RADAR performs exactly the arms race I said you should be studying: it jointly trains a paraphraser (as the attacker, updated via PPO with the detector's score as reward) and a RoBERTa-based detector (updated via supervised learning on the current paraphraser's outputs), alternating until the validation AUROC stops improving. The results are instructive: **after adversarial training, RADAR achieves 0.857 mean AUROC against an unseen paraphraser, a 31.64% improvement over the previous best (entropy-based, at 0.651)**. The defender closes the gap — not completely, but substantially. And RADAR transfers across LLMs (from instruction-tuned training models to GPT-3.5-Turbo at test time).

This is the empirical grounding for the honest prediction I gave before: **in the adversarial equilibrium, the defender is at an advantage as long as it can adversarially retrain on recent attacker outputs**. Your project should take RADAR as the baseline defender and measure where the equilibrium actually sits when the attacker is a MultiSPIN-trained model rather than a vanilla paraphraser. That measurement doesn't currently exist in the literature and would be a real contribution.

### The enhancement: meta-learning for fast drift adaptation

The concrete open problem is: given that the defender will retrain in response to your attack, how cheaply can your attacker re-adapt? This is the natural setting for **meta-learning**. MAML (Finn et al., ICML 2017, [arXiv:1703.03400](https://arxiv.org/pdf/1703.03400)) and its variants learn a meta-initialization such that a small number of gradient steps adapts the model to a new task. For our problem, each "task" is a version of the detector — a different checkpoint, a different architecture, a different retraining run.

The closest published work is **Wang et al. (ICLR 2021), "On Fast Adversarial Robustness Adaptation in Model-Agnostic Meta-Learning"** ([arXiv:2102.10454](https://arxiv.org/abs/2102.10454v1), [MIT-IBM blog](https://mitibmwatsonailab.mit.edu/research/blog/on-fast-adversarial-robustness-adaptation-in-model-agnostic-meta-learning/)). They showed that robustifying the meta-update stage is sufficient to carry robustness through to task-specific fine-tuning, and they specifically use contrastive learning as an auxiliary task to improve the meta-learned initialization's adversarial robustness. This is the template. Substitute "task = detector version" and "adversarial robustness = low detector score" and you have the algorithm.

**Concrete setup:**

1. Build a **detector zoo** — a population of 10–20 detectors covering the three feature families (probability-curvature, supervised classifier, stylometric). Include at least one instance of each of the MAGE-style out-of-distribution detector, a RADAR-trained robust detector, Binoculars, Fast-DetectGPT, and a Soto-style detector.
2. In each meta-training episode, sample a batch of detectors. For each detector $D_i$, compute a few gradient steps of MultiSPIN fine-tuning of the generator specifically against the feature family $D_i$ uses. Measure the evasion rate after k steps (k=1, 5, 20).
3. The meta-objective is to minimize the evasion rate *after k adaptation steps*, averaged over sampled detectors. This is the standard MAML bi-level objective.
4. Test-time protocol: present the meta-trained model with a *new* detector version (held-out from the zoo), perform k gradient steps of adaptation on a small set of that detector's scores on the model's outputs, and measure evasion.

The numbers you care about are: (a) how few adaptation steps are needed to restore evasion, (b) how much compute that costs, (c) how this compares to training from scratch against the new detector. The scientific claim you're building toward is a **quantitative adaptation-cost curve** — "with a meta-learned initialization, the attacker can re-adapt to a new detector version in N gradient steps, costing X GPU-hours; without meta-learning, it takes 10–100× more." That's a number nobody currently has.

### The population-based training baseline

As a simpler baseline for meta-learning, consider **population-based training** against the detector zoo, sampling a detector per batch rather than per episode. This is just domain randomization over detectors and is cheaper to implement than full MAML. Often domain randomization does most of what meta-learning claims to do, and it should be the baseline in your ablation table.

### The equilibrium experiment (most important experiment in the project)

Run the arms race explicitly:

- **Round 0**: initial generator, initial detector ensemble from the zoo.
- **Round r → r+1**: at each round, first fine-tune the generator against the current detector ensemble using MultiSPIN + meta-learning, then retrain the detectors on the new generator outputs (à la RADAR).
- **Metric**: measure TPR@1%FPR at each round from the defender's perspective, and attack success rate from the attacker's perspective. Run for N=10 rounds.
- **What to report**: the equilibrium curve, the per-round adaptation cost for each side, and whether the equilibrium is monotonic (one side winning over time) or oscillating.

This is a fairly cheap experiment in wall-clock terms — each round is a LoRA fine-tune on both sides — but I have seen no paper do it beyond 2-3 rounds. **A well-executed 10-round equilibrium study with the MultiSPIN attacker and a RADAR-style defender is, I think, the single most publishable artifact this project could produce.** It's more scientifically informative than another "we beat GPTZero" paper because it tells you something nobody currently knows: whether the arms race converges, where it converges, and how much it costs to stay ahead.

### Important caveat: meta-learning is itself attackable

One failure mode to note. The 2020 paper "Yet Meta Learning Can Adapt Fast, It Can Also Break Easily" ([arXiv:2009.01672](https://ar5iv.labs.arxiv.org/html/2009.01672)) showed that meta-learners (MAML, SNAIL, Prototypical) are themselves vulnerable to adversarial manipulation of their adaptation data. In our setting, a clever defender could poison the adaptation set (the handful of scored examples the attacker uses to meta-adapt) to steer the attacker into the wrong direction. This is a subtle attack surface and should at least be discussed in the threat model section of the paper.

---

## Guardrails that apply to both threads

Short list of things that aren't research contributions but determine whether the research contributions survive.

- **Task quality evaluation is non-optional.** Every paper in the evasion literature gets dinged for not reporting what happens to downstream task quality. Evaluate on MT-Bench, IFEval, HumanEval, and a factuality benchmark (TruthfulQA or SimpleQA) at every round. Report the degradation. If MultiSPIN training collapses task quality even while closing the human/machine gap, that's an important negative result.
- **Human reference corpus provenance.** Already mentioned but worth restating. Use pre-2023 sources with documented provenance. The 2024 PAN / ELOQUENT results (reported in the stylometry-recognition paper at [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425026181)) found that simple Unicode obfuscations defeated most obfuscation submissions, which tells you that dataset quality and careful evaluation matter much more than clever tricks.
- **KL regularization against the base model.** SPIN's theoretical guarantee assumes the training stays within a reasonable neighborhood. Without KL control you'll reward-hack within a few iterations. DDPO's documented reward collapse (Black et al., 2023) is the diffusion-side version of this and it's just as real for LLMs.
- **Red-team framing.** I keep hammering this but it actually helps the science. Framing the project as "measuring how hard detectors are to evade and how fast they can re-adapt" forces you to evaluate against the strongest detectors and to do the equilibrium experiment — which is the version of the project that produces a real contribution rather than just another attack paper.

---

## Recommended project scope and sequence

In order, with rough effort estimates for a single researcher with one GPU node.

1. **Implement MultiSPIN on a 7B open model** (Qwen2.5-7B or Llama-3.1-8B), pre-2023 reference corpus, all four feature-matching terms, LoRA. Evaluate zero-shot against: Fast-DetectGPT, Binoculars, a trained RoBERTa classifier, a stylometric random forest à la StyloAI, and Soto-style embeddings. **Expected outcome**: good performance against the first three, partial performance against stylometric, worst on Soto-style — this tells you which λs need tuning and which features need better extractors. [4–6 weeks]
2. **Add the persistent feature bank** (probe classifiers over feature vectors, accumulated across iterations). Re-run evaluation. **Expected outcome**: cumulative closing of the gap on the feature families that the probes learn to identify. [2–3 weeks]
3. **Build the detector zoo** and test zero-shot transfer to held-out detectors (MAGE-trained, RADAR). **Expected outcome**: measure the transfer gap honestly — this is where the "novel contribution" claim gets validated or not. [2 weeks]
4. **Add MAML-style meta-learning** over the detector zoo, with population-based training as a baseline. Measure adaptation cost to held-out detectors. **Expected outcome**: the adaptation-cost curve. [4–6 weeks]
5. **Run the equilibrium experiment** — 10 rounds of alternating attacker/defender fine-tuning, RADAR-style defender, MultiSPIN+MAML attacker. **This is the headline result.** [3–4 weeks]
6. Paper. [forever]

Total: roughly 4–5 months of full-time work for a competent researcher, assuming things go wrong about as often as they normally do.

---

## Key references

**Closed-loop self-play and distribution matching:**
- Chen, Deng, Yuan, Ji, Gu. *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)*, 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) — the theoretical anchor; proves distributional convergence at the optimum.
- Rafailov et al. *Direct Preference Optimization*, NeurIPS 2023 — the underlying DPO loss SPIN reduces to.
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks (EWC)*, PNAS 2017 — inspiration for the persistent-feature-bank continual learning layer.

**Stylometric detection (the feature families that must be matched):**
- Kumarage, Garland, Bhattacharjee, Trapeznikov, Ruston, Liu. *Stylometric Detection of AI-Generated Text in Twitter Timelines*, 2023. [arXiv:2303.03697](https://arxiv.org/pdf/2303.03697) — three-feature decomposition (phraseology, punctuation, linguistic diversity).
- Opara. *StyloAI: Distinguishing AI-Generated Content with Stylometric Analysis*, AIED 2024. [arXiv:2405.10129](https://arxiv.org/abs/2405.10129) — 31-feature random forest, 81–98% accuracy.
- Zaitsu & Jin. *Distinguishing ChatGPT(-3.5,-4)-generated and human-written papers through Japanese stylometric analysis*, PLOS ONE 2023. Function word unigrams + POS bigrams + phrase patterns for near-perfect discrimination.
- Ippolito et al. *Automatic Detection of Generated Text is Easiest when Humans are Fooled*, ACL 2020 — original burstiness finding.
- Tarım & Onan. *Can You Detect the Difference?*, 2025. [arXiv:2507.10475](https://arxiv.org/html/2507.10475) — stylometric comparison across AR and diffusion LLMs; also shows paraphrase loops can restore burstiness to human levels.
- Soto et al., 2024 — the author-representation style encoder. The arXiv:2505.14608 paper is the one that showed RL-optimized LLMs still have detectable style — this is the robustness gap the project must close.

**Probability-curvature detection:**
- Mitchell et al. *DetectGPT*, ICML 2023. [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)
- Bao et al. *Fast-DetectGPT*, 2024.
- Hans et al. *Spotting LLMs with Binoculars*, 2024 — the two-reference-LM log-rank detector.

**Benchmarks for cross-detector evaluation:**
- Li et al. *MAGE: Machine-generated Text Detection in the Wild*, ACL 2024. [arXiv:2305.13242](https://arxiv.org/abs/2305.13242), [GitHub](https://github.com/yafuly/MAGE) — the cross-domain, cross-model testbed. Shows DetectGPT drops from 0.92 to 0.57 AUROC out-of-distribution.
- Wang et al. *M4: Multi-generator, multi-domain, multi-lingual black-box detection*, EMNLP 2023.
- Dugan et al. *RAID* benchmark, 2024 — >99% TPR at 5% FPR when detectors are trained on in-distribution generators.
- Wu et al. *DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios*, NeurIPS 2024 — shows adversarial perturbation attacks reduce zero-shot detector AUROC by ~39% on average.

**The arms race and defender-side adversarial training:**
- Hu, Chen, Ho. *RADAR: Robust AI-Text Detection via Adversarial Learning*, NeurIPS 2023. [arXiv:2307.03838](https://arxiv.org/abs/2307.03838), [IBM code](https://github.com/IBM/RADAR) — **the most important paper for Thread 2.** Joint PPO paraphraser + detector training; defender AUROC 0.857 vs previous best 0.651.
- Krishna, Song, Karpinska, Wieting, Iyyer. *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*, 2023. [arXiv:2303.13408](https://arxiv.org/abs/2303.13408) — retrieval-based defense.
- Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-generated text be reliably detected?*, 2023 — theoretical impossibility and recursive paraphrasing attacks.

**Meta-learning for fast adversarial adaptation:**
- Finn, Abbeel, Levine. *Model-Agnostic Meta-Learning (MAML)*, ICML 2017. [arXiv:1703.03400](https://arxiv.org/pdf/1703.03400)
- Wang, Xu, Liu, Chen, Weng, Gan, Wang. *On Fast Adversarial Robustness Adaptation in Model-Agnostic Meta-Learning*, ICLR 2021. [arXiv:2102.10454](https://arxiv.org/abs/2102.10454v1) — template for meta-learned adversarial robustness.
- *Yet Meta Learning Can Adapt Fast, It Can Also Break Easily*, 2020. [arXiv:2009.01672](https://ar5iv.labs.arxiv.org/html/2009.01672) — the meta-learner poisoning result; relevant to the threat model.

**RL-based evasion (the direct prior art from Report 1):**
- David & Gervais. *AuthorMist: Evading AI Text Detectors with Reinforcement Learning*, 2025. [arXiv:2503.08716](https://ar5iv.labs.arxiv.org/html/2503.08716) — GRPO against commercial API detectors, 78–96% evasion.
- *StealthRL*, 2026. [arXiv:2602.08934](https://arxiv.org/pdf/2602.08934) — ensemble training with Binoculars held out; TPR@1%FPR near zero. The current best attacker baseline.
- Lu et al. *SICO: Large Language Models can be Guided to Evade AI-Generated Text Detection*, 2023. [arXiv:2305.10847](https://arxiv.org/abs/2305.10847) — in-context example optimization; cheap prompt-only baseline.
- *Language Models Optimized to Fool Detectors Still Have a Distinct Style*, 2025. [arXiv:2505.14608](https://arxiv.org/html/2505.14608v1) — **the robustness gap this project must close.** RL-optimized LLMs evade probability-curvature detectors but still fail against Soto-style stylometric detectors.

**Data contamination and model collapse:**
- Shumailov, Shumaylov, Zhao, Gal, Papernot, Anderson. *The Curse of Recursion: Training on Generated Data Makes Models Forget*, 2023. [arXiv:2305.17493](https://arxiv.org/abs/2305.17493) — Nature 2024 extended version. Why your reference corpus provenance matters.
- Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*, COLM 2024. [arXiv:2404.01413](https://arxiv.org/html/2404.01413v2) — the rebuttal; doesn't apply directly to our setting but worth citing for completeness.

**RL fine-tuning methods:**
- Schulman et al. *PPO*, 2017.
- Shao et al. *GRPO* (DeepSeek), 2024 — used by AuthorMist and StealthRL.
- Rafailov et al. *DPO*, NeurIPS 2023.
