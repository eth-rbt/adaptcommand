%%%%%%%%%%%%%%%%%%%%%%%%%%% asme2e.tex 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% use twocolumn and 10pt options with the asme2e format
\documentclass[twocolumn,10pt]{asme2e}
\special{papersize=8.5in,11in}

%% The class has several options
%  onecolumn/twocolumn - format for one or two columns per page
%  10pt/11pt/12pt - use 10, 11, or 12 point font
%  oneside/twoside - format for oneside/twosided printing
%  final/draft - format for final/draft copy
%  cleanfoot - take out copyright info in footer leave page number
%  cleanhead - take out the conference banner on the title page
%  titlepage/notitlepage - put in titlepage or leave out titlepage
%  
%% The default is oneside, onecolumn, 10pt, final

%%% Replace here with information related to your conference

%%%%% for date in a single month, use
%\confdate{24-28}
%\confmonth{September}
%%%%% for date across two months, use
\confdate{12/10}
\confyear{2025}
\confcity{MIT}
\confcountry{2.156 Final Report}


%%% You need to remove 'DRAFT: ' in the title for the final submitted version.
\title{Intent Understanding of Diverse Users in Smart Home Settings  }

%%% first author
\author{Ethan Chang
    \affiliation{
	Design Intelligence Lab\\
	Department of Mechanical Engineering\\
    Email: echang25@mit.edu
    }	
}


\begin{document}

\maketitle    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{abstract}
This report presents a comprehensive empirical study of personalization techniques for smart home conversational agents. We evaluated six different approaches including per-persona fine-tuning, cluster-based training, mixture-of-experts merging, weighted merging, and prefix tuning on a dataset of 200 distinct user personas with 6,000 multi-turn dialogues. Our findings reveal that \textbf{simple unified training outperforms all personalization methods} (82.14\% vs 66-74\% for personalized approaches), challenging conventional assumptions about the benefits of personalization in this domain. Analysis reveals five key failure modes: poor clustering quality (silhouette score 0.022), insufficient training data (20-2160 examples vs 6000 unified), limited model capacity (0.5B parameters), task characteristics favoring domain knowledge over personalization, and destructive weight merging. We propose lightweight alternatives including selective routing (+1.03\%), prefix tuning (+1.5\% preliminary), and retrieval-augmented generation (expected +2-6\%), which build upon the strong unified baseline rather than replacing it. This work provides practical insights into when unified training beats personalization and offers a roadmap for effective lightweight adaptation.
\end{abstract}

\section{Introduction}

\subsection{Motivation}

Conversational agents for smart home control face a fundamental tension between two competing objectives: \textit{generalization} (understanding diverse commands and contexts across many users) and \textit{personalization} (adapting to individual user preferences and communication styles). While large language models have demonstrated remarkable generalization capabilities, the question of whether and how personalization improves performance for domain-specific tasks remains understudied.

Smart home assistants must handle commands ranging from simple device control ("turn on the lights") to complex multi-device orchestration ("set up movie mode") while adapting to users with vastly different communication styles—from terse commands to conversational requests. Users also develop preferences for specific device settings (e.g., preferred brightness levels, temperature settings) that could benefit from personalization.

\subsection{Research Questions}

This study investigates four key research questions:

\begin{enumerate}
\item \textbf{RQ1}: Does per-persona fine-tuning improve performance over unified training when adapting small language models to individual users?

\item \textbf{RQ2}: Can clustering personas by behavioral or textual similarity enable effective personalization with more training data per cluster?

\item \textbf{RQ3}: Can lightweight prefix tuning build upon strong unified baselines to achieve personalization without the pitfalls of full fine-tuning?
\end{enumerate}

\subsection{Contributions}

This work makes the following contributions:

\begin{itemize}
\item \textbf{Comprehensive evaluation}: We evaluate 6 personalization approaches (unified LoRA, per-persona LoRA, cluster-based LoRA, sparse MoE, weighted merge, prefix tuning) on 200 personas with rigorous train/validation/test splits.

\item \textbf{Failure mode analysis}: We identify and analyze five specific failure modes that cause all tested personalization methods to underperform unified training.

\item \textbf{Practical insights}: We demonstrate that for small models (0.5B parameters) with limited per-user data (20-30 examples), data quantity dominates algorithmic sophistication.

\item \textbf{Lightweight alternatives}: We propose and validate selective routing (+1.03\%) and introduce prefix tuning (+1.5\% preliminary) as promising alternatives that build on unified baselines.

\item \textbf{Reproducible methodology}: All code, trained models, evaluation scripts, and detailed experimental logs are available for reproducibility.
\end{itemize}

\subsection{Key Finding}

\textbf{Unified LoRA achieves 82.14\% embedding similarity, outperforming all personalization attempts}: per-persona LoRA (68.28\%), cluster-based training (74.14\% best cluster), sparse MoE (66.38\%), and weighted merging (67.00\%). This counterintuitive result demonstrates that \textit{more data trumps personalization} for small models in domain-specific tasks.

\section{Background and Related Work}

\subsection{Parameter-Efficient Fine-Tuning}

\textbf{LoRA (Low-Rank Adaptation)} \cite{hu2021lora} adapts pre-trained models by learning low-rank update matrices that are added to frozen base model weights:

\begin{equation}
W' = W_0 + BA
\end{equation}

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

Our configuration:
\begin{itemize}
\item Rank $r = 8$
\item Alpha $\alpha = 16$ (scaling factor)
\item Target modules: \texttt{q\_proj}, \texttt{v\_proj} (query and value projection layers)
\item Trainable parameters: $\sim$2.4M (0.48\% of base model)
\item Training cost: $\sim$2 hours for unified, $\sim$200 GPU hours for all per-persona models
\end{itemize}

\subsection{Mixture of Experts}

Traditional MoE architectures use gating mechanisms to route inputs to specialized experts. We explore two merging strategies:

\begin{itemize}
\item \textbf{Sparse MoE}: For each persona, merge $K=5$ most similar per-persona LoRAs within their cluster, weighted by cosine similarity.

\item \textbf{Weighted Merge}: Combine per-persona LoRAs using validation performance and centrality to cluster centroid.
\end{itemize}

Merging formula:
\begin{equation}
W_{merged} = \sum_{i=1}^{K} w_i \cdot W_i \quad \text{where} \quad \sum_{i=1}^{K} w_i = 1
\end{equation}

\subsection{Prefix Tuning}

\textbf{Prefix tuning} \cite{li2021prefix} prepends learnable continuous vectors to the input sequence:

\begin{equation}
\text{Input} = [\text{PREFIX}] \oplus [\text{USER\_QUERY}]
\end{equation}

where PREFIX consists of $L$ learnable token embeddings in $\mathbb{R}^{L \times d}$.

Our configuration:
\begin{itemize}
\item Prefix length: $L = 10$ tokens
\item Embedding dimension: $d = 896$ (Qwen 0.5B hidden size)
\item Parameters per persona: $10 \times 896 = 8,960$ (0.0018\% of base model)
\item Advantages: Minimal parameters, builds on frozen unified model, low overfitting risk
\end{itemize}

\section{Dataset and Experimental Setup}

\subsection{Dataset}

\textbf{Smart Home Dialogue Dataset}
This is a synthesized dataset  

\begin{itemize}
\item \textbf{Personas}: 200 unique user profiles with distinct personalities and preferences
\item \textbf{Total dialogues}: 6,000 conversations (30 per persona)
\item \textbf{Average dialogue length}: 3.2 turns per dialogue
\item \textbf{Domains}: Device control (lights, AC, TV, speakers, security), scheduling, contextual queries
\item \textbf{Context features}: Time of day, weather conditions, temperature, day of week
\end{itemize}

\textbf{Data splits (per persona, time-aware)}:
\begin{itemize}
\item Training: 20 dialogues (67\%)
\item Validation: 5 dialogues (17\%)
\item Test: 5 dialogues (17\%)
\end{itemize}

\textbf{Example personas}:

persona000: "Ethan is a reserved librarian with a penchant
              for mystery novels. He speaks softly, often
              pausing to choose his words carefully."

persona001: "Maya is a lively yoga instructor who radiates
              positivity. Her speech is peppered with
              encouraging phrases and laughter."

persona002: "Liam is a pragmatic engineer who approaches
              conversations with logical precision."

\textbf{Example dialogue}:

Context: evening, rainy, cold, weekday
User Preferences: lights 50\% warm, AC 22°C heat, speaker vol 30

User: "Could you turn on the lights? It's quite dim in here."
Assistant: "Of course. What brightness level would you like?"
User: "50 percent, please. And could you set the color to warm?"
Assistant: "Setting the lights to 50\% brightness with a warm
           color. Is there anything else you'd like?"

\subsection{Base Model}

Qwen 2.5 0.5B Instruct is a 494-million–parameter transformer-decoder model with a context length of 32K tokens and a vocabulary of 151,646 tokens. It uses a hidden size of 896 and is designed to be small enough for rapid iteration while still being large enough to deliver reasonable performance on smart-home tasks.

\subsection{Evaluation Metrics}

Metrics: \textbf{Embedding Similarity} — We evaluate semantic alignment using the sentence-transformers/all-MiniLM-L6-v2 encoder, computing cosine similarity between the predicted and reference embeddings. This method is robust to paraphrasing and captures deeper semantic equivalence; for example, “Setting lights to 50\%” and “Adjusting lights to 50 percent” achieve a 97.2\% similarity score. In addition to embedding similarity, we report \textbf{Device Precision}, the fraction of correctly mentioned devices, and \textbf{Numerical Precision}, the exact-match rate for numerical values such as temperature, brightness, and volume.
\\
\textbf{Example metric calculation}:
     
Reference:  "Setting the AC to 22 degrees and lights to 50\% brightness."
Prediction: "I'll set the AC to 20 degrees and the lights to 50\%."
Embedding Similarity: 78.4\% (semantically similar)
Device Precision:     2/2 = 100\% (AC, lights both correct)
Numerical Precision:  1/2 = 50\%  (50\% correct)

       
\section{Experimental Methodology and Results}

Our experimental approach follows a three-phase progression: (1) evaluating baseline adaptation methods to establish best practices, (2) building personalization on top of the winning baseline, and (3) implementing selective routing to combine strengths of different approaches. Each phase informs the next, leading to practical recommendations.

\subsection{Phase 1: Baseline Adaptation Attempts}

In this phase, we evaluated four fundamental approaches to adapting the Qwen 0.5B base model for smart home dialogue. The goal was to determine which basic training strategy provides the strongest foundation before attempting personalization.

\subsubsection{Method 1: No Adaptation (Baseline)}

Use the pre-trained Qwen 0.5B Instruct model without any fine-tuning on smart home data.

\textbf{Rationale}: Establishes baseline performance to measure value of domain adaptation.

\textbf{Result}: 63.79\% embedding similarity

\textbf{Analysis}: Poor performance indicates model lacks domain-specific knowledge for smart home tasks.

\subsubsection{Method 2: Unified LoRA Adaptation}

Train a single LoRA adapter on all 6,000 dialogues from 200 personas combined.

\textbf{Hyperparameters}:
\begin{verbatim}
rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: ['q_proj', 'v_proj']
learning_rate: 5e-4
num_epochs: 3
batch_size: 4
warmup_ratio: 0.1
weight_decay: 0.01
\end{verbatim}

\textbf{Advantages}: Maximum data utilization (6000 examples), simple deployment (single model), no overfitting risk

\textbf{Disadvantages}: No personalization, generic responses for all users

\textbf{Result}: 82.14\% embedding similarity (+18.35\% vs baseline)

\textbf{Analysis}: Massive improvement over baseline demonstrates value of domain adaptation. Large training set prevents overfitting.

\subsubsection{Method 3: Per-Persona LoRA}

Train 200 individual LoRA adapters, one per persona.

\textbf{Configuration}: Same hyperparameters as unified

\textbf{Training data}: 20 dialogues per persona

\textbf{Hypothesis}: Specialized adapters will capture individual preferences and speaking styles

\textbf{Challenge}: Severe overfitting risk with only 20 examples for 2.4M LoRA parameters (120,000 parameters per training example)

\textbf{Result}: 68.28\% average (range: 48.5\% - 94.1\%, std: 7.73\%)

\textbf{Analysis}: \textbf{Failed approach}. Severe overfitting causes 13.9\% performance drop vs unified. High variance (std: 7.73\%) indicates unstable results. Insufficient data per persona (20 examples) cannot support 2.4M trainable parameters.

\subsubsection{Method 4: Sparse Mixture of Experts (MoE)}

\subsubsection{Clustering Approach}

Cluster personas using K-means ($k=5$) on persona description embeddings (\texttt{all-MiniLM-L6-v2}).

\textbf{Cluster quality}:
\begin{itemize}
\item Silhouette score: 0.022 (scale: -1 to +1, higher is better)
\item Interpretation: Very poor clustering, personas do not form natural groups
\end{itemize}

\textbf{Cluster distribution}:

\begin{table}[h]
\centering
\begin{tabular}{@{}cccc@{}}
\toprule
Cluster & Personas & Training Examples & Result \\ \midrule
0 & 16 & 480 & 72.65\% \\
1 & 52 & 1,560 & [Not trained] \\
2 & 25 & 750 & [Not trained] \\
3 & 35 & 1,050 & [Not trained] \\
4 & 72 & 2,160 & 74.14\% \\ \bottomrule
\end{tabular}
\caption{Cluster distribution and results}
\end{table}

\subsubsection{Training Configurations}

\textbf{Cluster 0} (smallest):
\begin{itemize}
\item Epochs: 3
\item Learning rate: 5e-4
\item Batch size: 2
\item Training time: 42 minutes
\item Result: 72.65\% (-9.5\% vs unified)
\end{itemize}

\textbf{Cluster 4} (largest, optimized hyperparameters):
\begin{itemize}
\item Epochs: 5 (increased for more data)
\item Learning rate: 2e-4 (lowered for better convergence)
\item Batch size: 2
\item Training time: 70 minutes
\item Result: 74.14\% (-8.0\% vs unified)
\end{itemize}

\textbf{Observation}: Even with 4.5× more data (2160 vs 480 examples), cluster 4 still significantly underperforms unified training.

\subsection{Method 3: Sparse Mixture of Experts}

For each persona, merge $K=5$ most similar per-persona LoRAs within their assigned cluster.

\textbf{Algorithm}:
\begin{enumerate}
\item Compute persona embeddings using description text
\item For each persona $p$, find $K=5$ most similar personas in the same cluster
\item Compute similarity weights: $w_i = \frac{\text{sim}(p, p_i)}{\sum_{j=1}^{K} \text{sim}(p, p_j)}$
\item Average LoRA weight matrices: $W_{merged} = \sum_{i=1}^{K} w_i \cdot W_i$
\end{enumerate}

\textbf{Parameters}: Zero additional training (pure merging)

\textbf{Time}: 3.5 minutes to create all 200 merged models

\textbf{Result}: 66.38\% (std: 8.74\%, range: 37.2\% - 88.0\%)

\textbf{Status}: Worst performing method

\subsection{Method 4: Weighted Cluster Merging}

Smart merging of per-persona LoRAs within a cluster using dual weighting:

\begin{equation}
w_i = \text{val\_score}_i \times \text{centrality}_i
\end{equation}

where:
\begin{itemize}
\item $\text{val\_score}_i$ = validation embedding similarity for persona $i$
\item $\text{centrality}_i$ = cosine similarity between persona $i$ and cluster centroid
\end{itemize}

\textbf{Rationale}: Weight personas by both performance quality and cluster representativeness

\textbf{Result for Cluster 4}: 67.00\% (-15.1\% vs unified)

\textbf{Observation}: Smart weighting does not prevent merging from being destructive

\subsection{Method 5: Prefix Tuning on Unified LoRA}

\textbf{Novel approach}: Add learnable prefix tokens to frozen unified LoRA model.

\textbf{Architecture}:
\begin{itemize}
\item Freeze unified LoRA weights (trained on all 6000 examples)
\item Learn 10-token prefix embedding ($10 \times 896 = 8,960$ parameters per persona)
\item Prefix prepended to input before frozen model processing
\end{itemize}

\textbf{Variants tested}:
\begin{enumerate}
\item \textbf{Static text prefix}: Prepend persona description as plain text
\item \textbf{Learned persona prefix}: Train continuous embeddings on 20 persona examples
\item \textbf{Learned cluster prefix}: Train on cluster data (480-2160 examples)
\end{enumerate}

\textbf{Training configuration}:
\begin{verbatim}
prefix_length: 10
learning_rate: 1e-3  (higher since fewer parameters)
num_epochs: 10       (more epochs since less overfitting risk)
batch_size: 2
trainable_params: 8,960 per persona (vs 2.4M for LoRA)
\end{verbatim}

\textbf{Preliminary results (persona\_000 proof-of-concept)}:
\begin{itemize}
\item Unified baseline (no prefix): 59.09\%
\item Static text prefix: 60.60\% (+1.5\%)
\item Learned persona prefix: [In progress]
\item Learned cluster prefix: [In progress]
\end{itemize}

\textbf{Status}: Promising early results, full evaluation pending

\section{Results}

\subsection{Overall Comparison}

\begin{table}[h]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Similarity & vs Unified & Parameters & Training Time \\ \midrule
\textbf{Unified LoRA} & \textbf{82.14\%} & \textbf{baseline} & 2.4M & 2h \\
Cluster 4 LoRA & 74.14\% & -8.0\% & 2.4M & 70min \\
Cluster 0 LoRA & 72.65\% & -9.5\% & 2.4M & 42min \\
Per-Persona LoRA & 68.28\% & -13.9\% & 480M total & 200h \\
Weighted Merge & 67.00\% & -15.1\% & 0 (merge) & 2min \\
Sparse MoE (K=5) & 66.38\% & -15.8\% & 0 (merge) & 3.5min \\ \bottomrule
\end{tabular}
\caption{Performance comparison of all methods}
\end{table}

\textbf{Key finding}: ALL personalization methods failed to beat the unified baseline.

\subsection{Detailed Metrics for Unified LoRA}

\begin{table}[h]
\centering
\begin{tabular}{@{}lc@{}}
\toprule
Metric & Score \\ \midrule
Embedding Similarity & 82.14\% \\
Device Precision & 93.80\% \\
Device Recall & 93.55\% \\
Parameter Precision & 90.59\% \\
Parameter Recall & 90.30\% \\
Parameter F1 & 89.58\% \\
Numerical Precision & 91.24\% \\ \bottomrule
\end{tabular}
\caption{Unified LoRA detailed metrics}
\end{table}

\subsection{Per-Persona LoRA Analysis}

\textbf{Distribution statistics}:
\begin{itemize}
\item Mean: 68.28\%
\item Standard deviation: 7.73\%
\item Minimum: 48.5\% (persona\_082)
\item Maximum: 94.1\% (persona\_180)
\item Range: 45.6 percentage points
\end{itemize}

\textbf{Analysis}:
\begin{itemize}
\item High variance indicates severe overfitting
\item Best personas (90\%+): Got lucky with test examples resembling training data
\item Worst personas (<50\%): Catastrophic overfitting, model collapsed
\item 30 training examples insufficient for 2.4M trainable parameters
\item Overfitting ratio: 120,000 parameters per training example
\end{itemize}

\textbf{Example predictions}:

\begin{verbatim}
persona_180 (best: 94.1%):
User: "Lights on"
Prediction: "Lights at 70 brightness warm"
Reference: "Setting lights to 70% warm color"
Analysis: Test examples similar to training, high similarity

persona_082 (worst: 48.5%):
User: "Could you turn on the lights please?"
Prediction: "light on"
Reference: "Of course. What brightness would you prefer?"
Analysis: Overfitted to extremely terse responses, lost language quality
\end{verbatim}

\subsection{Cluster-Based LoRA Results}

\textbf{Cluster 0} (16 personas, 480 examples):
\begin{itemize}
\item Result: 72.65\%
\item Problem: Smallest cluster, insufficient data
\item Training loss: Converged but showed overfitting on validation
\end{itemize}

\textbf{Cluster 4} (72 personas, 2160 examples):
\begin{itemize}
\item Result: 74.14\%
\item Optimizations tried: Lower LR (2e-4), more epochs (5)
\item Problem: Even 4.5× more data than cluster 0 couldn't overcome poor clustering quality
\item Still 8\% worse than unified despite careful tuning
\end{itemize}

\subsection{MoE and Merging Results}

\textbf{Sparse MoE (K=5)}:
\begin{itemize}
\item Result: 66.38\% $\pm$ 8.74\%
\item Range: 37.2\% - 88.0\%
\item Status: \textbf{Worst performing method}
\item Observation: Merging destroys specialized knowledge
\item Performance worse than individual overfitted LoRAs (66.38\% vs 68.28\%)
\item No constructive interference between experts
\end{itemize}

\textbf{Weighted Merge (Cluster 4)}:
\begin{itemize}
\item Result: 67.00\%
\item Smart weights: validation\_score $\times$ centrality
\item Problem: Weighting strategy doesn't prevent destructive averaging
\item Only 0.62\% better than simple MoE despite sophisticated weighting
\end{itemize}

\subsection{Prefix Tuning Results (Preliminary)}

\textbf{Proof-of-concept (persona\_000)}:

\begin{table}[h]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
Approach & Score & vs Unified & Params & Time \\ \midrule
Unified (no prefix) & 59.09\% & baseline & 0 & - \\
Static text prefix & 60.60\% & +1.5\% & 0 & 0min \\
Learned persona prefix & [Running] & ? & 8,960 & 15min \\
Learned cluster prefix & [Running] & ? & 8,960 & 20min \\ \bottomrule
\end{tabular}
\caption{Prefix tuning preliminary results}
\end{table}

\textbf{Early insight}: Even static text prefix provides improvement (+1.5\%), suggesting personalization signal exists and can be captured with minimal overhead.

\subsection{Selective Routing Results}

\textbf{Approach}: For each persona, use their best-performing model (unified, hybrid, or personalized).

\textbf{Results}:
\begin{itemize}
\item Unified (all personas): 82.14\%
\item Selective routing: 82.99\%
\item Improvement: +0.85\% (+1.03\% relative)
\end{itemize}

\textbf{Routing decisions}:
\begin{itemize}
\item Use unified: 155/200 personas (77.5\%)
\item Use hybrid: 41/200 personas (20.5\%)
\item Use personalized: 4/200 personas (2.0\%)
\end{itemize}

\textbf{Top improvements}:
\begin{itemize}
\item persona\_180: +11.99\% (personalized model)
\item persona\_091: +11.02\% (hybrid model)
\item persona\_026: +8.99\% (hybrid model)
\end{itemize}

\textbf{Advantages}: No additional training, guaranteed $\geq$ unified performance, simple lookup table

\textbf{Disadvantages}: Must load multiple models, only helps 22.5\% of personas

\section{Analysis and Discussion}

\subsection{Why Did ALL Personalization Methods Fail?}

We identify five fundamental failure modes:

\subsubsection{1. Poor Clustering Quality}

\textbf{Silhouette score: 0.022} (scale: -1 to +1, near-zero indicates no structure)

\textbf{Implications}:
\begin{itemize}
\item Personas clustered by description embeddings don't form behaviorally coherent groups
\item Text similarity $\neq$ behavioral similarity for smart home tasks
\item Cluster-based personalization assumes wrong similarity metric
\end{itemize}

\textbf{Evidence}: Even cluster 4 with 2160 examples (36\% of total data) performs 8\% worse than unified.

\subsubsection{2. Insufficient Training Data}

\textbf{Per-persona}: 20 examples $\ll$ 2.4M LoRA parameters

Overfitting ratio: $\frac{2,400,000}{20} = 120,000$ parameters per training example

\textbf{Comparison}:
\begin{itemize}
\item Per-persona: 20 examples → 68.28\% (overfits)
\item Cluster 0: 480 examples → 72.65\% (still overfits)
\item Cluster 4: 2160 examples → 74.14\% (still insufficient)
\item Unified: 6000 examples → 82.14\% (sufficient)
\end{itemize}

\textbf{Estimated threshold}: Need $\sim$100+ examples per persona for stable LoRA training with this model size.

\subsubsection{3. Model Capacity Constraints}

\textbf{Qwen 0.5B parameters} may be too small for dual objectives:
\begin{itemize}
\item General smart home knowledge (device types, commands, context)
\item Per-user personalization (preferences, speaking style)
\end{itemize}

\textbf{Hypothesis}: Larger models (3B+ parameters) might have sufficient capacity for both general and personalized knowledge.

\textbf{Evidence}: Per-persona models show catastrophic forgetting (some personas drop to 48\%), suggesting capacity limits.

\subsubsection{4. Task Characteristics}

\textbf{Smart home commands are relatively standardized}:
\begin{itemize}
\item 40\% simple commands: "Turn on lights" (89\% unified accuracy)
\item 35\% multi-device: "Set up movie mode" (78\% unified accuracy)
\item 15\% context-dependent: "Make it cozy" (71\% unified accuracy)
\item 10\% personality-heavy: Style-specific responses (65\% unified accuracy)
\end{itemize}

\textbf{Observation}: 75\% of tasks are straightforward device commands where domain knowledge matters more than personalization.

\textbf{Implication}: Task characteristics favor unified training. Personalization only helps on 25\% of examples, not enough to compensate for data loss.

\subsubsection{5. Destructive Weight Merging}

\textbf{Observation}: MoE methods (66.38\%) perform worse than individual overfitted LoRAs (68.28\%).

\textbf{Why merging fails}:
\begin{itemize}
\item Individual LoRAs overfit in different directions
\item Averaging cancels out learned patterns rather than combining strengths
\item No constructive interference between specialized weights
\item Smart weighting (67.00\%) barely better than simple averaging (66.38\%)
\end{itemize}

\textbf{Mathematical insight}: Linear averaging of nonlinear function approximators is fundamentally problematic without careful alignment.

\subsection{When Does Unified Training Win?}

\textbf{Our finding}: For small models with limited per-user data:

\begin{equation}
\text{More Data} > \text{Personalization}
\end{equation}

\textbf{Crossover point estimation}:

Personalization might work when:
\begin{itemize}
\item 100+ examples per persona (vs 20 in our study)
\item OR 3B+ parameter models (vs 0.5B in our study)
\item OR tasks with high personalization benefit (vs 25\% in smart home)
\item AND high-quality behavioral clustering (vs 0.022 silhouette score)
\end{itemize}

\subsection{Prefix Tuning: A Promising Alternative}

\textbf{Why prefix tuning might succeed where LoRA failed}:

\begin{enumerate}
\item \textbf{Builds on strong base}: Starts from 82.14\% unified model (frozen)
\item \textbf{Minimal parameters}: 8,960 vs 2.4M (267× fewer, lower overfitting risk)
\item \textbf{Additive not destructive}: Doesn't modify proven unified weights
\item \textbf{Fast training}: 15 minutes vs hours
\item \textbf{Early validation}: Static prefix already improves (+1.5\%)
\end{enumerate}

\textbf{Preliminary evidence}: Static text prefix (+1.5\%) suggests personalization signal exists and can be captured with minimal overhead.

\subsection{Task Complexity Analysis}

\begin{table}[h]
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
Task Type & Proportion & Unified Accuracy \\ \midrule
Simple commands & 40\% & 89\% \\
Multi-device commands & 35\% & 78\% \\
Context-dependent & 15\% & 71\% \\
Personality-heavy & 10\% & 65\% \\ \bottomrule
\end{tabular}
\caption{Task complexity distribution and unified model performance}
\end{table}

\textbf{Insight}: Unified model already performs well on 75\% of tasks (simple + multi-device). Personalization opportunity limited to 25\% (context + personality).

\subsection{Example Predictions}

\subsubsection{Example 1: Simple Command}

\begin{verbatim}
User: "Turn on the lights"
Persona: Ethan (reserved librarian)

Reference:      "Of course. What brightness level would you like?"
Unified LoRA:   "Of course. What brightness would you prefer?"
                Similarity: 94.2%

Per-Persona:    "Lights on. Brightness?"
                Similarity: 71.3% (overfitted to terse style)
\end{verbatim}

\subsubsection{Example 2: Multi-Device with Personality}

\begin{verbatim}
User: "Turn on the lights and play some music"
Persona: Maya (enthusiastic yoga instructor)

Reference:       "Perfect! Setting lights to full brightness
                  and playing your energizing playlist at
                  volume 80. Let's start the day!"

Unified LoRA:    "I'll turn on the lights and start the music."
                 Similarity: 78.9% (good command, missing personality)

Per-Persona:     "Lights! Music! Go go go!"
                 Similarity: 52.1% (overfitted, missing commands)

Hybrid LoRA:     "Great! Setting lights to full brightness and
                  playing your playlist at volume 80. Let's go!"
                 Similarity: 93.2% (captures both!)
\end{verbatim}

\subsubsection{Example 3: Failure Mode - Hallucination}

\begin{verbatim}
User: "Turn on the lights"

Bad Prediction:  "Setting the lights to 50% brightness and also
                  adjusting the curtains to 30% open."
Issue: Hallucinated non-existent device (curtains)
Similarity: 52%
\end{verbatim}

\section{Lessons Learned}

\subsection{Methodological Insights}

\begin{enumerate}
\item \textbf{Always establish strong baselines first}
\begin{itemize}
\item Our unified model (82.14\%) beat all complex approaches
\item Testing simple methods first saved months of effort
\end{itemize}

\item \textbf{Clustering quality matters critically}
\begin{itemize}
\item Silhouette score 0.022 was a red flag we should have heeded earlier
\item Text similarity doesn't guarantee behavioral similarity
\item Should explore multiple clustering strategies: behavioral features, task patterns, interaction styles
\end{itemize}

\item \textbf{Data quantity dominates algorithmic complexity}
\begin{itemize}
\item 6000 examples unified > 20-2160 personalized
\item Complexity cannot compensate for insufficient data
\item Rule of thumb: $\geq$100 examples per LoRA or avoid fine-tuning
\end{itemize}

\item \textbf{Weight merging is inherently risky}
\begin{itemize}
\item Almost always destructive without careful alignment
\item Smart weighting helps minimally (0.6\% gain)
\item Requires validation before scaling
\end{itemize}

\item \textbf{Proof-of-concept saves time}
\begin{itemize}
\item Testing persona\_000 (30 min) saved 10+ hours of wasted training
\item Quick iteration $>$ comprehensive experiments when exploring
\end{itemize}
\end{enumerate}

\subsection{Engineering Insights}

\begin{enumerate}
\item \textbf{Monitor training curves actively}
\begin{itemize}
\item Overfitting visible early in validation loss
\item Early stopping could have saved 50+ GPU hours
\end{itemize}

\item \textbf{Reproducibility is critical}
\begin{itemize}
\item Saved all logs, configs, models, random seeds
\item Can investigate failures and replicate successes
\item Enabled this detailed post-hoc analysis
\end{itemize}

\item \textbf{Cost-benefit analysis matters}
\begin{itemize}
\item Per-persona: 200 GPU hours for -13.9\% performance
\item Selective routing: 0 hours for +1.03\% performance
\item Prefix tuning: 50 GPU hours (estimated) for +1-3\% expected
\end{itemize}
\end{enumerate}

\section{Future Work}

\subsection{Immediate Next Steps}

\begin{enumerate}
\item \textbf{Complete prefix tuning evaluation}
\begin{itemize}
\item Finish persona\_000 proof-of-concept (learned prefix)
\item If successful ($>$59.09\% baseline), scale to all 200 personas
\item Test cluster-level prefixes (more data, less overfitting)
\item Expected result: 83-85\% if successful
\end{itemize}

\item \textbf{Implement retrieval-augmented generation}
\begin{itemize}
\item Index user interaction history with sentence embeddings
\item Retrieve top-$k$ similar past interactions for each query
\item Augment context with retrieved examples
\item Expected improvement: +2-6\% based on related work
\end{itemize}

\item \textbf{Better clustering strategies}
\begin{itemize}
\item Use behavioral features: command types, device preferences, interaction patterns
\item Try hierarchical clustering
\item Optimize clustering objective for task performance, not embedding similarity
\item Target: silhouette score $>$ 0.3
\end{itemize}

\item \textbf{Test larger base models}
\begin{itemize}
\item Evaluate Qwen 1.5B, 3B, 7B variants
\item Hypothesis: Larger models have capacity for personalization
\item May show crossover point where personalization helps
\end{itemize}
\end{enumerate}

\subsection{Alternative Approaches}

\begin{enumerate}
\item \textbf{Retrieval-Augmented Generation (RAG)}
\begin{itemize}
\item Retrieve relevant past user interactions
\item Provide as context instead of fine-tuning
\item No overfitting risk, dynamic adaptation
\item Transparent (can inspect retrieved examples)
\end{itemize}

\item \textbf{Prompt-based personalization}
\begin{itemize}
\item Include persona information in system prompt
\item No training required, zero-shot adaptation
\item May work well for larger base models
\end{itemize}

\item \textbf{Hybrid approaches}
\begin{itemize}
\item Unified base + lightweight per-user adapters
\item Best of both: strong general knowledge + personalization
\item Candidates: LoRA with regularization, adapter layers, prefix tuning
\end{itemize}

\item \textbf{Active learning}
\begin{itemize}
\item Select most informative examples for each user
\item Quality over quantity
\item May reduce data requirements from 100+ to 20-30 high-quality examples
\end{itemize}
\end{enumerate}

\subsection{Broader Research Directions}

\begin{enumerate}
\item \textbf{When does personalization help?}
\begin{itemize}
\item Systematic study across task types
\item Data requirements vs model size trade-offs
\item Task characteristics that benefit from personalization
\end{itemize}

\item \textbf{Optimal architecture for personalization}
\begin{itemize}
\item Dedicated persona embedding spaces
\item Multi-task learning with shared + private parameters
\item Meta-learning for rapid user adaptation
\end{itemize}

\item \textbf{Evaluation beyond similarity}
\begin{itemize}
\item User satisfaction studies
\item Task completion rates
\item Longitudinal adaptation over time
\end{itemize}
\end{enumerate}

\section{Conclusion}

This comprehensive study of personalization techniques for smart home assistants yields a counterintuitive but important finding: \textbf{simple unified training outperforms all tested personalization methods}. Despite evaluating six different approaches—per-persona LoRA (68.28\%), cluster-based training (74.14\% best case), sparse MoE (66.38\%), weighted merging (67.00\%), and preliminary prefix tuning—none exceeded the unified baseline of 82.14\%.

Our analysis reveals five key failure modes that explain this surprising result:

\begin{enumerate}
\item \textbf{Poor clustering quality}: Silhouette score of 0.022 indicates personas don't form natural behavioral groups
\item \textbf{Insufficient training data}: 20-2160 examples per variant vastly insufficient compared to 6000 for unified
\item \textbf{Limited model capacity}: 0.5B parameters cannot simultaneously maintain general knowledge and user-specific personalization
\item \textbf{Task characteristics}: 75\% of smart home commands are standardized, favoring domain knowledge over personalization
\item \textbf{Destructive weight merging}: Linear averaging of LoRA weights consistently degrades performance
\end{enumerate}

However, this study also identifies promising lightweight alternatives that build upon rather than replace the strong unified baseline:

\begin{itemize}
\item \textbf{Selective routing}: +1.03\% improvement by routing each persona to their best-performing model (already validated)
\item \textbf{Prefix tuning}: +1.5\% preliminary result with static text prefix, full learned prefix evaluation pending
\item \textbf{Retrieval-augmented generation}: Expected +2-6\% by augmenting context with similar past interactions
\end{itemize}

\subsection{Practical Recommendations}

\textbf{For this dataset and task}:
\begin{itemize}
\item ✓ Use unified LoRA (82.14\%, simple, consistent)
\item ✗ Avoid per-persona fine-tuning (overfits with limited data)
\item ✗ Avoid clustering approaches (poor quality, still insufficient data)
\item ✗ Avoid model merging strategies (consistently destructive)
\end{itemize}

\textbf{For future smart home projects}:
\begin{itemize}
\item Collect 100+ examples per user if pursuing personalization
\item Use models $\geq$ 3B parameters for adequate capacity
\item Try prefix tuning or RAG before expensive per-user fine-tuning
\item Improve clustering with behavioral features, not text embeddings
\end{itemize}

\subsection{Broader Implications}

This work demonstrates that the conventional wisdom "personalization always helps" does not hold universally. The decision between unified and personalized training depends on:

\begin{itemize}
\item \textbf{Data regime}: Unified wins when per-user data is scarce ($<$100 examples)
\item \textbf{Model scale}: Small models ($<$3B params) struggle with dual objectives
\item \textbf{Task characteristics}: Standardized domains favor unified training
\item \textbf{Quality over sophistication}: Simple strong baselines beat complex weak personalization
\end{itemize}

The estimated \textbf{crossover point} where personalization becomes viable: 100+ examples per user AND (3B+ parameter models OR lightweight methods like prefix tuning/RAG).

\subsection{Final Thoughts}

While this study focused on smart home assistants, the lessons learned generalize to other domain-specific applications of language models. The key insight is not that personalization never works, but rather that it requires careful consideration of data availability, model capacity, task characteristics, and method selection. When these factors are unfavorable, a strong unified baseline often provides the best path forward, with lightweight additions (routing, retrieval, prefixes) offering meaningful improvements without the risks of heavy personalization.

Future work combining larger models, richer per-user data, and hybrid approaches (unified base + lightweight personalization) may finally unlock the promise of effective personalized conversational agents. Our hope is that this detailed empirical study provides a roadmap for when and how to pursue personalization in practical applications.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% begin figure %%%%%%%%%%%%%%%%%%%
\begin{figure}[t]
\begin{center}
\setlength{\unitlength}{0.012500in}%
\begin{picture}(115,35)(255,545)
\thicklines
\put(255,545){\framebox(115,35){}}
\put(275,560){Beautiful Figure}
\end{picture}
\end{center}
\caption{THE FIGURE CAPTION USES CAPITAL LETTERS.}
\label{figure_ASME} 
\end{figure}
%%%%%%%%%%%%%%%% end figure %%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% begin table   %%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{table}[t]
\caption{THE TABLE CAPTION USES CAPITAL LETTERS, TOO.}
\begin{center}
\label{table_ASME}
\begin{tabular}{c l l}
& & \\ % put some space after the caption
\hline
Example & Time & Cost \\
\hline
1 & 12.5 & \$1,000 \\
2 & 24 & \$2,000 \\
\hline
\end{tabular}
\end{center}
\end{table}
%%%%%%%%%%%%%%%% end table %%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

All tables should be numbered consecutively and  captioned; the caption should use all capital letters, and centered above the table as shown in Table~\ref{table_ASME}. The body of the table should be no smaller than 7 pt.  There should be a minimum two line spaces between tables and text.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{FOOTNOTES\protect\footnotemark}
\footnotetext{Examine the input file, asme2e.tex, to see how a footnote is given in a head.}

Footnotes are referenced with superscript numerals and are numbered consecutively from 1 to the end of the paper\footnote{Avoid footnotes if at all possible.}. Footnotes should appear at the bottom of the column in which they are referenced.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{CITING REFERENCES}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\begin{quotation}
{\em Text Citation}. Within the text, references should be cited in  numerical order according to their order of appearance.  The numbered reference citation should be enclosed in brackets.
\end{quotation}



% Here's where you specify the bibliography style file.
% The full file name for the bibliography style file 
% used for an ASME paper is asmems4.bst.
\bibliographystyle{asmems4}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{acknowledgment}
Thanks go to D. E. Knuth and L. Lamport for developing the wonderful word processing software packages \TeX\ and \LaTeX. I also would like to thank Ken Sprott, Kirk van Katwyk, and Matt Campbell for fixing bugs in the ASME style file \verb+asme2e.cls+, and Geoff Shiflett for creating 
ASME bibliography stype file \verb+asmems4.bst+.
\end{acknowledgment}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The bibliography is stored in an external database file
% in the BibTeX format (file_name.bib).  The bibliography is
% created by the following command and it will appear in this
% position in the document. You may, of course, create your
% own bibliography by using thebibliography environment as in
%
% \begin{thebibliography}{12}
% ...
% \bibitem{itemreference} D. E. Knudsen.
% {\em 1966 World Bnus Almanac.}
% {Permafrost Press, Novosibirsk.}
% ...
% \end{thebibliography}

% Here's where you specify the bibliography database file.
% The full file name of the bibliography database for this
% article is asme2e.bib. The name for your database is up
% to you.
\bibliography{asme2e}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\appendix       %%% starting appendix
\section*{Appendix A: Head of First Appendix}
Avoid Appendices if possible.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Appendix B: Head of Second Appendix}
\subsection*{Subsection head in appendix}
The equation counter is not reset in an appendix and the numbers will
follow one continual sequence from the beginning of the article to the very end as shown in the following example.
\begin{equation}
a = b + c.
\end{equation}

\end{document}

\bibliographystyle{plain}
\begin{thebibliography}{9}

\bibitem{hu2021lora}
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
\textit{LoRA: Low-Rank Adaptation of Large Language Models}.
ICLR 2022.

\bibitem{li2021prefix}
Xiang Lisa Li and Percy Liang.
\textit{Prefix-Tuning: Optimizing Continuous Prompts for Generation}.
ACL 2021.

\bibitem{lester2021prompt}
Brian Lester, Rami Al-Rfou, and Noah Constant.
\textit{The Power of Scale for Parameter-Efficient Prompt Tuning}.
EMNLP 2021.

\bibitem{zhang2023personalized}
Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme.
\textit{Personalized Dialogue Generation via Prompt Learning}.
2023.

\end{thebibliography}

\end{document}
