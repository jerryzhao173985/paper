
# Understanding Hilbert Foundation Policies (HILPs) for Unsupervised Pre-Training

This paper introduces Hilbert foundation policies (HILPs), a general unsupervised pre-training objective for foundation policies that aims to capture diverse, optimal long-horizon behaviors from unlabeled data to facilitate downstream task learning. The key aspects and contributions of the method are:

- Hilbert representations: They first learn a geometric abstraction of the dataset by training a representation function `$begin:math:text$\\phi:S\\rightarrow Z$end:math:text$` that maps states to a Hilbert space `$begin:math:text$Z$end:math:text$` such that distances in `$begin:math:text$Z$end:math:text$` correspond to the temporal distances between states in the original MDP. This distance-preserving mapping abstracts the state space while preserving the long-term global relationships between states.

- Unsupervised policy training: After obtaining the Hilbert representation `$begin:math:text$\\phi$end:math:text$`, they train a latent-conditioned policy `$begin:math:text$\\pi(a|s,z)$end:math:text$` using offline RL to span the latent space `$begin:math:text$Z$end:math:text$` with skills that correspond to directional movements. The reward function is defined as the inner product between `$begin:math:text$\\phi(s')-\\phi(s)$end:math:text$` and a randomly sampled unit vector `$begin:math:text$z$end:math:text$`. By learning to move in every possible direction, the policy learns diverse long-horizon behaviors that optimally span both the latent and state spaces. The resulting multi-task policy `$begin:math:text$\\pi(a|s,z)$end:math:text$` is called a Hilbert foundation policy (HILP).

- Zero-shot prompting: HILPs provide multiple ways to quickly adapt to downstream tasks in a zero-shot manner:
    - Zero-shot RL: Given a reward function at test time, the optimal latent vector `$begin:math:text$z$end:math:text$` that maximizes it can be found via linear regression without additional training, enabled by the successor feature structure of the HILP reward function.
    - Zero-shot goal-conditioned RL: Given a target goal state, moving in the latent direction of `$begin:math:text$\\phi(g)-\\phi(s)$end:math:text$` is proved to be optimal for reaching the goal if embedding errors are sufficiently small.
    - Test-time planning: The structured Hilbert representation enables efficiently finding an optimal subgoal between the current state and goal to refine the policy prompts, which can be done iteratively to further improve performance.

- Strong empirical results: On simulated robotic benchmarks, HILPs outperform prior methods specialized for zero-shot RL, goal-conditioned RL and hierarchical RL, demonstrating the effectiveness of capturing state-spanning long-horizon behaviors for unsupervised pre-training.

The key insights are:
1. Capturing the long-term temporal structure of the environment in a Hilbert space enables learning diverse useful behaviors.
2. The inner product structure allows versatile prompting of the learned policy for efficient zero-shot adaptation.
3. Pre-training with a principled objective leads to a general-purpose foundation policy effective for multiple downstream settings.

The Hilbert space plays a central role in the proposed method for unsupervised pre-training of foundation policies. In this paper, the authors use a Hilbert space `$begin:math:text$Z$end:math:text$` as the latent space to embed the state space `$begin:math:text$S$end:math:text$` of the MDP. The key property they seek is to learn a representation function `$begin:math:text$\\phi:S\\rightarrow Z$end:math:text$` that preserves the temporal structure of the MDP in the geometry of the Hilbert space. Specifically, they aim to learn `$begin:math:text$\\phi$end:math:text$` such that the Euclidean distance between two states' embeddings in `$begin:math:text$Z$end:math:text$` equals the temporal distance between them in the original MDP:
`$begin:math:text$d^*(s,g) = ||\\phi(s) - \\phi(g)||$end:math:text$`
where `$begin:math:text$d^*(s,g)$end:math:text$` denotes the minimum number of time steps needed for an optimal policy to transition from state `$begin:math:text$s$end:math:text$` to `$begin:math:text$g$end:math:text$`.

The motivation for using a Hilbert space representation is twofold:
1. Geometric abstraction: By embedding states into a metric space where distances capture long-term relationships, the Hilbert representation provides a meaningful abstraction of the state space that focuses on the relevant temporal structure for long-horizon behaviors. This is in contrast to the raw state space where Euclidean distances are often semantically meaningless.
2. Structured latent space: Hilbert spaces are inner product spaces, which is a desirable property for the latent space. The inner product structure enables simple and principled ways to direct the learned behaviors, by defining directional rewards and policy prompts based on inner products.

To learn the Hilbert representation, the authors leverage the equivalence between temporal distances and optimal goal-conditioned value functions: `\(d^*(s,g) = -V^*(