Formulas and expressions
- The representation function `φ:S→Z` maps states to a Hilbert space.
- The latent-conditioned policy `π(a|s,z)` outputs an action given a state and latent vector.
- The reward function is defined by the inner product `r(s,z,s') = <φ(s')-φ(s), z>`.
- To find the optimal latent direction, we minimize the expected difference `z* = arg min_z E_D [(r(s,a,s') - <φ(s')-φ(s), z>)^2]`.
- The linear regression solution for `z*` can be expressed as `z* = (E[φφ^T])^(-1) E[r(s,a,s')φ]`.
- The goal-conditioned prompt for zero-shot RL is given by `z* = (φ(g)-φ(s)) / ||φ(g)-φ(s)||`.


This paper introduces Hilbert foundation policies (HILPs), a general unsupervised pre-training objective for foundation policies that aims to capture diverse, optimal long-horizon behaviors from unlabeled data to facilitate downstream task learning. The key aspects and contributions of the method are:
- **Hilbert representations**: They first learn a geometric abstraction of the dataset by training a representation function \( \phi:S \rightarrow Z \) that maps states to a Hilbert space \( Z \) such that distances in \( Z \) correspond to the temporal distances between states in the original MDP. This distance-preserving mapping abstracts the state space while preserving the long-term global relationships between states.
- **Unsupervised policy training**: After obtaining the Hilbert representation \( \phi \), they train a latent-conditioned policy \( \pi(a|s,z) \) using offline RL to span the latent space \( Z \) with skills that correspond to directional movements. The reward function is defined as the inner product between \( \phi(s')-\phi(s) \) and a randomly sampled unit vector \( z \). By learning to move in every possible direction, the policy learns diverse long-horizon behaviors that optimally span both the latent and state spaces. The resulting multi-task policy \( \pi(a|s,z) \) is called a Hilbert foundation policy (HILP).
- **Zero-shot prompting**: HILPs provide multiple ways to quickly adapt to downstream tasks in a zero-shot manner:
  - Zero-shot RL: Given a reward function at test time, the optimal latent vector \( z \) that maximizes it can be found via linear regression without additional training, enabled by the successor feature structure of the HILP reward function.
  - Zero-shot goal-conditioned RL: Given a target goal state, moving in the latent direction of \( \phi(g)-\phi(s) \) is proved to be optimal for reaching the goal if embedding errors are sufficiently small.
  - Test-time planning: The structured Hilbert representation enables efficiently finding an optimal subgoal between the current state and goal to refine the policy prompts, which can be done iteratively to further improve performance.
- **Strong empirical results**: On simulated robotic benchmarks, HILPs outperform prior methods specialized for zero-shot RL, goal-conditioned RL, and hierarchical RL, demonstrating the effectiveness of capturing state-spanning long-horizon behaviors for unsupervised pre-training.
The key insights are:
1. Capturing the long-term temporal structure of the environment in a Hilbert space enables learning diverse useful behaviors.
2. The inner product structure allows versatile prompting of the learned policy for efficient zero-shot adaptation.
3. Pre-training with a principled objective leads to a general-purpose foundation policy effective for multiple downstream settings.

The Hilbert space plays a central role in the proposed method for unsupervised pre-training of foundation policies. In this paper, the authors use a Hilbert space \( Z \) as the latent space to embed the state space \( S \) of the MDP. The key property they seek is to learn a representation function \( \phi:S \rightarrow Z \) that preserves the temporal structure of the MDP in the geometry of the Hilbert space. Specifically, they aim to learn \( \phi \) such that the Euclidean distance between two states' embeddings in \( Z \) equals the temporal distance between them in the original MDP:
\[ d^*(s,g) = \|\phi(s) - \phi(g)\| \]
where \( d^*(s,g) \) denotes the minimum number of time steps needed for an optimal policy to transition from state \( s \) to \( g \).
The motivation for using a Hilbert space representation is twofold:
1. **Geometric abstraction**: By embedding states into a metric space where distances capture long-term relationships, the Hilbert representation provides a meaningful abstraction of the state space that focuses on the relevant temporal structure for long-horizon behaviors. This is in contrast to the raw state space where Euclidean distances are often semantically meaningless.
2. **Structured latent space**: Hilbert spaces are inner product spaces, which is a desirable property for the latent space. The inner product structure enables simple and principled ways to direct the learned behaviors, by defining directional rewards and policy prompts based on inner products.
To learn the Hilbert representation, the authors leverage the equivalence between temporal distances and optimal goal-conditioned value functions: \( d^*(s,g) = -V^*(s,g) \), where \( V^*(s,g) \) is the optimal goal-conditioned value function. They parameterize the goal-conditioned value function as \( V(s,g) = -\|\phi(s) - \phi(g)\| \) and train it using off-policy learning on the offline dataset. This allows learning \( \phi \) from the offline data without explicit temporal distance labels.
After learning \( \phi \), the next step is to train a latent-conditioned policy \( \pi(a|s,z) \) to maximize an inner product reward \( r(s,z,s') = \langle \phi(s')-\phi(s), z \rangle \), where \( z \) is a randomly sampled unit vector in \( Z \). This reward encourages the policy to move in the direction specified by \( z \) in the latent space. By maximizing this reward for all sampled directions, the policy learns to span the latent space, capturing diverse long-horizon skills that transition between distant states.

The Hilbert space representation enables several downstream applications of the learned policy in a zero-shot manner:
1. **Zero-shot RL via successor features**: For a given reward function, the optimal direction \( z \) can be found by linear regression thanks to the successor feature structure of the HILP reward function.
2. **Zero-shot goal-conditioned RL**: The policy can be prompted to reach a goal state \( g \) by setting \( z \) to be the normalized direction from the current state to the goal in the latent space. The authors prove that this is optimal under assumptions on the quality of the learned representation.
3. **Test-time planning**: By leveraging the structure of the Hilbert space, an efficient planning procedure can be used to refine the policy prompts, by finding intermediate subgoals based on distances in the latent space without additional learning.

In the HILP framework, the reward function used to train the latent-conditioned policy \( \pi(a|s,z) \) has a special structure that resembles successor features. Specifically, the reward is defined as an inner product between the difference of state embeddings and a direction vector \( z \):
\[ r(s,z,s') = \langle \phi(s')-\phi(s), z \rangle \]
Here, the term \( \phi(s')-\phi(s) \) can be viewed as a cumulant in the successor feature framework. This connection allows HILPs to perform zero-shot RL via a simple method based on linear regression.

In the successor feature framework, the value function can be decomposed into two parts: the expected discounted sum of cumulants (successor features), and a linear weighting of the cumulants (reward weights). The key idea is that if the successor features are known, then the value function can be computed for any reward function by just learning the linear reward weights, without needing to estimate the full value function from scratch.

HILPs leverage this idea for zero-shot RL as follows. Let \( \tilde{\phi}(s,a,s') := \phi(s')-\phi(s) \) denote the cumulant. If the agent has access to an arbitrary reward function \( r(s,a,s') \) at test time, it can find the optimal direction \( z^* \) in the latent space by solving a linear regression problem:
\[ z^* = \arg\min_z E_D \left[(r(s,a,s') - \langle \tilde{\phi}(s,a,s'), z \rangle)^2\right] \]
Intuitively, this finds the direction \( z^* \) such that the HILP reward \( \langle \tilde{\phi}(s,a,s'), z^* \rangle \) best approximates the true reward \( r(s,a,s') \) in expectation over the offline dataset \( D \). If the latent space \( Z \) is a Euclidean space, this linear regression problem has a closed-form solution:
\[ z^* = \left(E[\tilde{\phi}\tilde{\phi}^T]^{-1} E[r(s,a,s')\tilde{\phi}]\right) \]
where expectations are over \( (s,a,s') \) tuples from \( D \).

The authors use this approach for zero-shot RL at test time by first sampling a small number of transitions from the dataset, then computing \( z^* \) using the above closed-form solution, and finally executing the policy \( \pi(a|s,z^*) \) conditioned on the optimal direction.

The effectiveness of this zero-shot RL method hinges on two key properties of HILPs:
1. The successor feature structure of the HILP reward function, which enables the linear regression approach.
2. The expressivity of the HILP policy in capturing diverse behaviors that can be used to maximize arbitrary reward functions.

The theoretical justification for this approach is based on the idea that if the HILP reward \( \langle \tilde{\phi}(s,a,s'), z \rangle \) can closely approximate the true reward \( r(s,a,s') \) for some direction \( z \), then executing the policy \( \pi(a|s,z) \) should lead to good performance on the true reward optimization problem.

The "principled ways" to direct learned behaviors refer to the use of the inner product structure of the Hilbert space to define meaningful rewards and policy prompts for the latent-conditioned policy. Let's break this down:

1. **Directional rewards**: The HILP reward function is defined as an inner product between the difference of state embeddings and a direction vector \(z\):
\[ r(s,z,s') = \langle \phi(s')-\phi(s), z \rangle \]
This reward is "directional" because it depends on the direction of the vector \(z\) in the latent space. Maximizing this reward encourages the policy to move in the direction specified by \(z\). The intuition is that by learning to move in all possible directions, the policy will learn to traverse the latent space in a way that captures diverse and meaningful transitions in the original state space.

2. **Policy prompts**: In the HILP framework, "policy prompts" refer to the use of the latent direction vector \(z\) to condition the policy \(\pi(a|s,z)\) at test time. The idea is that by specifying different \(z\) vectors, we can "prompt" the policy to exhibit different behaviors, without needing to retrain it. This is used in two ways:
    a. For zero-shot goal-conditioned RL, the policy is prompted with \(z^* = \frac{\phi(g)-\phi(s)}{\|\phi(g)-\phi(s)\|}\), which is the normalized direction from the current state \(s\) to the goal state \(g\) in the latent space. The authors prove that this prompt is optimal for reaching the goal, under assumptions on the quality of the learned Hilbert space representation.
    b. For zero-shot RL with a novel reward function, the policy is prompted with \(z^* = \arg\min_z E[(r(s,a,s') - \langle \phi(s')-\phi(s), z \rangle)^2]\), which is the direction that makes the HILP reward best approximate the true reward in expectation over the dataset. This allows the policy to adapt to new rewards without retraining.

3. **Why inner products work**: The inner product (or dot product) between two vectors measures their alignment: it is positive when the vectors point in similar directions, negative when they point in opposite directions, and zero when they are orthogonal. In the HILP framework, the inner product \(\langle \phi(s')-\phi(s), z \rangle\) measures how much the transition from \(s\) to \(s'\) aligns with the direction \(z\) in the latent space.

By defining the reward based on this inner product, the agent is incentivized to take transitions that align with the specified direction \(z\). This provides a geometrically meaningful way to direct the agent's behavior in the latent space. Moreover, the inner product provides a natural way to measure the compatibility between a transition \((s,s')\) and a target direction \(z\), which is useful for defining policy prompts. For example, the goal-conditioned prompt \(z^* = \frac{\phi(g)-\phi(s)}{\|\phi(g)-\phi(s)\|}\) maximizes the inner product \(\langle \phi(s')-\phi(s), z^* \rangle\), which means it selects the direction that best aligns with the transition from the current state to the goal state.

In summary, the Hilbert space representation provides a structured latent space where directions have a geometric meaning related to the temporal structure of the MDP. This allows defining directional rewards and policy prompts based on inner products, which provide a principled way to direct the learned behaviors. The inner product is a mathematically natural way to measure the alignment between transitions and target directions in the latent space, making it a useful tool for specifying rewards and prompts that lead to meaningful behaviors in the state space.

Formulas revisit

Representation function: φ:S→Z
	•	Describes a function that maps states in the state space S to a Hilbert space Z.
Latent-conditioned policy: π(a|s,z)
	•	Defines a policy that given a state s and a latent vector z, outputs an action a.
Inner product reward function: r(s,z,s’) = <φ(s’)-φ(s), z>
	•	This formula calculates the reward based on the inner product between the difference in state representations and a latent vector z.
Zero-shot RL optimization: z* = arg min_z E_D [(r(s,a,s’) - <φ(s’)-φ(s), z>)^2]
	•	Finds the optimal latent direction z* that minimizes the difference between the expected reward and the inner product-based reward in the dataset D.
Linear regression solution for z*: z* = (E[φφ^T])^(-1) E[r(s,a,s’)φ]
	•	Provides a closed-form solution for calculating z* based on expected values over the dataset D.
Goal-conditioned prompt for zero-shot RL: z* = (φ(g)-φ(s)) / ||φ(g)-φ(s)||
	•	Calculates the normalized direction from the current state s to the goal state g in the latent space for zero-shot goal-conditioned RL.
