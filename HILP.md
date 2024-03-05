# Understanding Hilbert Foundation Policies (HILPs) for Unsupervised Pre-Training

This paper introduces Hilbert foundation policies (HILPs), a general unsupervised pre-training objective for foundation policies that aims to capture diverse, optimal long-horizon behaviors from unlabeled data to facilitate downstream task learning. The key aspects and contributions of the method are:

- **Hilbert representations**: They first learn a geometric abstraction of the dataset by training a representation function φ:S→Z that maps states to a Hilbert space Z such that distances in Z correspond to the temporal distances between states in the original MDP. This distance-preserving mapping abstracts the state space while preserving the long-term global relationships between states.

- **Unsupervised policy training**: After obtaining the Hilbert representation φ, they train a latent-conditioned policy π(a|s,z) using offline RL to span the latent space Z with skills that correspond to directional movements. The reward function is defined as the inner product between φ(s')-φ(s) and a randomly sampled unit vector z. By learning to move in every possible direction, the policy learns diverse long-horizon behaviors that optimally span both the latent and state spaces. The resulting multi-task policy π(a|s,z) is called a Hilbert foundation policy (HILP).

- **Zero-shot prompting**: HILPs provide multiple ways to quickly adapt to downstream tasks in a zero-shot manner:
    - Zero-shot RL: Given a reward function at test time, the optimal latent vector z that maximizes it can be found via linear regression without additional training, enabled by the successor feature structure of the HILP reward function.
    - Zero-shot goal-conditioned RL: Given a target goal state, moving in the latent direction of φ(g)-φ(s) is proved to be optimal for reaching the goal if embedding errors are sufficiently small.
    - Test-time planning: The structured Hilbert representation enables efficiently finding an optimal subgoal between the current state and goal to refine the policy prompts, which can be done iteratively to further improve performance.

- **Strong empirical results**: On simulated robotic benchmarks, HILPs outperform prior methods specialized for zero-shot RL, goal-conditioned RL, and hierarchical RL, demonstrating the effectiveness of capturing state-spanning long-horizon behaviors for unsupervised pre-training.

The key insights are:
1. Capturing the long-term temporal structure of the environment in a Hilbert space enables learning diverse useful behaviors.
2. The inner product structure allows versatile prompting of the learned policy for efficient zero-shot adaptation.
3. Pre-training with a principled objective leads to a general-purpose foundation policy effective for multiple downstream settings.

The Hilbert space plays a central role in the proposed method for unsupervised pre-training of foundation policies. In this paper, the authors use a Hilbert space Z as the latent space to embed the state space S of the MDP. The key property they seek is to learn a representation function φ:S→Z that preserves the temporal structure of the MDP in the geometry of the Hilbert space. Specifically, they aim to learn φ such that the Euclidean distance between two states' embeddings in Z equals the temporal distance between them in the original MDP:
d*(s,g) = ||φ(s) - φ(g)||
where d*(s,g) denotes the minimum number of time steps needed for an optimal policy to transition from state s to g.

The motivation for using a Hilbert space representation is twofold:
1. Geometric abstraction: By embedding states into a metric space where distances capture long-term relationships, the Hilbert representation provides a meaningful abstraction of the state space that focuses on the relevant temporal structure for long-horizon behaviors. This is in contrast to the raw state space where Euclidean distances are often semantically meaningless.
2. Structured latent space: Hilbert spaces are inner product spaces, which is a desirable property for the latent space. The inner product structure enables simple and principled ways to direct the learned behaviors, by defining directional rewards and policy prompts based on inner products.

To learn the Hilbert representation, the authors leverage the equivalence between temporal distances and optimal goal-conditioned value functions: d*(s,g) = -V*(s,g), where V*(s,g) is the optimal goal-conditioned value function. They parameterize the goal-conditioned value function as V(s,g) = -||φ(s) - φ(g)|| and train it using off-policy learning on the offline dataset. This allows learning φ from the offline data without explicit temporal distance labels.

After learning φ, the next step is to train a latent-conditioned policy π(a|s,z) to maximize an inner product reward r(s,z,s') = <φ(s')-φ(s), z>, where z is a randomly sampled unit vector in Z. This reward encourages the policy to move in the direction specified by z in the latent space. By maximizing this reward for all sampled directions, the policy learns to span the latent space, capturing diverse long-horizon skills that transition between distant states.

The Hilbert space representation enables several downstream applications of the learned policy in a zero-shot manner:
- **Zero-shot RL via successor features**: For a given reward function, the optimal direction z can be found by linear regression thanks to the successor feature structure of the HILP reward function.
- **Zero-shot goal-conditioned RL**: The policy can be prompted to reach a goal state g by setting z to be the normalized direction from the current state to the goal in the latent space. The authors prove that this is optimal under assumptions on the quality of the learned representation.
- **Test-time planning**: By leveraging the structure of the Hilbert space, an efficient planning procedure can be used to refine the policy prompts, by finding intermediate subgoals based on distances in the latent space without additional learning.

In the HILP framework, the reward function used to train the latent-conditioned policy π(a|s,z) has a special structure that resembles successor features. Specifically, the reward is defined as an inner product between the difference of state embeddings and a direction vector z:
r(s,z,s') = <φ(s')-φ(s), z>
Here, the term φ(s')-φ(s) can be viewed as a cumulant in the successor feature framework. This connection allows HILPs to perform zero-shot RL via a simple method based on linear regression.

In the successor feature framework, the value function can be decomposed into two parts: the expected discounted sum of cumulants (successor features), and a linear weighting of the cumulants (reward weights). The key idea is that if the successor features are known, then the value function can be computed for any reward function by just learning the linear reward weights, without needing to estimate the full value function from scratch.

HILPs leverage this idea for zero-shot RL as follows. Let ˜φ(s,a,s') := φ(s')-φ(s) denote the cumulant. If the agent has access to an arbitrary reward function r(s,a,s') at test time, it can find the optimal direction z* in the latent space by solving a linear regression problem:
z* = arg min_z E_D [(r(s,a,s') - <˜φ(s,a,s'), z>)^2]
Intuitively, this finds the direction z* such that the HILP reward <˜φ(s,a,s'), z*> best approximates the true reward r(s,a,s') in expectation over the offline dataset D. If the latent space Z is a Euclidean space, this linear regression problem has a closed-form solution:
z* = E[˜φ˜φ^T]^(−1) E[r(s,a,s')˜φ]
where expectations are over (s,a,s') tuples from D.

The authors use this approach for zero-shot RL at test time by first sampling a small number of transitions from the dataset, then computing z* using the above closed-form solution, and finally executing the policy π(a|s,z*) conditioned on the optimal direction.

The effectiveness of this zero-shot RL method hinges on two key properties of HILPs:
1. The successor feature structure of the HILP reward function, which enables the linear regression approach.
2. The expressivity of the HILP policy in capturing diverse behaviors that can be used to maximize arbitrary reward functions.

The "principled ways" to direct learned behaviors refer to the use of the inner product structure of the Hilbert space to define meaningful rewards and policy prompts for the latent-conditioned policy. By leveraging the geometric and mathematical properties inherent to the Hilbert space, the framework provides a structured approach to directing the learned behaviors, enabling the policy to adapt and respond effectively to a wide range of tasks and challenges in a zero-shot manner.

<end>