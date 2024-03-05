
# Understanding Hilbert Foundation Policies (HILPs) for Unsupervised Pre-Training

This paper introduces **Hilbert Foundation Policies (HILPs)**, a novel approach to unsupervised pre-training of foundation policies aiming to capture diverse, optimal long-horizon behaviors from unlabeled data for enhanced downstream task learning.

## Key Contributions
- **Hilbert Representations**: Initiate by learning a geometric abstraction of the dataset. A representation function <img src="https://latex.codecogs.com/svg.image?\phi:S\rightarrow&space;Z" title="phi" /> maps states to a Hilbert space <img src="https://latex.codecogs.com/svg.image?Z" title="Z" />, ensuring distances in <img src="https://latex.codecogs.com/svg.image?Z" title="Z" /> correspond to temporal distances in the original Markov Decision Process (MDP), preserving long-term state relationships.

- **Unsupervised Policy Training**: Following the establishment of Hilbert representation <img src="https://latex.codecogs.com/svg.image?\phi" title="phi" />, a latent-conditioned policy <img src="https://latex.codecogs.com/svg.image?\pi(a|s,z)" title="pi(a|s,z)" /> is trained via offline Reinforcement Learning (RL) to navigate the latent space <img src="https://latex.codecogs.com/svg.image?Z" title="Z" /> with directionally diverse skills, culminating in the Hilbert Foundation Policy (HILP).

- **Zero-shot Prompting**: Demonstrates rapid adaptability to downstream tasks without additional training through:
  - Zero-shot RL: Given a test-time reward function, identifies the optimal latent vector <img src="https://latex.codecogs.com/svg.image?z" title="z" /> via linear regression, facilitated by the HILP reward function's successor feature structure.
  - Zero-shot goal-conditioned RL and Test-time Planning: Offers methodologies for achieving specified goals or refining policy prompts leveraging the structured Hilbert space for optimal subgoal identification.

- **Strong Empirical Results**: Validates HILPs' superiority in capturing extensive long-horizon behaviors compared to existing methods in simulated robotic benchmarks.

## Key Insights

1. **Long-term Temporal Structure Capture**: Achieved by embedding environmental dynamics in a Hilbert space, enabling the learning of diverse, beneficial behaviors.
2. **Versatile Zero-shot Policy Prompting**: Utilizes the inner product structure of the Hilbert space for efficient policy adaptation to new tasks.
3. **Principled Objective Pre-training**: Establishes a foundation for a universally applicable policy effective across various downstream applications.

## Hilbert Space in Unsupervised Pre-training

At the core of HILPs is the utilization of a Hilbert space <img src="https://latex.codecogs.com/svg.image?Z" title="Z" /> as the latent embedding space for the state space <img src="https://latex.codecogs.com/svg.image?S" title="S" /> of the MDP, with the prime objective being the preservation of the MDP's temporal structure within this geometric framework:

- **Euclidean Distance Representation**: The aim is for the Euclidean distance between any two state embeddings in <img src="https://latex.codecogs.com/svg.image?Z" title="Z" /> to mirror the temporal distance between them in the MDP, formalized as <img src="https://latex.codecogs.com/svg.image?d^*(s,g)&space;=&space;\|\phi(s)&space;-&space;\phi(g)\|" title="d^*(s,g)" />, where <img src="https://latex.codecogs.com/svg.image?d^*(s,g)" title="d^*(s,g)" /> denotes the optimal transition steps from state <img src="https://latex.codecogs.com/svg.image?s" title="s" /> to <img src="https://latex.codecogs.com/svg.image?g" title="g" />.

### Motivation Behind Hilbert Space Representation

1. **Geometric Abstraction**: Offers a meaningful state space abstraction by emphasizing long-term relationships over raw state metrics.
2. **Structured Latent Space**: Facilitates behavior direction through inner product space properties, enabling simple, yet effective, learning and policy direction strategies.

### Learning and Application of HILPs

Post representation learning, the focus shifts to maximizing an inner product-based reward function for policy training, allowing for extensive latent space navigation and skill acquisition. This foundational groundwork permits zero-shot application of learned policies across various scenarios, showcasing HILPs' flexibility and effectiveness.
