# Reading list

Papers and data collected while scoping Brownout (moved from Tier1.txt, Tier2.txt, data.txt).

## Tier 1

- https://arxiv.org/abs/2201.01811 - causual sim 
- https://arxiv.org/abs/2207.06272 - hindight learning for EXO -MDPs
- https://www.microsoft.com/en-us/research/publication/gray-failure-achilles-heel-cloud-scale-systems/Gray Failure: The Achilles’ Heel of Cloud-Scale Systems
- https://people.cs.uchicago.edu/~junchenj/docs/sayer.pdf -> Sayer: Using Implicit Feedback to Improve System Policies
- https://www.semanticscholar.org/paper/Harvesting-Randomness-to-Optimize-Distributed-L%C3%A9cuyer-Lockerman/6160e79c5fadf2bcfa30008f1cbd95a1f39c1b5d -< harvesting randomness to opimize distrbuted systems 
- https://arxiv.org/abs/1003.5956 -> Unbiased Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms


- https://arxiv.org/abs/2208.12596 -> answering Causual queries from Video streaming Traces 

## Tier 2

- https://arxiv.org/abs/2409.14557 -> Exploiting Exogenous Structure for Sample-Efficient Reinforcement Learning
- https://proceedings.mlr.press/v70/hanna17a.html -> Data-Efficient Policy Evaluation Through Behavior Policy Search
- https://arxiv.org/abs/2202.13234 -> Safe Exploration for Efficient Policy Evaluation and Comparison
- https://arxiv.org/abs/2003.05623 -> Off-policy Policy Evaluation For Sequential Decisions Under Unobserved Confounding
- https://proceedings.neurips.cc/paper_files/paper/2022/file/3bf80b34f731313b8292f4578e820c90-Paper-Conference.pdf -> Off-Policy Evaluation for Action-Dependent
- Non-Stationary Environments
- https://www.usenix.org/legacy/events/nsdi06/tech/full_papers/schroeder/schroeder.pdf -> open vs closed a causionary tale 

## Data

- Azure copilot coding agent traces 2026 
- Burst GPT kdd 2025 
- vs Alibaba 2026 traces ( to check for robusntess)


- One scope warning, plus a good follow-up idea

- The Copilot paper's central finding is that KV-cache reuse depends on session structure. In real LLM routers that means cache-affinity routing: keep a session on the replica holding its cache. Combined with a gray failure, that's an even stronger version of your thesis. A sticky router keeps sending a session to a slowly failing replica, and its logs almost never show what the other replicas would have done.

- But for ICPE, don't model the cache. Treat routing as cache-agnostic over aggregate traffic, and state cache affinity as a limitation and future work. It's an excellent follow-up paper, and it connects Paper 1 directly to agentic serving. Six weeks before the deadline is the wrong time to add it.