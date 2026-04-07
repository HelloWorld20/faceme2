# Identity-Preserving Blind Face Restoration

## Slide 1

- COURSE：SDSC 8016 Deep Learning for Computer Vision
- MEMBERS：LU Yang, WANG Xuanyi, Yuan Ye, Wei Jianghong,  LIU Mingxuan, YAO Chenxi
- DATE：March 7th, 2026
- Identity-Preserving Blind Face Restoration

## Slide 2

- Why This Project? The Identity Drift Problem
- Existing models (e.g., GFPGAN) often lose key facial features during enhancement, leading to misidentification.
- Blurry Input
- Low-quality source image
- ⚠ Identity Drift
- PULSE Output (Problem)
- Clear but altered features
- Ideal Output (Goal)
- Sharp & Identity Preserved
- Critical Insight:
- Identity preservation is paramount for high-stakes applications like security and forensics. Current models often sacrifice identity consistency for visual quality.
- Our goal:
- Maximize identity preservation while maintaining image quality (PSNR) in blurry face restoration, achieving an optimal balance between clarity and identity consistency

## Slide 3

- Existing Solutions & Our Niche
- Method
- Core Idea
- ID Retention
- Advantage
- GFPGAN/CF
- GAN/Codebook
- ⭐⭐⭐
- Explicit ID Loss
- Diffusion
- Denoising+Ref
- ⭐⭐⭐⭐
- Efficient Inference
- Ours
- Dual-Branch Decouple
- ⭐⭐⭐⭐
- Core Optimization
- Key Insight:
- Existing methods treat ID retention as an auxiliary constraint. We elevate it to the core objective with explicit loss and trade-off control.
- Existing Solutions
- ID retention is an auxiliary constraint
- Fixed weight, limited flexibility
- Our Solution
- ID retention as the core goal
- Adjustable trade-off & Feedback
- "From Auxiliary to Core"
- Elevating the standard of face restoration

## Slide 4

- Key Innovations & Research Questions
- Innovation 1: Feature Decoupling Theory
- Inspired by Multi-Head Attention, explicitly separating Quality and Identity feature learning paths.
- Innovation 2: Dual-Branch MVP Architecture
- Integrating Quality Branch (SwinIR) and Identity Branch (ResNet-18) with a novel Identity Loss.
- Innovation 3: Dual-Metric Evaluation System
- Using PSNR & ArcFace\_Sim to analyze the trade-off curve between image fidelity and identity preservation.
- Research Questions (RQs)
- RQ1: Dual vs Single?
- RQ2: λ Weight Impact?
- RQ3: Degradation Effect?
- Research Logic Flow
- Theoretical Motivation
- Feature Decoupling & Attention
- Implementation Scheme
- MVP Architecture & Identity Loss
- Validation Method
- Dual-Metric Analysis & Trade-off

## Slide 5

- Dataset & Preprocessing Pipeline
- Training
- CelebA-HQ
- <br />
  - FFHQHigh-QualityFacial Data
- Testing
- CelebA-HQ SynthLFW / LFW-blurGeneralization
- Ability Evaluation
- Degradation Simulation
- Gaussian Blur • Downsampling • Noise (Light/Med/Heavy)
- Standardized Preprocessing Workflow
- Raw Image
- Face Alignment
- dlib 68 Landmarks
- Normalization
- Range \[-1, 1]
- ArcFace
- Identity Loss
- Strategy Insight: Combining synthetic data for training with real-world data for testing ensures both controllability and generalizability. The ArcFace feature extraction step is critical for preserving identity information during restoration.

## Slide 6

- Dual-Branch MVP Architecture
- Quality Branch (SwinIR)
- Leverage the powerful feature extraction capability of the SwinIR model to focus on restoring high-frequency texture details in images and address blurring issues.
- Identity Branch (ResNet-18)
- A lightweight ResNet-18 is used to extract identity features, explicitly preserving key facial identity information through a decoupling structure.
- Loss Function Strategy
- Quality Branch
- (SwinIR)
- Identity Branch
- (ResNet-18)
- Fusion
- L1 Loss +
- Perceptual Loss
- ArcFace
- Loss
- Combined loss: L1 loss + perceptual loss + identity loss. The cosine distance of features is calculated using ArcFace to enforce identity alignment.

## Slide 7

- Alternating Training Strategy
- Training Stage Division
- Core Idea
- Quality branch and identity branch alternate in dominating the training process, avoiding competition conflicts in multi-task learning and achieving smooth transition.
- Stage
- Epoch Ratio
- Identity Branch
- Quality Branch
- Fusion Module
- ID Weight
- Phase 1
- 0-40%
- Normal Training
- (Maintain EMA)
- Frozen
- Light Training
- 1.0
- Phase 2
- 40-80%
- EMA Frozen
- Unfrozen, Low LR
- Normal Training
- 0.8 (constant)
- Phase 3
- 80-100%
- Unfrozen, Low LR
- Continue Low LR
- Fine-tuning
- 0.7-0.8

## Slide 8

- Evaluation Metrics & Ablation Design
- Dual Metric Evaluation System
- Image Quality: PSNR, SSIM (Objective Quality)
- Identity Preservation: ArcFace Cosine Similarity, Top-1 Acc
- Ablation Study Design
- Architecture Validation: Single vs. Dual-branch Structure
- Weight Sensitivity: Impact of λ on Identity Loss
- Error Analysis
- Failure Case Study: Profile views, Occlusions, Extreme Lighting
- Insight: Understanding model limitations for improvement
- Trade-off Analysis & Optimization
- Balancing Act: Visualizing the inverse relationship between image fidelity (PSNR) and identity similarity.
- Optimal Point: Identified the "Recommended Working Point" for practical deployment.

## Slide 9

- 6-Week Execution Plan
- Week 1
- Baseline reproduction
- Deliverables: Comparison Chart + Baseline Data
- Week 2
- Single Branch + Loss
- Deliverables:
- Single Branch Experiment Table
- Week 3
- Dual Branch MVP
- Deliverables:
- Code +
- Preliminary Results
- Week 4
- Ablation & Trade-offs
- Deliverables:
- Ablation table +
- trade-off curve
- Week 5
- Real Data Testing
- Deliverables:
- Results +
- Error Analysis
- Week 6
- Report and PPT
- Deliverables:
- Final Report +
- Code Repository
- Milestones
- End of W2: Identity Loss Verification
- ArcFace\_Sim improvement >0.1, proving effectiveness
- W6 Preliminary: Report Completed
- PPT/Report draft completed, preparing for defense
- Core Insight: We break down complex tasks into 6 manageable steps, aligning with the rigorous narrative structure of the Sample Proposal to ensure efficient project progression.
- End of W4: Core data completed
- Completed all architecture comparisons and ablation experiment collection

## Slide 10

- Team Division of Labor
- Member
- Responsible Module
- Execution Duties
- WANG Xuanyi
- Background & Introduction
- Literature collection, background research, presentation introduction script
- LU Yang
- Related Work
- Comparative analysis of SOTA methods, summary of research gaps
- Yuan Ye
- Innovation & Methodology
- Propose core ideas, design the overall algorithm framework
- Wei Jianghong
- Model Architecture
- Model coding implementation, network structure tuning
- YAO Chenxi
- Evaluation & Analysis
- Conduct experiments, data visualization, result analysis
- LIU Mingxuan
- Dataset & Appendix
- Data preprocessing, dataset construction, supplementary materials
- Collaboration Tools
- GitHub
- Code Version Control
- Weights & Biases
- Experiment Tracking
- Notion
- Documentation & Wiki

## Slide 11

- Summary & Future Work
- Project Summary
- Core Hypothesis Validation: Focus on identity-aware restoration using a minimalist dual-branch architecture to ensure experimental rigor.
- MVP Success Criteria: ArcFace\_Sim improvement > 0.1, and PSNR decrease < 0.5dB.
- Deliverables: Reproducible code, dual-metric evaluation table, technical report, and presentation PPT.
- Future Work
- Module upgrade: Introducing attention mechanism for weighted fusion to enhance repair details.
- Backbone comparison: Comparing the performance of ResNet-18 and VMamba in the Identity Branch.
- Scenario expansion: Exploring temporal consistency in videos to extend static image repair to dynamic video.

## Slide 12

- THANK YOUQ & A

