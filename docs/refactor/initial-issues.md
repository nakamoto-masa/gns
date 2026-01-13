# Refactoring Starting Point

Code quality issues identified during the design review conducted in November 2025

- The design review focused only on the GNS model

## Issues

- Tight coupling of features that should be independent
    - Parts of the GNS algorithm are implemented in scripts
        - Particle position updates
        - Special handling of KINEMATIC particles
    - Multiple features written inline within a single function
        - Configuration value interpretation
        - Parts of the GNS algorithm
        - Training procedures
    - NN model itself and external interface as a prediction model
        - LearnedSimulator class
    - Execution of inference and evaluation of inference results
        - rollout function
- Incorrect type annotations
- Passing information through global variables
- Multiple layers of passing dict-type configuration values
