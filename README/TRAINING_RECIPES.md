# PPO Training Recipes

## Issues and Solutions

### PPO Stable Training Issue (Approx KL vs KL)
The main issue with PPO stable training relates to the KL divergence calculation and adaptive learning rate mechanism. The approximate KL divergence used in PPO training can lead to instability when the adaptive learning rate is not properly bounded.

## Training Recipes

### Negative Impacts (-)
- **Adaptive Learning Rate**: Higher values need to be maximum 1e-3. Values larger than 1e-3 unlearn the policy
- **Entropy Coefficient**: Negatively impacts performance (set it to 0)
- **Large Learning Rate**: Negatively impacts training (maximum 1e-3)
- **Action Clipping**: Be careful of clipping actions to 1.0 in steps or env wrapper - this can hinder learning
- **Large Negative Penalties**: For avoidance tasks, large negative penalties are less effective than termination
- **InteractivePathPlanning**: Less efficient than RayTracing - increases memory consumption significantly
- **Bootstrapping in Finite Horizon Tasks**: Bootstrapping values work for infinite horizon tasks but can overestimate in finite horizon tasks, leading to non-learning of the agent (be careful)

### Positive Impacts (+)
- **Higher Action Noise**: In the beginning provides good performance (but not too much) - 1.0
- **Initial Noise Standard Deviation**: Set init_noise_std to 10-20% of the action space range (max - min) for proper exploration initialization
- **EMA Agent for Inference**: Use Exponential Moving Average (EMA) agent for inference during PPO rollout to smoothen the learning for faster and smoother learning
- **Higher Steps**: Rollout steps for sparse reward tasks helps
- **Update Epochs**: Larger update epochs needed to learn - 10 (lower or larger than this may under or overfit)
- **Batch Size**: Large mini batch size improves performance - 4096 (mini batch size) with num_mini_batches = 8
- **Larger Image Size Observations**: Larger image size observations help achieve better performance
- **Termination for Avoidance**: Works better than large negative penalty for avoidance tasks
- **Increased Episode Length**: Sometimes increasing episode length works better for learning
- **RayTracing**: Way more efficient than InteractivePathPlanning for training/rendering
- **Lower Precision Training**: Training in lower floating point saves memory (bfloat16 is recommended if GPU VRAM is a concern)

### Environment Design Considerations
- **Goal Distance**: Closer goals make the task more difficult - consider this when designing reward structures and task complexity

### Policy Frequency Considerations
- **Low Level Policy Training**: Typically trained at 1/500 Hz with 16 as decimation
- **Recommended Training Frequency**: Use 1/200 Hz with 8 as decimation to stay closer to lower level policy but with fewer updates

## Configuration Example

```python
@configclass
class ExperimentArgs:
    learning_rate: float = 0.0003  # 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 64  # 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    
    num_minibatches: int = 8  # 8
    """the number of mini-batches"""
    update_epochs: int = 10  # 10
    """the K epochs to update the policy"""
    
    ent_coef: float = 0.0  # 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0  # 1.0
    """coefficient of the value function"""
    
    # Adaptive learning rate parameters
    adaptive_lr: bool = True
    """Use adaptive learning rate based on KL divergence"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""
    lr_multiplier: float = 1.5
    """Factor to multiply/divide learning rate by"""
```

## Adaptive Learning Rate Implementation

```python
def update_learning_rate_adaptive(
    optimizer, kl_divergence, desired_kl, lr_multiplier, min_lr=1e-6, max_lr=1e-3
):
    current_lr = optimizer.param_groups[0]["lr"]

    if kl_divergence > desired_kl * 2.0:
        new_lr = current_lr / lr_multiplier
    elif kl_divergence < desired_kl / 2.0:
        new_lr = current_lr * lr_multiplier
    else:
        new_lr = current_lr

    # Clamp learning rate to reasonable bounds
    new_lr = np.clip(new_lr, min_lr, max_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr
```

### Usage in Training Loop

```python
if args.target_kl is not None and args.adaptive_lr:
    new_lr = update_learning_rate_adaptive(
        optimizer, kl_mean.item(), args.target_kl, args.lr_multiplier, 
        min_lr=1e-7, max_lr=1e-3
    )
```

**Key Point**: The `max_lr` is set to 1e-3 to prevent policy unlearning, which is crucial for stable PPO training.
