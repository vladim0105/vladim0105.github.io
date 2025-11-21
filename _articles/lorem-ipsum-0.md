---
id: 0
title: "Vladim's guide to Deep Learning Compilation"
subtitle: "testest"
date: "2025.11.17"
tags: "tag1, tag2"
---

## Introduction

Deep learning compilation is a rapidly evolving field at the intersection of machine learning and systems optimization. But what exactly is it? At its core, deep learning compilation transforms the high-level computational graphs that ML researchers define in frameworks like PyTorch or TensorFlow into optimized code that executes efficiently on target hardware.

The compilation challenge varies dramatically depending on your deployment target. I find it helpful to consider two main categories: compilation for resource-constrained edge devices (such as microcontrollers with limited memory and compute) and compilation for high-performance accelerators (such as GPUs, TPUs, or distributed GPU clusters). In this post, we'll start with single-threaded microcontrollers to establish foundational concepts before tackling more complex scenarios.

Deep learning compilers generally pursue one of two execution strategies:

1. **Interpreter-based execution**: The compiler emits a bytecode or instruction set consumed by a runtime interpreter that executes operations sequentially
2. **Code generation (codegen)**: The compiler produces source code (typically C/C++) representing the entire model, which is then compiled with a traditional compiler into native machine code

Each approach involves fundamental tradeoffs in flexibility, performance, and binary size-themes we'll explore throughout this post.

## The Compilation Pipeline

Before a deep learning compiler can optimize anything, it needs to understand what you're asking it to compile. This is where input representations come in standardized formats that capture the structure and semantics of your neural network.

At their core, all deep learning models can be represented as directed acyclic graphs (DAGs) where:

* Nodes represent operations (convolution, matrix multiplication, activation functions)
* Edges represent tensors flowing between operations
* Attributes store operation-specific parameters (kernel sizes, strides, activation types)

For example, a simple two-layer neural network might be represented as:



### Input Representation: The Starting Point

Before the compiler can work its magic, it needs to understand what it is asked to compiler. This is where input representations come in. Each framework (PyTorch, Tensorflow, Jax, ...) has their own representation, but they are all directed acyclic graphs (DAGs). 

At their core, all deep learning models can be represented as a DAG where:

* Nodes represent operations (convolution, matrix multiplication, activation functions)
* Edges represent tensors flowing between operations
* Attributes store operation-specific parameters (kernel sizes, strides, activation types)

For instance, to inspect a PyTorch model's graph representation via TorchScript:

```python

scripted_model = torch.jit.script(model)
print(scripted_model.graph)

```

TODO: Show example output

Once parsed, the input model is converted into the compiler's native format-typically called an intermediate representation (IR). This is where all subsequent analysis and optimization takes place.

A compiler can support multiple input formats (PyTorch, TensorFlow, Jax, etc.), but in my experience, this approach rarely pays off. The fundamental problem is that frameworks make different design choices: PyTorch defaults to channels-first (NCHW) while TensorFlow prefers channels-last (NHWC); one framework's "Conv2D" may have subtly different padding semantics than another's. These discrepancies compound quickly. Operations that seem equivalent often aren't, leading to subtle bugs and edge cases that surface only on specific models or input shapes. Supporting a single, well-defined input format tends to be far more maintainable.

### Intermediate Representation

Once the input is parsed, the compiler converts it into its own internal format-the intermediate representation (IR). This is where all the real work happens: analysis, optimization, and eventually code generation all operate on this IR rather than the original input format.

A good IR strikes a balance between abstraction and detail. Too high-level, and you lose opportunities for optimization. Too low-level, and transformations become tedious and error-prone. Many compilers define multiple IR levels, progressively lowering from high-level operations (like "Conv2D") down to hardware-specific primitives.

MLIR (Multi-Level Intermediate Representation) has emerged as an influential framework in this space. Rather than prescribing a single IR, MLIR provides infrastructure for defining dialects-custom IR layers tailored to specific domains or abstraction levels. A compiler might represent a model in a high-level "tensor" dialect, lower it to a "loop" dialect for optimization, and finally emit a hardware-specific dialect for code generation. MLIR also provides common optimization passes out of the box (canonicalization, loop fusion, scalar replacement, etc.), reducing the amount of infrastructure you need to build from scratch.

That said, I find that most existing tooling focuses on typical feedforward architectures. Recurrent models, stateful inference, and non-standard control flow often lack first-class support. I'm also a strong believer in treating data preprocessing as part of the model itself-it dramatically simplifies deployment when your compiled artifact handles the entire pipeline. Features like persistent counters, early exit, or conditional computation may sound niche, but they're surprisingly valuable in production.

For these reasons, if resources permit, I'd recommend developing your own IR. A general-purpose solution that handles all of this well may never materialize-most projects optimize for benchmark performance rather than the practical concerns that make products shippable.

### Passes

Passes are transformations that a compiler applies to the computation graph to achieve a specific goal-typically optimization, but also tasks like hardware targeting or quantization.

A classic example is **layer fusion**. Consider a Linear layer followed by BatchNorm: since BatchNorm is an affine transformation applied to the Linear layer's output, we can fold its parameters directly into the Linear layer's weights and bias, eliminating an entire operation at inference time. 

While such fusions can be hardcoded, most compilers benefit enormously from a **pattern matching** system that makes expressing these transformations declarative and composable. TVM's pattern matcher is an excellent example of this approach (though at the time of writing, I've been unable to locate current documentation for it).

More sophisticated passes might mark specific operations to run on an NPU rather than a CPU, or transform a floating-point model into a quantized one. We'll cover quantization passes in a dedicated section.

#### A Note on Pattern Matching

In my experience, investing in a robust pattern matcher is one of the highest-leverage decisions you can make when building an ML compiler. A well-designed matcher should support:

- **Logical operators**: AND, OR, NOT for combining conditions
- **Wildcards**: matching arbitrary subgraphs or tensors
- **Attribute constraints**: filtering by operation parameters (e.g., convolutions with a specific stride)
- **Optional elements**: handling variations in graph structure

Consider matching a Linear-BatchNorm pair for fusion. The Linear layer may or may not have a bias term, so our pattern must accommodate both cases:

```python
def linear_bn_pattern(pm: PatternMatcher):
    # Input tensor-we don't care what produces it
    x = pm.Wildcard()
    
    # Linear layer parameters (bind names for later extraction)
    weight = pm.Wildcard().bind("weight")
    bias = pm.Tensor().bind("bias") | pm.None()  # Optional bias
    
    # BatchNorm parameters
    bn_mean = pm.Wildcard().bind("bn_mean")
    bn_std = pm.Wildcard().bind("bn_std")
    
    # The pattern itself: Linear → BatchNorm
    linear = pm.Op("Linear", args=(x, weight, bias)).bind("linear")
    bn = pm.Op("BatchNorm", args=(linear, bn_mean, bn_std)).bind("batchnorm")
    
    return bn

# Apply the pattern across the graph
for match in pm.match_all(graph, linear_bn_pattern):
    weight = match["weight"]
    bn_mean = match["bn_mean"]
    bn_std = match["bn_std"]
    
    # Compute fused weights: W_fused = W * (1 / std)
    # Compute fused bias: b_fused = (b - mean) / std (or just -mean/std if no bias)
    
    if "bias" in match:
        original_bias = match["bias"]
        # ... fuse with existing bias
    else:
        # ... create new bias from BatchNorm parameters
    
    # Replace matched subgraph with fused Linear
    ...
```

This framework naturally extends to more complex patterns-multi-branch structures, chains of operations, or hardware-specific sequences.

#### Beyond Matching: Debugging and Testing

A pattern matching system also provides a foundation for tooling:

- **Debug introspection**: When a pattern unexpectedly fails to match, you want to know *why*. A good framework can report which sub-pattern failed and what it found instead.
- **Pass testing**: Verify correctness by pattern-matching before and after a pass runs. Did the Linear-BatchNorm pair disappear? Did a single fused Linear take its place? This turns pass validation into a declarative specification rather than brittle output comparison.




