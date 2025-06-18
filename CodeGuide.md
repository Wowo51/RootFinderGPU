# RootFinderGPU – Code Guide

---

## 1 Purpose

RootFinderGPU is a minimal C# **root‑finding library** that implements the classic **Newton–Raphson** method while delegating derivative computation to **TorchSharp’s automatic differentiation** engine.  The goal is to provide a template for numerical algorithms that can **seamlessly switch between CPU and GPU** execution with only a device flag.

## 2 Project Structure

| Path                                       | Description                                                    |
| ------------------------------------------ | -------------------------------------------------------------- |
| `RootFinderGPU/NewtonRaphsonRootFinder.cs` | Core static class that performs the Newton–Raphson iterations. |
| `RootFinderGPU.csproj`                     | .NET 9 project file, references `TorchSharp‑cuda‑windows`.     |
| `RootFinderGPUTest/`                       | MSTest project exercising a wide range of unit tests.          |
| `RootFinderGPU.sln`                        | Visual Studio solution tying everything together.              |

```
RootFinderGPU
├── NewtonRaphsonRootFinder.cs
└── RootFinderGPU.csproj
RootFinderGPUTest
├── NewtonRaphsonTests.cs
└── MSTestSettings.cs
RootFinderGPU.sln
```

## 3 Prerequisites

* **.NET 9 SDK** (or higher)
* **TorchSharp 0.105.0** – the CUDA‑enabled build is referenced but falls back to CPU if a CUDA device is unavailable.
* A CUDA‑capable GPU **(optional)** – set the device to `torch.CUDA` to unleash GPU acceleration.

## 4 Building & Running

```bash
# clone your fork
> git clone https://github.com/<user>/RootFinderGPU.git
> cd RootFinderGPU/src

# restore, build, test
> dotnet restore
> dotnet build -c Release
> dotnet test RootFinderGPUTest -c Release
```

All tests should pass; failing tests almost always indicate an issue in the function you supply or in the tolerances chosen.

## 5 Using the API

### 5.1 Signature

```csharp
public static double FindRoot(
    Func<torch.Tensor, torch.Tensor> f,
    double initialGuess,
    double tolerance,
    int   maxIterations = 100 )
```

* **`f`** – a scalar‑to‑scalar tensor function *f(x)* defined with TorchSharp ops.
* **`initialGuess`** – starting point $x₀$.
* **`tolerance`** – absolute *|f(x)|* threshold to declare convergence.
* **`maxIterations`** – optional guard against infinite loops.

### 5.2 Example: Finding √2

```csharp
using RootFinderGPU;
using TorchSharp;

// f(x) = x² – 2  ⇒  root = √2 ≈ 1.41421356
Func<torch.Tensor, torch.Tensor> f = x => x.pow(2).sub(2.0);

double root = NewtonRaphsonRootFinder.FindRoot(
    f,
    initialGuess: 1.0,
    tolerance: 1e-8);

Console.WriteLine($"√2 ≈ {root}");
```

Switch to GPU by adding:

```csharp
torch.Device device = torch.CUDA;
```

and passing tensors on that device when constructing `x`.  In the current sample this line is commented for portability.

## 6 Algorithm Flow

1. **Instantiate x** as a tensor with `requires_grad_()` so TorchSharp tracks operations.
2. **Evaluate f(x)** – may call any differentiable TorchSharp ops.
3. **Backward pass** – `torch.autograd.grad` (or `fx.backward()`) populates `x.grad` with df/dx.
4. **Newton update** – `x₁ = x₀ – f(x₀)/f′(x₀)`.
5. **Convergence checks** – tolerance, small derivative, NaN/Inf guards, divergence caps.
6. **Iterate or return** – dispose tensors each loop to prevent native‑memory leaks.

## 7 Error Handling & Edge Cases

| Condition                                     | Return value | Notes                                  |
| --------------------------------------------- | ------------ | -------------------------------------- |
| `f(x)` returns NaN/Inf                        | `double.NaN` | Function invalid at current x.         |
| Derivative ≈ 0                                | `double.NaN` | Avoids divide‑by‑zero & flat tangents. |
| Divergent step size or guess magnitude > 1e10 | `double.NaN` | Likely runaway.                        |
| No convergence after `maxIterations`          | `double.NaN` | Signals caller to adjust guess/params. |

## 8 Extending the Library

* **Alternate methods** – add Secant, Bisection, or Brent algorithms in sibling classes for robustness where derivatives are unavailable.
* **Vector roots** – wrap TorchSharp’s `torch.autograd.functional.jacobian` to extend Newton’s method to ℝⁿ.
* **Batch evaluation** – treat multiple initial guesses in a tensor batch to exploit GPU parallelism.

## 9 Unit Testing

The `RootFinderGPUTest` project covers:

* Linear, quadratic, cubic, transcendental, and multi‑root polynomials.
* Zero‑derivative hazards and functions without real roots.
* Convergence failure when iteration cap is deliberately tight.

Use these tests as templates to verify new algorithms.

## 10 Troubleshooting

| Symptom            | Likely Cause                                             | Fix                                                                |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------------ |
| Always returns NaN | Bad initial guess; derivative zero; tolerance too tight. | Try different guess/tolerance or switch to a bracket‑based method. |
| CUDA out‑of‑memory | Large batch size or many simultaneous roots.             | Free tensors promptly; reduce batch; monitor GPU memory.           |
| Gradient error     | Function uses non‑differentiable ops.                    | Replace with differentiable ops or finite‑difference fallback.     |

## 11 License & Credits

### Happy root‑finding! 🎯
</br>
Copyright [TranscendAI.tech](https://TranscendAI.tech) 2025.<br>
Authored by Warren Harding. AI assisted.</br>
