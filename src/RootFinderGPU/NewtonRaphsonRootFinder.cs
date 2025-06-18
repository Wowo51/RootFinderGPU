using System;
using TorchSharp;
using static TorchSharp.torch;

namespace RootFinderGPU
{
    public static class NewtonRaphsonRootFinder
    {
        /// <summary>
        /// Finds the root of a function using the Newton-Raphson method with TorchSharp for automatic differentiation.
        /// </summary>
        /// <param name="function">The function f(x) for which to find the root.</param>
        /// <param name="initialGuess">The initial guess for the root.</param>
        /// <param name="tolerance">The convergence tolerance.</param>
        /// <param name="maxIterations">The maximum number of iterations.</param>
        /// <returns>The found root, or NaN if convergence is not achieved due to numerical instability or divergence.</returns>
        public static double FindRoot(Func<torch.Tensor, torch.Tensor> function, double initialGuess, double tolerance = 1e-6, int maxIterations = 100)
        {
            torch.Tensor? currentX = null;

            try // Outer try block to ensure the final currentX is disposed.
            {
                currentX = torch.tensor(initialGuess, dtype: torch.float64, requires_grad: true);

                for (int i = 0; i < maxIterations; i++)
                {
                    // Declare tensors that will be created and need disposal within this iteration.
                    torch.Tensor? fxTensor = null;
                    torch.Tensor? dfxTensorDetached = null;
                    torch.Tensor? updateTerm = null;
                    torch.Tensor? tempDetachedCurrentX = null; // A detached copy of the old currentX for calculation, to be disposed.
                    torch.Tensor? newXValAsTensor = null;       // The new x value as a tensor before converting to double.

                    try // Inner try block to ensure per-iteration tensors are disposed on any exit.
                    {
                        // Clear the gradient of currentX from the previous iteration.
                        currentX.grad?.Dispose(); // Safely dispose if not null.
                        currentX.grad = null;     // Ensure reference is nulled after disposal.

                        // 1. Calculate f(x).
                        try
                        {
                            fxTensor = function(currentX);
                        }
                        catch (Exception)
                        {
                            // Function evaluation itself failed. Intermediate tensors disposed by inner finally.
                            // currentX handled by outer finally.
                            return double.NaN;
                        }

                        // Check for invalid f(x) result (e.g., NaN, Inf from function).
                        if (fxTensor is null || fxTensor.isnan().item<bool>() || fxTensor.isinf().item<bool>())
                        {
                            return double.NaN;
                        }

                        double fxVal = fxTensor.ToDouble();

                        // 2. Check for convergence: if f(x) is close enough to zero.
                        if (System.Math.Abs(fxVal) < tolerance)
                        {
                            return currentX.ToDouble(); // Root found. Inner finally disposes per-iteration tensors.
                        }

                        // 3. Compute the derivative df/dx using automatic differentiation.
                        try
                        {
                            fxTensor.backward(); // This populates currentX.grad.
                        }
                        catch (Exception) // Catch any exceptions during backward pass (e.g., InvalidOperationException).
                        {
                            return double.NaN;
                        }

                        // Check if a valid gradient was computed.
                        if (currentX.grad is null || currentX.grad.isnan().item<bool>() || currentX.grad.isinf().item<bool>())
                        {
                            return double.NaN;
                        }
                        dfxTensorDetached = currentX.grad.detach(); // Detach to prevent future operations from altering past graph.

                        double dfxVal = dfxTensorDetached.ToDouble();

                        // 4. Specification 1: If derivative df_x is critically small AND f_x is not within tolerance.
                        double criticalDerivativeThreshold = 1e-12;
                        if (System.Math.Abs(dfxVal) < criticalDerivativeThreshold && System.Math.Abs(fxVal) >= tolerance)
                        {
                            return double.NaN; // Derivative too small, indicating numerical instability/far from root.
                        }

                        // 5. Specification 2: Check for divergence - excessively large step size.
                        double updateTermMagnitudeForCheck;
                        if (System.Math.Abs(dfxVal) == 0.0) // Prevent division by zero for magnitude check, consistent with original logic.
                        {
                            updateTermMagnitudeForCheck = double.PositiveInfinity;
                        }
                        else
                        {
                            updateTermMagnitudeForCheck = System.Math.Abs(fxVal / dfxVal); // Use doubles for this check.
                        }
                        double maxStepSize = 1e10; // "Excessively large" step as per specification.
                        if (updateTermMagnitudeForCheck > maxStepSize)
                        {
                            return double.NaN; // Diverging due to large step.
                        }

                        // If dfxVal is exactly 0.0, we would have returned NaN due to updateTermMagnitudeForCheck being Infinity.
                        // So, at this point, dfxTensorDetached should not contain exact zero for division.
                        updateTerm = fxTensor.div(dfxTensorDetached); // Calculate the Newton-Raphson update term as a tensor.

                        // 6. Calculate the next approximation for x.
                        // Create a detached tensor from currentX for arithmetic operations to avoid modifying currentX directly,
                        // and ensure the new tensor for next iteration is a fresh leaf.
                        tempDetachedCurrentX = currentX.detach();
                        newXValAsTensor = tempDetachedCurrentX.sub(updateTerm); // This creates a new tensor.

                        // 7. Specification 2: Check for divergence - currentGuess itself grows beyond reasonable bounds.
                        double newXValDouble = newXValAsTensor.ToDouble();
                        double maxGuessValue = 1e10; // "Beyond reasonable bounds" as per specification.
                        if (System.Math.Abs(newXValDouble) > maxGuessValue)
                        {
                            return double.NaN; // Diverging due to guess growing too large.
                        }

                        // Prepare for the next iteration: dispose the old currentX and create a new one.
                        currentX.Dispose();
                        currentX = torch.tensor(newXValDouble, dtype: torch.float64, requires_grad: true);
                    }
                    finally // This block ensures all temporary tensors created in THIS iteration are disposed.
                    {
                        fxTensor?.Dispose();
                        dfxTensorDetached?.Dispose();
                        updateTerm?.Dispose();
                        tempDetachedCurrentX?.Dispose();
                        newXValAsTensor?.Dispose();
                    }
                } // End of for loop (maxIterations reached)
            }
            finally // This block ensures the main currentX tensor and its gradient are disposed when the method exits.
            {
                currentX?.grad?.Dispose();
                currentX?.Dispose();
            }

            // No convergence within the maximum number of iterations.
            return double.NaN;
        }
    }
}