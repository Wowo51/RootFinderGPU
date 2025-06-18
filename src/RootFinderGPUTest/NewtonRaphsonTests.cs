using Microsoft.VisualStudio.TestTools.UnitTesting;
using RootFinderGPU;
using TorchSharp;
using static TorchSharp.torch;
using System;

namespace RootFinderGPUTest
{
    [TestClass]
    public sealed class NewtonRaphsonTests
    {
        // Removed CreateTorchFunction helper method as per instructions.

        [TestMethod]
        public void TestLinearFunction()
        {
            // f(x) = x - 5.0, root is 5.0
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.sub(5.0);

            double initialGuess = 4.0;
            double expectedRoot = 5.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for linear function.");
        }

        [TestMethod]
        public void TestQuadraticFunctionPositiveRoot()
        {
            // f(x) = x^2 - 9.0, roots are +/- 3.0
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(2).sub(9.0);

            double initialGuess = 2.0; // Guess near positive root
            double expectedRoot = 3.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for quadratic function (positive root).");
        }

        [TestMethod]
        public void TestQuadraticFunctionNegativeRoot()
        {
            // f(x) = x^2 - 9.0, roots are +/- 3.0
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(2).sub(9.0);

            double initialGuess = -2.0; // Guess near negative root
            double expectedRoot = -3.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for quadratic function (negative root).");
        }

        [TestMethod]
        public void TestCubicFunction()
        {
            // f(x) = x^3 - 8.0, root is 2.0
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(3).sub(8.0);

            double initialGuess = 1.0;
            double expectedRoot = 2.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for cubic function.");
        }

        [TestMethod]
        public void TestTrigonometricFunctionSineNearZero()
        {
            // f(x) = sin(x), root is 0, Pi, etc.
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => torch.sin(x);

            double initialGuess = 0.5; // Near 0
            double expectedRoot = 0.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for sin(x) near 0.");
        }

        [TestMethod]
        public void TestTrigonometricFunctionSineNearPi()
        {
            // f(x) = sin(x), root is 0, Pi, etc.
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => torch.sin(x);

            double initialGuess = 3.0; // Near Pi
            double expectedRoot = Math.PI;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found near Pi.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for sin(x) near Pi.");
        }

        [TestMethod]
        public void TestExponentialFunction()
        {
            // f(x) = e^x - 1.0, root is 0.0
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => torch.exp(x).sub(1.0);

            double initialGuess = 0.5;
            double expectedRoot = 0.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for exponential function.");
        }

        [TestMethod]
        public void TestNoRealRoot()
        {
            // f(x) = x^2 + 1.0, no real roots
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(2).add(1.0);

            double initialGuess = 0.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsTrue(double.IsNaN(actualRoot), "Expected no root to be found (NaN).");
        }

        [TestMethod]
        public void TestZeroDerivativeAtRoot()
        {
            // f(x) = (x-5)^3, root at x=5. Derivative f'(x) = 3(x-5)^2. f'(5) = 0.
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => (x.sub(5.0)).pow(3.0);

            double initialGuess = 4.9; // Close to root, where derivative is small but not zero
            double expectedRoot = 5.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance, maxIterations: 1000);

            // Newton-Raphson can struggle with zero derivatives at the root itself, but should converge if initial guess
            // is not exactly at a point where the derivative becomes zero and stays zero throughout iterations.
            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found, despite small derivative near root.");
            Assert.AreEqual(expectedRoot, actualRoot, 1e-2, "Root not found for function with zero derivative at root.");
        }

        [TestMethod]
        public void TestZeroDerivativeFarFromRoot()
        {
            // f(x) = |x| - 5.0, derivative is undefined at 0. This can expose derivative issues for Newton-Raphson.
            // Using torch.abs(x) which has an undefined derivative at x=0, a common cause for issues.
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => torch.abs(x).sub(5.0);

            double initialGuess = 0.0f; 
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            // Expect NaN because the derivative will be problematic (e.g., undefined or leading to division by zero near 0).
            Assert.IsTrue(double.IsNaN(actualRoot), "Expected NaN due to problematic derivative behavior.");
        }

        [TestMethod]
        public void TestFunctionWithMultipleRoots_Root1()
        {
            // f(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(3).sub(6.0 * x.pow(2)).add(11.0 * x).sub(6.0);

            double initialGuess = 0.5;
            double expectedRoot = 1.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for (x-1)(x-2)(x-3) - root 1.");
        }

        [TestMethod]
        public void TestFunctionWithMultipleRoots_Root2()
        {
            // f(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(3).sub(6.0 * x.pow(2)).add(11.0 * x).sub(6.0);

            double initialGuess = 1.9;
            double expectedRoot = 2.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for (x-1)(x-2)(x-3) - root 2.");
        }

        [TestMethod]
        public void TestFunctionWithMultipleRoots_Root3()
        {
            // f(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(3).sub(6.0 * x.pow(2)).add(11.0 * x).sub(6.0);

            double initialGuess = 2.9;
            double expectedRoot = 3.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found for (x-1)(x-2)(x-3) - root 3.");
        }

        [TestMethod]
        public void TestFunctionWithMaxIterationsReached()
        {
            // A function where convergence is slow or initial guess is far
            // For example, f(x) = x^2, with an initial guess far from 0 and tight tolerance/low max iterations
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.pow(2);

            double initialGuess = 100.0;
            double tolerance = 1e-9;
            int maxIterations = 5; // Low max iterations to force NaN

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance, maxIterations);

            Assert.IsTrue(double.IsNaN(actualRoot), "Expected no root due to max iterations reached.");
        }

        [TestMethod]
        public void TestInitialGuessIsRoot()
        {
            // f(x) = x - 7.0, initial guess is directly the root
            Func<torch.Tensor, torch.Tensor> torchFunc = (torch.Tensor x) => x.sub(7.0);

            double initialGuess = 7.0;
            double expectedRoot = 7.0;
            double tolerance = 1e-6;

            double actualRoot = NewtonRaphsonRootFinder.FindRoot(torchFunc, initialGuess, tolerance);

            Assert.IsFalse(double.IsNaN(actualRoot), "Expected a root to be found immediately.");
            Assert.AreEqual(expectedRoot, actualRoot, tolerance, "Root not found when initial guess is the root.");
        }
    }
}