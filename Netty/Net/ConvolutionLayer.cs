// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ConvolutionLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net
{
    using Netty.Net.Helpers;

    public class ConvolutionLayer
    {
        private readonly int size;

        private readonly int kernelSize;

        private readonly int padding;

        private readonly int outputSize;

        private readonly float[,] filter;

        private float bias;

        private readonly float[,] filterUnfolded;

        private readonly float[,] filterFlipped;

        private readonly float[,] filterFlippedUnfolded;

        private readonly float[,] inputWithPadding;

        private readonly float[,] inputUnfolded;

        private readonly float[,] output;

        private readonly float[,] outputRaw;

        private readonly float[,] outputRawUnfolded;

        private readonly float[,] gradientOutputOverRawOutput;

        private readonly float[,] gradientCostOverRawOutput;

        private readonly float[,] gradientCostOverRawOutputUnfolded;

        private readonly float[,] gradientCostOverRawOutputWithPadding;

        private readonly float[,] gradientCostOverRawOutputWithPaddingUnfolded;

        private readonly float[,] inputUnfoldedAlt;

        private readonly float[,] gradientCostOverWeights;

        private readonly float[,] gradientCostOverWeightsUnfolded;

        private readonly float[,] gradientCostOverInput;

        private readonly float[,] gradientCostOverInputUnfolded;

        public ConvolutionLayer(int size, int kernelSize, int padding)
        {
            var random = new RandomInitializer();
            this.size = size;
            this.kernelSize = kernelSize;
            this.padding = padding;
            this.outputSize = size - kernelSize + 1 + (2 * padding);
            this.filter = new float[kernelSize, kernelSize];
            for (var i = 0; i < kernelSize; ++i)
            {
                for (var j = 0; j < kernelSize; ++j)
                {
                    this.filter[i, j] = random.NextFloat();
                }
            }

            this.bias = random.NextFloat();
            this.filterUnfolded = new float[(kernelSize * kernelSize) + 1, 1];
            this.filterFlipped = new float[kernelSize, kernelSize];
            this.filterFlippedUnfolded = new float[kernelSize * kernelSize, 1];
            this.inputWithPadding = new float[size + (2 * padding), size + (2 * padding)];
            this.inputUnfolded = new float[this.outputSize * this.outputSize, (kernelSize * kernelSize) + 1];
            this.output = new float[this.outputSize, this.outputSize];
            this.outputRaw = new float[this.outputSize, this.outputSize];
            this.outputRawUnfolded = new float[this.outputSize * this.outputSize, 1];
            this.gradientOutputOverRawOutput = new float[this.outputSize, this.outputSize];
            this.gradientCostOverRawOutput = new float[this.outputSize, this.outputSize];
            this.gradientCostOverRawOutputUnfolded = new float[this.outputSize * this.outputSize, 1];
            this.gradientCostOverRawOutputWithPadding = new float[size + kernelSize - 1, size + kernelSize - 1];
            this.gradientCostOverRawOutputWithPaddingUnfolded = new float[size * size, (kernelSize * kernelSize)];
            this.inputUnfoldedAlt = new float[(kernelSize * kernelSize), this.outputSize * this.outputSize];
            this.gradientCostOverWeights = new float[kernelSize, kernelSize];
            this.gradientCostOverWeightsUnfolded = new float[(kernelSize * kernelSize), 1];
            this.gradientCostOverInput = new float[size, size];
            this.gradientCostOverInputUnfolded = new float[size * size, 1];
        }

        public float[,] FeedForward(float[,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, this.padding);
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfolded, this.kernelSize);
            for (var i = 0; i < this.outputSize * this.outputSize; ++i)
            {
                this.inputUnfolded[i, (this.kernelSize * this.kernelSize)] = 1f;
            }

            MatrixHelper.UnfoldConvolutionFilter(this.filter, this.filterUnfolded);
            this.filterUnfolded[(this.kernelSize * this.kernelSize), 0] = this.bias;
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputRawUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.outputRawUnfolded, this.outputRaw);
            for (var i = 0; i < this.outputSize; ++i)
            {
                for (var j = 0; j < this.outputSize; ++j)
                {
                    this.output[i, j] = ActivationHelper.Activation(this.outputRaw[i, j]);
                }
            }

            return this.output;
        }

        public float[,] BackPropagate(float[,] gradientCostOverOutput)
        {
            // Undo activation & calculate bias gradient.
            var gradientCostOverBias = 0f;
            for (var i = 0; i < this.outputSize; ++i)
            {
                for (var j = 0; j < this.outputSize; ++j)
                {
                    this.gradientOutputOverRawOutput[i, j] = ActivationHelper.ActivationGradient(this.outputRaw[i, j]);
                    this.gradientCostOverRawOutput[i, j] = gradientCostOverOutput[i, j] * this.gradientOutputOverRawOutput[i, j];
                    gradientCostOverBias += this.gradientCostOverRawOutput[i, j];
                }
            }

            // Calculate filter gradient.
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfoldedAlt, this.outputSize);
            MatrixHelper.UnfoldConvolutionFilter(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputUnfolded);
            MatrixHelper.Multiply(this.inputUnfoldedAlt, this.gradientCostOverRawOutputUnfolded, this.gradientCostOverWeightsUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverWeightsUnfolded, this.gradientCostOverWeights);

            // Calculate inputs gradient.
            MatrixHelper.Pad(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputWithPadding, this.kernelSize - 1 - this.padding);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            MatrixHelper.UnfoldConvolutionInput(this.gradientCostOverRawOutputWithPadding, this.gradientCostOverRawOutputWithPaddingUnfolded, this.kernelSize);
            MatrixHelper.UnfoldConvolutionFilter(this.filterFlipped, this.filterFlippedUnfolded);
            MatrixHelper.Multiply(this.gradientCostOverRawOutputWithPaddingUnfolded, this.filterFlippedUnfolded, this.gradientCostOverInputUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverInputUnfolded, this.gradientCostOverInput);

            // Apply bias gradient.
            this.bias -= gradientCostOverBias;

            // Apply filter gradient.
            for (var i = 0; i < this.kernelSize; ++i)
            {
                for (var j = 0; j < this.kernelSize; ++j)
                {
                    this.filter[i, j] -= this.gradientCostOverWeights[i, j];
                }
            }

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}
