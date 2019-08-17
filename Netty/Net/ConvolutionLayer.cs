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
        private readonly int height;

        private readonly int width;

        private readonly int kernelSize;

        private readonly int padding;

        private readonly int outputHeight;

        private readonly int outputWidth;

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

        public ConvolutionLayer(int height, int width, int kernelSize, int padding)
        {
            var random = new RandomInitializer();
            this.height = height;
            this.width = width;
            this.kernelSize = kernelSize;
            this.padding = padding;
            this.outputHeight = height - kernelSize + 1 + (2 * padding);
            this.outputWidth = width - kernelSize + 1 + (2 * padding);
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
            this.inputWithPadding = new float[height + (2 * padding), width + (2 * padding)];
            this.inputUnfolded = new float[this.outputHeight * this.outputWidth, (kernelSize * kernelSize) + 1];
            this.output = new float[this.outputHeight, this.outputWidth];
            this.outputRaw = new float[this.outputHeight, this.outputWidth];
            this.outputRawUnfolded = new float[this.outputHeight * this.outputWidth, 1];
            this.gradientOutputOverRawOutput = new float[this.outputHeight, this.outputWidth];
            this.gradientCostOverRawOutput = new float[this.outputHeight, this.outputWidth];
            this.gradientCostOverRawOutputUnfolded = new float[this.outputHeight * this.outputWidth, 1];
            this.gradientCostOverRawOutputWithPadding = new float[height + kernelSize - 1, width + kernelSize - 1];
            this.gradientCostOverRawOutputWithPaddingUnfolded = new float[height * width, (kernelSize * kernelSize)];
            this.inputUnfoldedAlt = new float[(kernelSize * kernelSize), this.outputHeight * this.outputWidth];
            this.gradientCostOverWeights = new float[kernelSize, kernelSize];
            this.gradientCostOverWeightsUnfolded = new float[(kernelSize * kernelSize), 1];
            this.gradientCostOverInput = new float[height, width];
            this.gradientCostOverInputUnfolded = new float[height * width, 1];
        }

        public float[,] FeedForward(float[,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, this.padding);
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfolded, this.kernelSize, this.kernelSize);
            for (var i = 0; i < this.outputHeight * this.outputWidth; ++i)
            {
                this.inputUnfolded[i, (this.kernelSize * this.kernelSize)] = 1f;
            }

            MatrixHelper.UnfoldConvolutionFilter(this.filter, this.filterUnfolded);
            this.filterUnfolded[(this.kernelSize * this.kernelSize), 0] = this.bias;
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputRawUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.outputRawUnfolded, this.outputRaw);
            for (var i = 0; i < this.outputHeight; ++i)
            {
                for (var j = 0; j < this.outputWidth; ++j)
                {
                    this.output[i, j] = ActivationHelper.Activation(this.outputRaw[i, j]);
                }
            }

            return this.output;
        }

        public float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f)
        {
            // Undo activation & calculate bias gradient.
            var gradientCostOverBias = 0f;
            for (var i = 0; i < this.outputHeight; ++i)
            {
                for (var j = 0; j < this.outputWidth; ++j)
                {
                    this.gradientOutputOverRawOutput[i, j] = ActivationHelper.ActivationGradient(this.outputRaw[i, j]);
                    this.gradientCostOverRawOutput[i, j] = gradientCostOverOutput[i, j] * this.gradientOutputOverRawOutput[i, j];
                    gradientCostOverBias += this.gradientCostOverRawOutput[i, j];
                }
            }

            // Calculate filter gradient.
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfoldedAlt, this.outputHeight, this.outputWidth);
            MatrixHelper.UnfoldConvolutionFilter(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputUnfolded);
            MatrixHelper.Multiply(this.inputUnfoldedAlt, this.gradientCostOverRawOutputUnfolded, this.gradientCostOverWeightsUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverWeightsUnfolded, this.gradientCostOverWeights);

            // Calculate inputs gradient.
            MatrixHelper.Pad(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputWithPadding, this.kernelSize - 1 - this.padding);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            MatrixHelper.UnfoldConvolutionInput(this.gradientCostOverRawOutputWithPadding, this.gradientCostOverRawOutputWithPaddingUnfolded, this.kernelSize, this.kernelSize);
            MatrixHelper.UnfoldConvolutionFilter(this.filterFlipped, this.filterFlippedUnfolded);
            MatrixHelper.Multiply(this.gradientCostOverRawOutputWithPaddingUnfolded, this.filterFlippedUnfolded, this.gradientCostOverInputUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverInputUnfolded, this.gradientCostOverInput);

            // Apply bias gradient.
            this.bias -= learningFactor * gradientCostOverBias;

            // Apply filter gradient.
            for (var i = 0; i < this.kernelSize; ++i)
            {
                for (var j = 0; j < this.kernelSize; ++j)
                {
                    this.filter[i, j] -= learningFactor * this.gradientCostOverWeights[i, j];
                }
            }

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}
