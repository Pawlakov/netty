// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ConvolutionLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net
{
    using Netty.Net.Helpers;

    /// <summary>
    /// The sequential model layer that performs a convolution on its input.
    /// </summary>
    public class ConvolutionLayer : ILayer
    {
        private readonly int height;

        private readonly int width;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int padding;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private float bias;

        private readonly float[,] filter;

        private readonly float[,] filterFlipped;

        private readonly float[,] inputWithPadding;

        private readonly float[,] output;

        private readonly float[,] gradientCostOverOutputWithPadding;

        private readonly float[,] gradientCostOverWeights;

        private readonly float[,] gradientCostOverInput;

        private readonly Convolution feedForwardConvolution;

        private readonly Convolution filterGradientConvolution;

        private readonly Convolution inputGradientConvolution;

        public ConvolutionLayer(int height, int width, int kernelHeight, int kernelWidth, int padding)
        {
            var random = new RandomInitializer();
            this.height = height;
            this.width = width;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.padding = padding;
            this.outputHeight = height - kernelHeight + 1 + (2 * padding);
            this.outputWidth = width - kernelWidth + 1 + (2 * padding);
            this.bias = 0f;
            this.filter = new float[kernelHeight, kernelWidth];
            for (var i = 0; i < kernelHeight; ++i)
            {
                for (var j = 0; j < kernelWidth; ++j)
                {
                    this.filter[i, j] = random.NextFloat();
                }
            }

            this.filterFlipped = new float[kernelHeight, kernelWidth];
            this.inputWithPadding = new float[height + (2 * padding), width + (2 * padding)];
            this.output = new float[this.outputHeight, this.outputWidth];
            this.gradientCostOverOutputWithPadding = new float[height + kernelHeight - 1, width + kernelWidth - 1];
            this.gradientCostOverWeights = new float[kernelHeight, kernelWidth];
            this.gradientCostOverInput = new float[height, width];
            this.feedForwardConvolution = new Convolution(height + (2 * padding), width + (2 * padding), kernelHeight, kernelWidth);
            this.filterGradientConvolution = new Convolution(height + (2 * padding), width + (2 * padding), this.outputHeight, this.outputWidth);
            this.inputGradientConvolution = new Convolution(height + kernelHeight - 1, width + kernelWidth - 1, kernelHeight, kernelWidth);
        }

        public float[,] FeedForward(float[,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, this.padding, this.padding);
            this.feedForwardConvolution.Convolve(this.inputWithPadding, this.filter, this.output);
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.output[i, j] = this.output[i, j] + this.bias;
                }
            }

            return this.output;
        }

        public float[,] BackPropagate(float[,] gradientCostOverOutput, float learningFactor = 1f)
        {
            // Calculate bias and input gradient.
            var gradientCostOverBias = 0f;
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.gradientCostOverInput[i, j] = gradientCostOverOutput[i, j];
                    gradientCostOverBias += gradientCostOverOutput[i, j];
                }
            }

            // Calculate filter gradient.
            this.filterGradientConvolution.Convolve(this.inputWithPadding, gradientCostOverOutput, this.gradientCostOverWeights);

            // Calculate inputs gradient.
            MatrixHelper.Pad(gradientCostOverOutput, this.gradientCostOverOutputWithPadding, this.kernelHeight - 1 - this.padding, this.kernelHeight - 1 - this.padding);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            this.inputGradientConvolution.Convolve(this.gradientCostOverOutputWithPadding, this.filterFlipped, this.gradientCostOverInput);

            // Apply bias gradient.
            this.bias -= learningFactor * gradientCostOverBias;

            // Apply filter gradient.
            for (var i = 0; i < this.kernelHeight; ++i)
            {
                for (var j = 0; j < this.kernelWidth; ++j)
                {
                    this.filter[i, j] -= learningFactor * this.gradientCostOverWeights[i, j];
                }
            }

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}
