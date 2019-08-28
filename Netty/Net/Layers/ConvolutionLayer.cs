// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ConvolutionLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Layers
{
    using Netty.Net.Helpers;

    /// <summary>
    /// The sequential model layer that performs a convolution on its input.
    /// </summary>
    public class ConvolutionLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly int filterCount;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int padding;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[] bias;

        private readonly float[,,,] filter;

        private readonly float[,,,] filterFlipped;

        private readonly float[,,] inputWithPadding;

        private readonly float[,,] output;

        private readonly float[] gradientCostOverBias;

        private readonly float[,,] gradientCostOverOutputWithPadding;

        private readonly float[,,,] gradientCostOverWeights;

        private readonly float[,,,] gradientCostOverWeightsTemporary;

        private readonly float[,,] gradientCostOverInput;

        private readonly MultiChannelConvolution feedForwardConvolution;

        private readonly MultipleMonoChannelConvolution filterGradientConvolution;

        private readonly MultiChannelConvolution inputGradientConvolution;

        public int OutputDepth => this.filterCount;

        public int OutputHeight => this.outputHeight;

        public int OutputWidth => this.outputWidth;

        public ConvolutionLayer(int depth, int height, int width, int filterCount, int kernelHeight, int kernelWidth, int padding)
        {
            var random = new RandomInitializer();
            this.depth = depth;
            this.height = height;
            this.width = width;
            this.filterCount = filterCount;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.padding = padding;
            this.outputHeight = height - kernelHeight + 1 + (2 * padding);
            this.outputWidth = width - kernelWidth + 1 + (2 * padding);
            this.bias = new float[filterCount];
            this.filter = new float[filterCount, depth, kernelHeight, kernelWidth];
            for (var i = 0; i < filterCount; ++i)
            {
                for (var j = 0; j < depth; ++j)
                {
                    for (var k = 0; k < kernelHeight; ++k)
                    {
                        for (var l = 0; l < kernelWidth; ++l)
                        {
                            this.filter[i, j, k, l] = random.NextFloat();
                        }
                    }
                }
            }

            this.filterFlipped = new float[depth, filterCount, kernelHeight, kernelWidth];
            this.inputWithPadding = new float[depth, height + (2 * padding), width + (2 * padding)];
            this.output = new float[filterCount, this.outputHeight, this.outputWidth];
            this.gradientCostOverBias = new float[filterCount];
            this.gradientCostOverOutputWithPadding = new float[filterCount, height + kernelHeight - 1, width + kernelWidth - 1];
            this.gradientCostOverWeights = new float[filterCount, depth, kernelHeight, kernelWidth];
            this.gradientCostOverWeightsTemporary = new float[filterCount, depth, kernelHeight, kernelWidth];
            this.gradientCostOverInput = new float[depth, height, width];

            this.feedForwardConvolution = new MultiChannelConvolution(depth, height + (2 * padding), width + (2 * padding), filterCount, kernelHeight, kernelWidth);
            this.filterGradientConvolution = new MultipleMonoChannelConvolution(depth, filterCount, height + (2 * padding), width + (2 * padding), this.outputHeight, this.outputWidth);
            this.inputGradientConvolution = new MultiChannelConvolution(filterCount, height + kernelHeight - 1, width + kernelWidth - 1, depth, kernelHeight, kernelWidth);
        }

        public float[,,] FeedForward(float[,,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, this.padding, this.padding);
            this.feedForwardConvolution.Convolve(this.inputWithPadding, this.filter, this.output);
            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        this.output[i, j, k] = this.output[i, j, k] + this.bias[i];
                    }
                }
            }

            return this.output;
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 0.01f)
        {
            // Calculate bias gradient.
            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        this.gradientCostOverBias[i] += learningFactor * gradientCostOverOutput[i, j, k];
                    }
                }
            }

            // Calculate filter gradient.
            this.filterGradientConvolution.Convolve(
                this.inputWithPadding,
                gradientCostOverOutput,
                this.gradientCostOverWeightsTemporary);
            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.depth; ++j)
                {
                    for (var k = 0; k < this.kernelHeight; ++k)
                    {
                        for (var l = 0; l < this.kernelWidth; ++l)
                        {
                            this.gradientCostOverWeights[i, j, k, l] += learningFactor * this.gradientCostOverWeightsTemporary[i, j, k, l];
                        }
                    }
                }
            }

            // Calculate inputs gradient.
            MatrixHelper.Pad(gradientCostOverOutput, this.gradientCostOverOutputWithPadding, this.kernelHeight - 1 - this.padding, this.kernelHeight - 1 - this.padding);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            this.inputGradientConvolution.Convolve(this.gradientCostOverOutputWithPadding, this.filterFlipped, this.gradientCostOverInput);

            // Return input gradient.
            return this.gradientCostOverInput;
        }

        public void UpdateParameters()
        {
            // Apply bias gradient.
            for (var i = 0; i < this.filterCount; ++i)
            {
                this.bias[i] -= this.gradientCostOverBias[i];
                this.gradientCostOverBias[i] = 0f;
            }

            // Apply filter gradient.
            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.depth; ++j)
                {
                    for (var k = 0; k < this.kernelHeight; ++k)
                    {
                        for (var l = 0; l < this.kernelWidth; ++l)
                        {
                            this.filter[i, j, k, l] -= this.gradientCostOverWeights[i, j, k, l];
                            this.gradientCostOverWeights[i, j, k, l] = 0f;
                        }
                    }
                }
            }
        }
    }
}