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

        private readonly float[] bias;

        private readonly float[,,,] filter;

        private readonly float[,,,] filterFlipped;

        private readonly float[,,] inputWithPadding;

        private readonly float[,,] output;

        private readonly float[] gradientCostOverBias;

        private readonly float[,,] gradientCostOverOutputWithPadding;

        private readonly float[,,,] gradientCostOverWeights;

        private readonly float[,,] gradientCostOverInput;

        private readonly float[][,] inputWithPaddingJagged;

        private readonly float[][,] gradientCostOverOutputJagged;

        private readonly float[,][,] gradientCostOverWeightsJagged;

        private readonly MultiChannelConvolution feedForwardConvolution;

        private readonly MonoChannelConvolution filterGradientConvolution;

        private readonly MultiChannelConvolution inputGradientConvolution;

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
            this.bias = new float[1];
            this.filter = new float[1, 1, kernelHeight, kernelWidth];
            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < 1; ++j)
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

            this.filterFlipped = new float[1, 1, kernelHeight, kernelWidth];
            this.inputWithPadding = new float[1, height + (2 * padding), width + (2 * padding)];
            this.output = new float[1, this.outputHeight, this.outputWidth];
            this.gradientCostOverBias = new float[1];
            this.gradientCostOverOutputWithPadding = new float[1, height + kernelHeight - 1, width + kernelWidth - 1];
            this.gradientCostOverWeights = new float[1, 1, kernelHeight, kernelWidth];
            this.gradientCostOverInput = new float[1, height, width];
            this.inputWithPaddingJagged = new float[1][,];
            this.gradientCostOverOutputJagged = new float[1][,];
            this.gradientCostOverWeightsJagged = new float[1, 1][,];
            for (var i = 0; i < 1; ++i)
            {
                this.inputWithPaddingJagged[i] = new float[this.height + (2 * this.padding), this.width + (2 * this.padding)];
                for (var j = 0; j < 1; ++j)
                {
                    this.gradientCostOverWeightsJagged[i, j] = new float[this.kernelHeight, this.kernelWidth];
                }
            }

            for (var i = 0; i < 1; ++i)
            {
                this.gradientCostOverOutputJagged[i] = new float[this.outputHeight, this.outputWidth];
            }

            this.feedForwardConvolution = new MultiChannelConvolution(1, height + (2 * padding), width + (2 * padding), 1, kernelHeight, kernelWidth);
            this.filterGradientConvolution = new MonoChannelConvolution(height + (2 * padding), width + (2 * padding), this.outputHeight, this.outputWidth);
            this.inputGradientConvolution = new MultiChannelConvolution(1, height + kernelHeight - 1, width + kernelWidth - 1, 1, kernelHeight, kernelWidth);
        }

        public float[,,] FeedForward(float[,,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, this.padding, this.padding);
            this.feedForwardConvolution.Convolve(this.inputWithPadding, this.filter, this.output);
            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.output[i, j, k] = this.output[i, j, k] + this.bias[i];
                    }
                }
            }

            return this.output;
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1f)
        {
            // Calculate bias gradient.
            for (var i = 0; i < 1; ++i)
            {
                this.gradientCostOverBias[i] = 0f;
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.gradientCostOverBias[i] += gradientCostOverOutput[i, j, k];
                    }
                }
            }

            // Calculate filter gradient.
            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < this.height + (2 * this.padding); ++j)
                {
                    for (var k = 0; k < this.width + (2 * this.padding); ++k)
                    {
                        this.inputWithPaddingJagged[i][j, k] = this.inputWithPadding[i, j, k];
                    }
                }

                for (var j = 0; j < 1; ++j)
                {
                    for (var k = 0; k < this.kernelHeight; ++k)
                    {
                        for (var l = 0; l < this.kernelWidth; ++l)
                        {
                            this.gradientCostOverWeightsJagged[i, j][k, l] = this.gradientCostOverWeights[i, j, k, l];
                        }
                    }
                }
            }

            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        this.gradientCostOverOutputJagged[i][j, k] = gradientCostOverOutput[i, j, k];
                    }
                }
            }

            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < 1; ++j)
                {
                    this.filterGradientConvolution.Convolve(
                        this.inputWithPaddingJagged[j],
                        this.gradientCostOverOutputJagged[i],
                        this.gradientCostOverWeightsJagged[i, j]);
                }
            }

            // Calculate inputs gradient.
            MatrixHelper.Pad(gradientCostOverOutput, this.gradientCostOverOutputWithPadding, this.kernelHeight - 1 - this.padding, this.kernelHeight - 1 - this.padding);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            this.inputGradientConvolution.Convolve(this.gradientCostOverOutputWithPadding, this.filterFlipped, this.gradientCostOverInput);

            // Apply bias gradient.
            for (var i = 0; i < 1; ++i)
            {
                this.bias[i] -= learningFactor * this.gradientCostOverBias[i];
            }

            // Apply filter gradient.
            for (var i = 0; i < 1; ++i)
            {
                for (var j = 0; j < 1; ++j)
                {
                    for (var k = 0; k < this.kernelHeight; ++k)
                    {
                        for (var l = 0; l < this.kernelWidth; ++l)
                        {
                            this.filter[i, j, k, l] -= learningFactor * this.gradientCostOverWeights[i, j, k, l];
                        }
                    }
                }
            }

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}
