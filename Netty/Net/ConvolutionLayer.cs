// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ConvolutionLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net
{
    using System;

    using Netty.Net.Helpers;

    public class ConvolutionLayer
    {
        private readonly int size;

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

        public ConvolutionLayer(int size)
        {
            var random = new RandomInitializer();
            this.size = size;
            this.filter = new float[3, 3];
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < 3; ++j)
                {
                    this.filter[i, j] = random.NextFloat();
                }
            }

            this.bias = random.NextFloat();
            this.filterUnfolded = new float[10, 1];
            this.filterFlipped = new float[3, 3];
            this.filterFlippedUnfolded = new float[9, 1];
            this.inputWithPadding = new float[size + 2, size + 2];
            this.inputUnfolded = new float[size * size, 10];
            this.output = new float[size, size];
            this.outputRaw = new float[size, size];
            this.outputRawUnfolded = new float[size * size, 1];
            this.gradientOutputOverRawOutput = new float[size, size];
            this.gradientCostOverRawOutput = new float[size, size];
            this.gradientCostOverRawOutputUnfolded = new float[size * size, 1];
            this.gradientCostOverRawOutputWithPadding = new float[size + 2, size + 2];
            this.gradientCostOverRawOutputWithPaddingUnfolded = new float[size * size, 9];
            this.inputUnfoldedAlt = new float[9, size * size];
            this.gradientCostOverWeights = new float[3, 3];
            this.gradientCostOverWeightsUnfolded = new float[9, 1];
            this.gradientCostOverInput = new float[size, size];
            this.gradientCostOverInputUnfolded = new float[size * size, 1];
        }

        public float[,] FeedForward(float[,] input)
        {
            MatrixHelper.Pad(input, this.inputWithPadding, 1);
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfolded, 3);
            for (var i = 0; i < this.size * this.size; ++i)
            {
                this.inputUnfolded[i, 9] = 1f;
            }

            MatrixHelper.UnfoldConvolutionFilter(this.filter, this.filterUnfolded);
            this.filterUnfolded[9, 0] = this.bias;
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputRawUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.outputRawUnfolded, this.outputRaw);
            for (var i = 0; i < this.size; ++i)
            {
                for (var j = 0; j < this.size; ++j)
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
            for (var i = 0; i < this.size; ++i)
            {
                for (var j = 0; j < this.size; ++j)
                {
                    this.gradientOutputOverRawOutput[i, j] = ActivationHelper.ActivationGradient(this.outputRaw[i, j]);
                    this.gradientCostOverRawOutput[i, j] = gradientCostOverOutput[i, j] * this.gradientOutputOverRawOutput[i, j];
                    gradientCostOverBias += this.gradientCostOverRawOutput[i, j];
                }
            }

            // Calculate filter gradient.
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfoldedAlt, this.size);
            MatrixHelper.UnfoldConvolutionFilter(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputUnfolded);
            MatrixHelper.Multiply(this.inputUnfoldedAlt, this.gradientCostOverRawOutputUnfolded, this.gradientCostOverWeightsUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverWeightsUnfolded, this.gradientCostOverWeights);

            // Calculate inputs gradient.
            MatrixHelper.Pad(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputWithPadding, 1);
            MatrixHelper.Flip(this.filter, this.filterFlipped);
            MatrixHelper.UnfoldConvolutionInput(this.gradientCostOverRawOutputWithPadding, this.gradientCostOverRawOutputWithPaddingUnfolded, 3);
            MatrixHelper.UnfoldConvolutionFilter(this.filterFlipped, this.filterFlippedUnfolded);
            MatrixHelper.Multiply(this.gradientCostOverRawOutputWithPaddingUnfolded, this.filterFlippedUnfolded, this.gradientCostOverInputUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverInputUnfolded, this.gradientCostOverInput);

            // Apply bias gradient.
            this.bias -= gradientCostOverBias;

            // Apply filter gradient.
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < 3; ++j)
                {
                    this.filter[i, j] -= this.gradientCostOverWeights[i, j];
                }
            }

            // Return input gradient.
            return this.gradientCostOverInput;
        }
    }
}
