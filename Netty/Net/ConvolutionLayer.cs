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

        private readonly float[,] inputWithPadding;

        private readonly float[,] inputUnfolded;

        private readonly float[,] output;

        private readonly float[,] outputRaw;

        private readonly float[,] outputRawUnfolded;

        private readonly float[,] gradientOutputOverRawOutput;

        private readonly float[,] gradientCostOverRawOutput;

        private readonly float[,] gradientCostOverRawOutputUnfolded;

        private readonly float[,] inputUnfoldedAlt;

        private readonly float[,] gradientCostOverWeights;

        private readonly float[,] gradientCostOverWeightsUnfolded;

        private readonly float[,] inputGradient;

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
            this.inputWithPadding = new float[size + 2, size + 2];
            this.inputUnfolded = new float[size * size, 10];
            this.output = new float[size, size];
            this.outputRaw = new float[size, size];
            this.outputRawUnfolded = new float[size * size, 1];
            this.gradientOutputOverRawOutput = new float[size, size];
            this.gradientCostOverRawOutput = new float[size, size];
            this.gradientCostOverRawOutputUnfolded = new float[size * size, 1];
            this.inputUnfoldedAlt = new float[9, size * size];
            this.gradientCostOverWeights = new float[3, 3];
            this.gradientCostOverWeightsUnfolded = new float[9, 1];
            this.inputGradient = new float[size, size];
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
            //
            // Calculate inputs gradient.
            MatrixHelper.UnfoldConvolutionInput(this.inputWithPadding, this.inputUnfoldedAlt, this.size);
            MatrixHelper.UnfoldConvolutionFilter(this.gradientCostOverRawOutput, this.gradientCostOverRawOutputUnfolded);
            MatrixHelper.Multiply(this.inputUnfoldedAlt, this.gradientCostOverRawOutputUnfolded, this.gradientCostOverWeightsUnfolded);
            MatrixHelper.FoldConvolutionOutput(this.gradientCostOverWeightsUnfolded, this.gradientCostOverWeights);

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
            return this.inputGradient;
        }
    }
}
