using Netty.Net.Helpers;
using System;

namespace Netty.Net.Layers
{
    public class DenseLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly int outputDepth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[,] weights;

        private float bias;

        private readonly float[,] inputUnfolded;

        private readonly float[,] outputUnfolded;

        private readonly float[,,] output;

        private float gradientCostOverBias;

        private readonly float[,] gradientCostOverWeights;

        private readonly float[,,] gradientCostOverInput;

        public int OutputDepth => this.outputDepth;

        public int OutputHeight => this.outputHeight;

        public int OutputWidth => this.outputWidth;

        public DenseLayer(int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth)
        {
            var random = new RandomInitializer();
            this.depth = inputDepth;
            this.height = inputHeight;
            this.width = inputWidth;
            this.outputDepth = outputDepth;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;

            this.weights = new float[inputDepth * inputHeight * inputWidth, this.outputDepth * this.outputHeight * this.outputWidth];
            for (var i = 0; i < weights.GetLength(0); ++i)
            {
                for (var j = 0; j < weights.GetLength(1); ++j)
                {
                    weights[i, j] = random.NextFloat();
                }
            }

            this.bias = 0f;
            this.inputUnfolded = new float[1, inputDepth * inputHeight * inputWidth];
            this.outputUnfolded = new float[1, this.outputDepth * this.outputHeight * this.outputWidth];
            this.output = new float[this.outputDepth, this.outputHeight, this.outputWidth];
            this.gradientCostOverWeights = new float[weights.GetLength(0), weights.GetLength(1)];
            this.gradientCostOverInput = new float[depth, height, width];
        }

        public float[,,] FeedForward(float[,,] input)
        {
            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        inputUnfolded[0, ((i * this.height) + j) * this.width + k] = input[i, j, k];
                    }
                }
            }

            MatrixHelper.Multiply(this.inputUnfolded, this.weights, this.outputUnfolded);

            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.output[i, j, k] = this.outputUnfolded[0, ((i * this.height) + j) * this.width + k] + this.bias;
                    }
                }
            }

            return this.output;
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1)
        {
            for (var i = 0; i < this.outputDepth; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        this.gradientCostOverBias += learningFactor * gradientCostOverOutput[i, j, k];
                    }
                }
            }

            for (var i = 0; i < this.weights.GetLength(0); ++i)
            {
                for (var j = 0; j < this.weights.GetLength(1); ++j)
                {
                    this.gradientCostOverWeights[i, j] += learningFactor * this.inputUnfolded[0, i];
                }
            }

            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.gradientCostOverInput[i, j, k] = 0f;
                        for (var l = 0; l < this.weights.GetLength(1); ++l)
                        {
                            var a = ((i * this.height) + j) * this.width + k;
                            this.gradientCostOverInput[i, j, k] += learningFactor * this.weights[a, l];
                        }
                    }
                }
            }

            return this.gradientCostOverInput;
        }

        public void UpdateParameters()
        {
            this.bias -= this.gradientCostOverBias;
            this.gradientCostOverBias = 0f;

            for (var i = 0; i < this.weights.GetLength(0); ++i)
            {
                for (var j = 0; j < this.weights.GetLength(1); ++j)
                {
                    this.weights[i, j] -= this.gradientCostOverWeights[i, j];
                    this.gradientCostOverWeights[i, j] = 0f;
                }
            }
        }
    }
}
