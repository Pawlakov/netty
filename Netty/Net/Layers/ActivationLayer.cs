// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ActivationLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Layers
{
    using Netty.Net.Helpers;

    /// <summary>
    /// The sequential model layer that transforms the input with an activation function.
    /// </summary>
    public class ActivationLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly float[,,] output;

        private readonly float[,,] gradientCostOverInput;

        public int OutputDepth => depth;

        public int OutputHeight => height;

        public int OutputWidth => width;

        public ActivationLayer(int depth, int height, int width)
        {
            this.depth = depth;
            this.height = height;
            this.width = width;
            this.output = new float[depth, height, width];
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
                        this.output[i, j, k] = ActivationHelper.Activation(input[i, j, k]);
                    }
                }
            }

            return this.output;
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1f)
        {
            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.gradientCostOverInput[i, j, k] = gradientCostOverOutput[i, j, k] * ActivationHelper.ActivationGradient(this.output[i, j, k]);
                    }
                }
            }

            return this.gradientCostOverInput;
        }
    }
}