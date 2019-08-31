using System;
using System.Collections.Generic;
using System.Text;

namespace Netty.Net.Layers
{
    public class UnpoolingLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[,,] output;

        private readonly float[,,] gradientCostOverInput;

        public int OutputDepth => this.depth;

        public int OutputHeight => this.outputHeight;

        public int OutputWidth => this.outputWidth;

        public UnpoolingLayer(int depth, int height, int width, int kernelHeight, int kernelWidth)
        {
            this.depth = depth;
            this.height = height;
            this.width = width;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.outputHeight = height * kernelHeight;
            this.outputWidth = width * kernelWidth;

            this.output = new float[depth, this.outputHeight, this.outputWidth];
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
                        this.output[i, j * this.kernelHeight, k * this.kernelWidth] = input[i, j, k];
                    }
                }
            }

            return this.output;
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1)
        {
            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.gradientCostOverInput[i, j, k] = gradientCostOverOutput[i, j * this.kernelHeight, k * this.kernelWidth];
                    }
                }
            }

            return this.gradientCostOverInput;
        }

        public void UpdateParameters()
        {
        }
    }
}
