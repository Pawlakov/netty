namespace Netty.Net.Layers
{
    using System;

    public class PoolingLayer : ILayer
    {
        private readonly int depth;

        private readonly int height;

        private readonly int width;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[,,] inputWithPadding;

        private readonly float[,,] output;

        private readonly float[,,] gradientCostOverInput;

        private readonly ValueTuple<int, int>[,,] inputSwitches;

        public int OutputDepth => this.depth;

        public int OutputHeight => this.outputHeight;

        public int OutputWidth => this.outputWidth;

        public PoolingLayer(int depth, int height, int width, int kernelHeight, int kernelWidth)
        {
            this.depth = depth;
            this.height = height;
            this.width = width;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.outputHeight = (height / kernelHeight) + (height % kernelHeight > 0 ? 1 : 0);
            this.outputWidth = (width / kernelWidth) + (width % kernelWidth > 0 ? 1 : 0);

            this.inputWithPadding = new float[depth, this.outputHeight * kernelHeight, this.outputWidth * kernelWidth];
            this.output = new float[depth, this.outputHeight, this.outputWidth];
            this.gradientCostOverInput = new float[depth, height, width];
            this.inputSwitches = new ValueTuple<int, int>[depth, this.outputHeight, this.outputWidth];
        }

        public float[,,] FeedForward(float[,,] input)
        {
            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        this.inputWithPadding[i, j, k] = input[i, j, k];
                    }
                }
            }

            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        this.output[i, j, k] = this.inputWithPadding[i, j * this.kernelHeight, k * this.kernelWidth];
                        this.inputSwitches[i, j, k] = ValueTuple.Create(0, 0);
                        for (var l = 0; l < this.outputHeight; ++l)
                        {
                            for (var m = l == 0 ? 1 : 0; m < this.outputWidth; ++m)
                            {
                                var value = this.inputWithPadding[i, (j * this.kernelHeight) + l, (k * this.kernelWidth) + m];
                                if (value > this.output[i, j, k])
                                {
                                    this.output[i, j, k] = value;
                                    this.inputSwitches[i, j, k] = ValueTuple.Create(l, m);
                                }
                            }
                        }
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
                        this.gradientCostOverInput[i, j, k] = 0f;
                    }
                }
            }

            for (var i = 0; i < this.depth; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        var (item1, item2) = this.inputSwitches[i, j, k];
                        this.gradientCostOverInput[i,
                            (j * this.kernelHeight) + item1,
                            (k * this.kernelWidth) + item2] = gradientCostOverOutput[i, j, k];
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