namespace Netty.Net.Helpers
{
    using System;

    public class MultipleMonoChannelConvolution
    {
        private readonly int inputDepth;

        private readonly int filterCount;

        private readonly int inputHeight;

        private readonly int inputWidth;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[][,] inputJagged;

        private readonly float[][,] filterJagged;

        private readonly float[,][,] outputJagged;

        private readonly MonoChannelConvolution convolution;

        public MultipleMonoChannelConvolution(int inputDepth, int filterCount, int inputHeight, int inputWidth, int kernelHeight, int kernelWidth)
        {
            this.inputDepth = inputDepth;
            this.filterCount = filterCount;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.outputHeight = inputHeight - kernelHeight + 1;
            this.outputWidth = inputWidth - kernelWidth + 1;

            this.inputJagged = new float[inputDepth][,];
            this.filterJagged = new float[filterCount][,];
            this.outputJagged = new float[filterCount, inputDepth][,];
            for (var i = 0; i < filterCount; ++i)
            {
                this.filterJagged[i] = new float[kernelHeight, kernelWidth];
                for (var j = 0; j < inputDepth; ++j)
                {
                    this.outputJagged[i, j] = new float[this.outputHeight, this.outputWidth];
                }
            }

            for (var i = 0; i < inputDepth; ++i)
            {
                this.inputJagged[i] = new float[inputHeight, inputWidth];
            }

            this.convolution = new MonoChannelConvolution(inputHeight, inputWidth, kernelHeight, kernelWidth);
        }

        public void Convolve(float[,,] input, float[,,] filter, float[,,,] output)
        {
            if (input.GetLength(0) != this.inputDepth || input.GetLength(1) != this.inputHeight || input.GetLength(2) != this.inputWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(input));
            }

            if (filter.GetLength(0) != this.filterCount || filter.GetLength(1) != this.kernelHeight || filter.GetLength(2) != this.kernelWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(filter));
            }

            if (output.GetLength(0) != this.filterCount || output.GetLength(1) != this.inputDepth || output.GetLength(2) != this.outputHeight || output.GetLength(3) != this.outputWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(output));
            }

            for (var i = 0; i < this.inputDepth; ++i)
            {
                for (var j = 0; j < this.inputHeight; ++j)
                {
                    for (var k = 0; k < this.inputWidth; ++k)
                    {
                        this.inputJagged[i][j, k] = input[i, j, k];
                    }
                }
            }

            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.kernelHeight; ++j)
                {
                    for (var k = 0; k < this.kernelWidth; ++k)
                    {
                        this.filterJagged[i][j, k] = filter[i, j, k];
                    }
                }
            }

            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.inputDepth; ++j)
                {
                    this.convolution.Convolve(
                        this.inputJagged[j],
                        this.filterJagged[i],
                        this.outputJagged[i, j]);
                }
            }

            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.inputDepth; ++j)
                {
                    for (var k = 0; k < this.outputHeight; ++k)
                    {
                        for (var l = 0; l < this.outputWidth; ++l)
                        {
                            output[i, j, k, l] = this.outputJagged[i, j][k, l];
                        }
                    }
                }
            }
        }
    }
}