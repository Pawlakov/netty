namespace Netty.Net.Helpers
{
    using System;

    using Netty.Net.Exceptions;

    public class Convolution
    {
        private readonly int inputHeight;

        private readonly int inputWidth;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        private readonly float[,] inputUnfolded;

        private readonly float[,] filterUnfolded;

        private readonly float[,] outputUnfolded;

        public Convolution(int inputHeight, int inputWidth, int kernelHeight, int kernelWidth)
        {
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.outputHeight = inputHeight - kernelHeight + 1;
            this.outputWidth = inputWidth - kernelWidth + 1;

            this.filterUnfolded = new float[kernelHeight * kernelWidth, 1];
            this.inputUnfolded = new float[this.outputHeight * this.outputWidth, kernelHeight * kernelWidth];
            this.outputUnfolded = new float[this.outputHeight * this.outputWidth, 1];
        }

        public void Convolve(float[,] input, float[,] filter, float[,] output)
        {
            this.UnfoldConvolutionInput(input);
            this.UnfoldConvolutionFilter(filter);
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputUnfolded);
            this.FoldConvolutionOutput(output);
        }

        private void UnfoldConvolutionInput(float[,] input)
        {
            for (var i = 0; i < (this.inputHeight - this.kernelHeight + 1) * (this.inputWidth - this.kernelWidth + 1); ++i)
            {
                for (var j = 0; j < this.kernelHeight * this.kernelWidth; ++j)
                {
                    var x = (i / (this.inputWidth - this.kernelWidth + 1)) + (j / this.kernelWidth);
                    var y = (i % (this.inputWidth - this.kernelWidth + 1)) + (j % this.kernelWidth);
                    this.inputUnfolded[i, j] = input[x, y];
                }
            }
        }

        private void UnfoldConvolutionFilter(float[,] filter)
        {
            for (var i = 0; i < this.kernelHeight; ++i)
            {
                for (var j = 0; j < this.kernelWidth; ++j)
                {
                    this.filterUnfolded[(i * this.kernelWidth) + j, 0] = filter[i, j];
                }
            }
        }

        private void FoldConvolutionOutput(float[,] output)
        {
            for (var i = 0; i < this.outputHeight; ++i)
            {
                for (var j = 0; j < this.outputWidth; ++j)
                {
                    output[i, j] = this.outputUnfolded[(i * this.outputWidth) + j, 0];
                }
            }
        }
    }
}