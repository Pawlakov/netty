// --------------------------------------------------------------------------------------------------------------------
// <copyright file="MultiChannelConvolution.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net.Helpers
{
    using System;

    /// <summary>
    /// Encapsulates the operation of convolution by matrix multiplication.
    /// </summary>
    public class MultiChannelConvolution
    {
        private readonly int inputDepth;

        /// <summary>
        /// The input height.
        /// </summary>
        private readonly int inputHeight;

        /// <summary>
        /// The input width.
        /// </summary>
        private readonly int inputWidth;

        private readonly int filterCount;

        /// <summary>
        /// The kernel height.
        /// </summary>
        private readonly int kernelHeight;

        /// <summary>
        /// The kernel width.
        /// </summary>
        private readonly int kernelWidth;

        /// <summary>
        /// The output height.
        /// </summary>
        private readonly int outputHeight;

        /// <summary>
        /// The output width.
        /// </summary>
        private readonly int outputWidth;

        /// <summary>
        /// The input restructured for matrix multiplication.
        /// </summary>
        private readonly float[,] inputUnfolded;

        /// <summary>
        /// The filter restructured for matrix multiplication.
        /// </summary>
        private readonly float[,] filterUnfolded;

        /// <summary>
        /// The output restructured as a result of matrix multiplication.
        /// </summary>
        private readonly float[,] outputUnfolded;

        /// <summary>
        /// Initializes a new instance of the <see cref="MultiChannelConvolution"/> class.
        /// </summary>
        /// <param name="inputHeight">
        /// The input height.
        /// </param>
        /// <param name="inputWidth">
        /// The input width.
        /// </param>
        /// <param name="kernelHeight">
        /// The kernel height.
        /// </param>
        /// <param name="kernelWidth">
        /// The kernel width.
        /// </param>
        /// <exception cref="ArgumentException">
        /// Thrown when the kernel is bigger than the input.
        /// </exception>
        public MultiChannelConvolution(int inputDepth, int inputHeight, int inputWidth, int filterCount, int kernelHeight, int kernelWidth)
        {
            if (inputHeight < kernelHeight)
            {
                throw new ArgumentException("Kernel cannot be bigger than the input.", nameof(kernelHeight));
            }

            if (inputWidth < kernelWidth)
            {
                throw new ArgumentException("Kernel cannot be bigger than the input.", nameof(kernelWidth));
            }

            this.inputDepth = inputDepth;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.filterCount = filterCount;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.outputHeight = inputHeight - kernelHeight + 1;
            this.outputWidth = inputWidth - kernelWidth + 1;

            this.filterUnfolded = new float[kernelHeight * kernelWidth * inputDepth, filterCount];
            this.inputUnfolded = new float[this.outputHeight * this.outputWidth, kernelHeight * kernelWidth * inputDepth];
            this.outputUnfolded = new float[this.outputHeight * this.outputWidth, filterCount];
        }

        /// <summary>
        /// Performs the convolution.
        /// </summary>
        /// <param name="input">
        /// The input matrix.
        /// </param>
        /// <param name="filter">
        /// The filter matrix.
        /// </param>
        /// <param name="output">
        /// The output matrix.
        /// </param>
        /// <exception cref="ArgumentException">
        /// Thrown when the size of matrices is incorrect.
        /// </exception>
        public void Convolve(float[,,] input, float[,,,] filter, float[,,] output)
        {
            if (input.GetLength(0) != this.inputDepth || input.GetLength(1) != this.inputHeight || input.GetLength(2) != this.inputWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(input));
            }

            if (filter.GetLength(0) != this.filterCount || filter.GetLength(1) != this.inputDepth || filter.GetLength(2) != this.kernelHeight || filter.GetLength(3) != this.kernelWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(filter));
            }

            if (output.GetLength(0) != this.filterCount || output.GetLength(1) != this.outputHeight || output.GetLength(2) != this.outputWidth)
            {
                throw new ArgumentException("Wrong input size.", nameof(output));
            }

            this.UnfoldConvolutionInput(input);
            this.UnfoldConvolutionFilter(filter);
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputUnfolded);
            this.FoldConvolutionOutput(output);
        }

        /// <summary>
        /// Restructures the input for matrix multiplication.
        /// </summary>
        /// <param name="input">
        /// The input matrix.
        /// </param>
        private void UnfoldConvolutionInput(float[,,] input)
        {
            for (var i = 0; i < this.outputHeight * this.outputWidth; ++i)
            {
                for (var j = 0; j < this.kernelHeight * this.kernelWidth * this.inputDepth; ++j)
                {
                    var x = j / (this.kernelHeight * this.kernelWidth);
                    var y = (i / (this.inputWidth - this.kernelWidth + 1)) + ((j % (this.kernelHeight * this.kernelWidth)) / this.kernelWidth);
                    var z = (i % (this.inputWidth - this.kernelWidth + 1)) + (j % this.kernelWidth);
                    this.inputUnfolded[i, j] = input[x, y, z];
                }
            }
        }

        /// <summary>
        /// Restructures the filter for matrix multiplication.
        /// </summary>
        /// <param name="filter">
        /// The filter matrix.
        /// </param>
        private void UnfoldConvolutionFilter(float[,,,] filter)
        {
            for (var i = 0; i < this.filterCount; i++)
            {
                for (var j = 0; j < this.inputDepth; j++)
                {
                    for (var k = 0; k < this.kernelHeight; ++k)
                    {
                        for (var l = 0; l < this.kernelWidth; ++l)
                        {
                            this.filterUnfolded[(((j * this.kernelHeight) + k) * this.kernelWidth) + l, i] = filter[i, j, k, l];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Restructures the output to the correct shape.
        /// </summary>
        /// <param name="output">
        /// The output matrix.
        /// </param>
        private void FoldConvolutionOutput(float[,,] output)
        {
            for (var i = 0; i < this.filterCount; ++i)
            {
                for (var j = 0; j < this.outputHeight; ++j)
                {
                    for (var k = 0; k < this.outputWidth; ++k)
                    {
                        output[i, j, k] = this.outputUnfolded[(j * this.outputWidth) + k, i];
                    }
                }
            }
        }
    }
}