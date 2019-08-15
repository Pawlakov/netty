// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ConvolutionLayer.cs" company="Paweł Matusek">
//   Copyright (c) Paweł Matusek. All rights reserved.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

namespace Netty.Net
{
    using Netty.Net.Helpers;

    public class ConvolutionLayer
    {
        private readonly int width;
        private readonly int height;
        private readonly float[,] filter;
        private readonly float bias;
        private readonly float[,] filterUnfolded;
        private readonly float[,] inputWithPadding;
        private readonly float[,] inputUnfolded;
        private readonly float[,] output;
        private readonly float[,] outputUnfolded;

        public ConvolutionLayer(int width, int height)
        {
            var random = new RandomInitializer();
            this.width = width;
            this.height = height;
            this.filter = new float[3, 3];
            for(var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < 3; ++j)
                {
                    this.filter[i, j] = random.NextFloat();
                }
            }

            this.bias = random.NextFloat();
            this.filterUnfolded = new float[10, 1];
            this.inputWithPadding = new float[height + 2, width + 2];
            this.inputUnfolded = new float[width * height, 10];
            for (var i = 0; i < width * height; ++i)
            {
                this.inputUnfolded[i, 9] = 1f;
            }

            this.output = new float[height, width];
            this.outputUnfolded = new float[width * height, 1];
        }

        public float[,] FeedForward(float[,] input)
        {
            this.ApplyInputPadding(input);
            this.UnfoldInput();
            this.UnfoldFilter();
            MatrixHelper.Multiply(this.inputUnfolded, this.filterUnfolded, this.outputUnfolded, ActivationHelper.Activation);
            this.FoldOutput();
            return this.output;
        }

        private void ApplyInputPadding(float[,] input)
        {
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.inputWithPadding[i + 1, j + 1] = input[i, j];
                }
            }
        }

        private void UnfoldInput()
        {
            for (var i = 0; i < this.height * this.width; ++i)
            {
                for (var j = 0; j < 9; ++j)
                {
                    var x = (i / this.width) + (j / 3);
                    var y = (i % this.width) + (j % 3);
                    this.inputUnfolded[i, j] = this.inputWithPadding[x, y];
                }
            }
        }

        private void UnfoldFilter()
        {
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < 3; ++j)
                {
                    this.filterUnfolded[(i * 3) + j, 0] = this.filter[i, j];
                }
            }

            this.filterUnfolded[9, 0] = this.bias;
        }

        private void FoldOutput()
        {
            for (var i = 0; i < this.height; ++i)
            {
                for (var j = 0; j < this.width; ++j)
                {
                    this.output[i, j] = this.outputUnfolded[(i * this.width) + j, 0];
                }
            }
        }
    }
}
