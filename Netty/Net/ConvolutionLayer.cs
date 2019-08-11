namespace Netty.Net
{
    using Netty.Net.Helpers;
    using System;

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

        public float[,] Calculate(float[,] input)
        {
            ApplyInputPadding(input);
            UnfoldInput();
            UnfoldFilter();
            MatrixHelper.Multiply(inputUnfolded, filterUnfolded, outputUnfolded, Activate);
            FoldOutput();
            return output;
        }

        private float Activate(float value)
        {
            var expResult = Math.Exp(value);
            var result = 1 / (float)(1 + expResult);
            return result;
        }

        private void ApplyInputPadding(float[,] input)
        {
            for (var i = 0; i < height; ++i)
            {
                for (var j = 0; j < width; ++j)
                {
                    inputWithPadding[i + 1, j + 1] = input[i, j];
                }
            }
        }

        private void UnfoldInput()
        {
            for (var i = 0; i < height * width; ++i)
            {
                for (var j = 0; j < 9; ++j)
                {
                    var x = (i / width) + (j / 3);
                    var y = (i % width) + (j % 3);
                    inputUnfolded[i, j] = inputWithPadding[x, y];
                }
            }
        }

        private void UnfoldFilter()
        {
            for (var i = 0; i < 3; ++i)
            {
                for (var j = 0; j < 3; ++j)
                {
                    this.filterUnfolded[(i * 3) + j, 0] = filter[i, j];
                }
            }

            this.filterUnfolded[9, 0] = bias;
        }

        private void FoldOutput()
        {
            for (var i = 0; i < height; ++i)
            {
                for (var j = 0; j < width; ++j)
                {
                    this.output[i, j] = outputUnfolded[(i * width) + j, 0];
                }
            }
        }
    }
}
