namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;

    public static class Program
    {
        private const int Size = 6;

        private const int Padding = 1;

        private const float TargetError = 0.0005f;

        public static void Main(string[] args)
        {
            var input = new float[Size, Size];
            for (var i = 0; i < Size; ++i)
            {
                for (var j = 0; j < Size; ++j)
                {
                    input[i, j] = 0.1f * (i + j);
                }
            }

            var inputUnpadded = new float[Size + 2 * (Padding - 1), Size + 2 * (Padding - 1)];
            MatrixHelper.Pad(input, inputUnpadded, Padding - 1);
            var layer = new ConvolutionLayer(Size, Padding);
            var error = 1f;
            float[,] output = null;
            while (error >= TargetError)
            {
                output = layer.FeedForward(input);
                error = ErrorHelper.CalculateError(inputUnpadded, output);
                var gradient = new float[output.GetLength(0), output.GetLength(1)];
                ErrorHelper.CalculateErrorGradient(inputUnpadded, output, gradient);
                layer.BackPropagate(gradient);
            }

            Console.WriteLine();
            for (var i = 0; i < output.GetLength(0); ++i)
            {
                Console.Write("[");
                for (var j = 0; j < output.GetLength(1); ++j)
                {
                    Console.Write("{0:0.0000} ", output[i, j]);
                }

                Console.WriteLine("]");
            }

            Console.WriteLine();
            Console.WriteLine("Error: {0:0.0000}", error);
        }
    }
}
