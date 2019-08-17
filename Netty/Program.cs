namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;

    public static class Program
    {
        private const int Height = 3;

        private const int Width = 5;

        private const int KernelSize = 3;

        private const int Padding = 0;

        private const float LearningFactor = 1f;

        public static void Main(string[] args)
        {
            var input = new float[Height, Width];
            for (var i = 0; i < Height; ++i)
            {
                for (var j = 0; j < Width; ++j)
                {
                    input[i, j] = 0.1f * (i + j);
                }
            }

            var template = new float[Height - KernelSize + (2 * Padding) + 1, Width - KernelSize + (2 * Padding) + 1];
            for (var i = 0; i < template.GetLength(0); ++i)
            {
                for (var j = 0; j < template.GetLength(1); ++j)
                {
                    template[i, j] = 0.1f * (i + j);
                }
            }

            var layer = new ConvolutionLayer(Height, Width, KernelSize, Padding);
            while (true)
            {
                var output = layer.FeedForward(input);
                var error = ErrorHelper.CalculateError(template, output);
                var gradient = new float[output.GetLength(0), output.GetLength(1)];
                ErrorHelper.CalculateErrorGradient(template, output, gradient);
                layer.BackPropagate(gradient, LearningFactor);
                
                Console.Clear();
                for (var i = 0; i < template.GetLength(0); ++i)
                {
                    Console.Write("[");
                    for (var j = 0; j < template.GetLength(1); ++j)
                    {
                        Console.Write("{0:0.0000} ", template[i, j]);
                    }

                    Console.WriteLine("]");
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
                Console.ReadKey();
            }
        }
    }
}
