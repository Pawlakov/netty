namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;

    public static class Program
    {
        private const int Height = 3;

        private const int Width = 3;

        private const int KernelHeight = 3;

        private const int KernelWidth = 3;

        private const int Padding = 1;

        private const float LearningFactor = 0.1f;

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

            var template = new float[Height - KernelHeight + (2 * Padding) + 1, Width - KernelWidth + (2 * Padding) + 1];
            for (var i = 0; i < template.GetLength(0); ++i)
            {
                for (var j = 0; j < template.GetLength(1); ++j)
                {
                    template[i, j] = 0.1f * (i + j);
                }
            }

            float[,] output = null;
            var net = new NeuralNet();
            net.Add(new ConvolutionLayer(Height, Width, KernelHeight, KernelWidth, Padding));
            net.Add(new ActivationLayer(Height - KernelHeight + (2 * Padding) + 1, Width - KernelWidth + (2 * Padding) + 1));
            for (var epoch = 0; epoch < 1000000; ++epoch)
            {
                output = net.FeedForward(input);
                var gradient = new float[output.GetLength(0), output.GetLength(1)];
                ErrorHelper.CalculateErrorGradient(template, output, gradient);
                net.BackPropagate(gradient, LearningFactor);
            }

            Console.Clear();
            var error = ErrorHelper.CalculateError(template, output);
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
        }
    }
}
