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

        private const float LearningFactor = 0.01f;

        public static void Main(string[] args)
        {
            var input = new float[1, Height, Width];
            for (var i = 0; i < Height; ++i)
            {
                for (var j = 0; j < Width; ++j)
                {
                    input[0, i, j] = 0.1f * i;
                }
            }

            var template = new float[1, Height - KernelHeight + (2 * Padding) + 1, Width - KernelWidth + (2 * Padding) + 1];
            for (var i = 0; i < template.GetLength(1); ++i)
            {
                for (var j = 0; j < template.GetLength(2); ++j)
                {
                    template[0, i, j] = 0.1f * i;
                }
            }

            float[,,] output = null;
            var net = new NeuralNet();
            net.Add(new ConvolutionLayer(Height, Width, KernelHeight, KernelWidth, Padding));
            //net.Add(new ActivationLayer(Height - KernelHeight + (2 * Padding) + 1, Width - KernelWidth + (2 * Padding) + 1));
            while (true)
            {
                output = net.FeedForward(input);
                var gradient = new float[output.GetLength(0), output.GetLength(1), output.GetLength(2)];
                ErrorHelper.CalculateErrorGradient(template, output, gradient);
                net.BackPropagate(gradient, LearningFactor);

                Console.Clear();
                var error = ErrorHelper.CalculateError(template, output);
                Console.WriteLine("Error: {0:0.0000}", error);
                Console.ReadKey();
            }
        }
    }
}
