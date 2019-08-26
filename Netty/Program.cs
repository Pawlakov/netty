namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;
    using Netty.Net.Layers;

    public static class Program
    {
        private const int Depth = 2;

        private const int Height = 2;

        private const int Width = 2;

        private const int FilterCount = 2;

        private const int KernelHeight = 2;

        private const int KernelWidth = 2;

        private const int Padding = 0;

        private const float LearningFactor = 0.1f;

        public static void Main(string[] args)
        {
            var input = new float[2, 2, 2] {{{0f, 1f}, {1f, 2f}}, {{2f, 1f}, {2f, 1f}}};
            var template = new float[2, 1, 1] {{{0.5f}}, {{0.5f}}};
            var gradient = new float[2, 1, 1];
            var net = new NeuralNet();
            net.Add(new ConvolutionLayer(2, 2, 2, 2, 2, 2, 0));
            net.Add(new ActivationLayer(2, 1, 1));
            while(true)
            {
                var output = net.FeedForward(input);
                ErrorHelper.CalculateErrorGradient(template, output, gradient);
                var inputGradient = net.BackPropagate(gradient, LearningFactor);
                Console.Clear();
                var error = ErrorHelper.CalculateError(template, output);
                Console.WriteLine("Error: {0:0.0000}", error);
                Console.ReadKey();
            }
        }
    }
}
