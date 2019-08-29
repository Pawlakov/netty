namespace Netty
{
    using System;

    using Netty.Net;
    using Netty.Net.Helpers;
    using Netty.Net.Layers.Builders;

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
            var input = new float[2, 2, 2] { { { 0f, 1f }, { 1f, 2f } }, { { 2f, 1f }, { 2f, 1f } } };
            var template =
                new float[2, 2, 2] { { { 0.5f, 0.1f }, { 0.0f, 0.6f } }, { { 1.0f, 1.0f }, { 0.5f, 0.9f } } };
            var net = new NeuralNet();
            net.Add(new ConvolutionLayerBuilder(2, 3, 3, 2));
            net.Add(new PoolingLayerBuilder(2, 2));
            net.Add(new ActivationLayerBuilder());
            net.Build(2, 2, 2);
            net.Learn(new[] { Tuple.Create(input, template) }, 100, 1);
            var output = net.FeedForward(input);
            Console.WriteLine("Error: {0:0.0000}", ErrorHelper.CalculateError(template, output));
        }
    }
}
