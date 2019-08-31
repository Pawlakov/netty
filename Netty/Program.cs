namespace Netty
{
    using System;
    using System.IO;
    using System.Linq;
    using Microsoft.Extensions.Configuration;
    using Netty.Floater;
    using Netty.Net;
    using Netty.Net.Helpers;
    using Netty.Net.Layers.Builders;

    public static class Program
    {
        public static void Main(string[] args)
        {
            var configBuilder = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", true, true);
            var configuration = configBuilder.Build();
            var inputFloater = new InputDataSerializer(configuration, 64, 64);
            var outputFloater = new OutputDataDeserializer(configuration, 64, 64);
            var input = inputFloater.ToList();
            input.ForEach(x => x.Open());
            var dataset = input.Select(x => Tuple.Create(x.FloatedContent, x.FloatedContent)).ToArray();

            var net = new NeuralNet(3, 64, 64);
            net.Add(new ConvolutionLayerBuilder(6, 3, 3, 1));
            net.Add(new PoolingLayerBuilder(2, 2));
            net.Add(new UnpoolingLayerBuilder(2, 2));
            net.Add(new ConvolutionLayerBuilder(6, 3, 3, 1));
            net.Add(new ActivationLayerBuilder());
            net.Add(new ConvolutionLayerBuilder(3, 3, 3, 1));
            net.Add(new ActivationLayerBuilder());
            net.Build();
            net.Learn(dataset, 400, 5);
            var output = net.FeedForward(dataset[0].Item1);
            Console.WriteLine("Error: {0:0.0000}", ErrorHelper.CalculateError(dataset[0].Item2, output));
            outputFloater.Save(output);
        }
    }
}
