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
            var events = new LearningEvents();
            events.AllDone += (sender, eventArgs) =>
            {
                Console.Clear();
                Console.WriteLine("Epochs done: {0}/{0}", eventArgs.TotalEpochs);
                Console.WriteLine("Average error: {0:0.0000}", eventArgs.FinalError);
            };

            events.EpochDone += (sender, eventArgs) =>
            {
                Console.Clear();
                Console.WriteLine("Epochs done: {0}/{1}", eventArgs.DoneEpochs, eventArgs.TotalEpochs);
                Console.WriteLine("Average error: {0:0.0000}", eventArgs.CurrentError);
            };

            events.EpochProgressUpdate += (sender, eventArgs) =>
            {
                var currentLineCursor = Console.CursorTop;
                Console.SetCursorPosition(0, currentLineCursor);
                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, currentLineCursor);
                Console.Write("Samples done: {0}/{1}", eventArgs.SamplesDone, eventArgs.SamplesTotal);
            };

            var net = new NeuralNet(3, 64, 64);
            net.Add(new ConvolutionLayerBuilder(6, 3, 3, 1));
            net.Add(new ActivationLayerBuilder());
            net.Add(new ConvolutionLayerBuilder(3, 3, 3, 1));
            net.Add(new ActivationLayerBuilder());
            net.Build();
            net.Learn(dataset, 300, 5, events);
            foreach (var sample in dataset)
            {
                var output = net.FeedForward(sample.Item1);
                outputFloater.Save(output);
            }
        }
    }
}
