namespace Netty.Net
{
    using System.Collections.Generic;
    using System.Linq;
    using Netty.Net.Helpers;
    using Netty.Net.Layers;
    using Netty.Net.Layers.Builders;

    public class NeuralNet
    {
        private readonly IList<ILayerBuilder> layerBuilders = new List<ILayerBuilder>();

        private readonly IList<ILayer> layers = new List<ILayer>();

        private float[,,] gradient;

        int OutputDepth => this.layers.Reverse().First().OutputDepth;

        int OutputHeight => this.layers.Reverse().First().OutputHeight;

        int OutputWidth => this.layers.Reverse().First().OutputWidth;

        public void Add(ILayerBuilder layerBuilder)
        {
            this.layerBuilders.Add(layerBuilder);
        }

        public void Build(int inputDepth, int inputHeight, int inputWidth)
        {
            foreach (var builder in this.layerBuilders)
            {
                var layer = builder.Build(inputDepth, inputHeight, inputWidth);
                inputDepth = layer.OutputDepth;
                inputHeight = layer.OutputHeight;
                inputWidth = layer.OutputWidth;
                this.layers.Add(layer);
            }

            gradient = new float[inputDepth, inputHeight, inputWidth];
        }

        public float Learn(float[,,] input, float[,,] target)
        {
            var output = FeedForward(input);
            ErrorHelper.CalculateErrorGradient(target, output, gradient);
            var inputGradient = BackPropagate(gradient);
            return ErrorHelper.CalculateError(target, output);
        }

        public float[,,] FeedForward(float[,,] input)
        {
            return this.layers.Aggregate(input, (current, layer) => layer.FeedForward(current));
        }

        private float[,,] BackPropagate(float[,,] gradientCostOverOutput)
        {
            return this.layers.Reverse().Aggregate(gradientCostOverOutput, (current, layer) => layer.BackPropagate(current, 1));
        }
    }
}