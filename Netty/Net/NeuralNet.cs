namespace Netty.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Medallion;
    using Netty.Net.Helpers;
    using Netty.Net.Layers;
    using Netty.Net.Layers.Builders;

    public class NeuralNet
    {
        private readonly IList<ILayerBuilder> layerBuilders = new List<ILayerBuilder>();

        private readonly IList<ILayer> layers = new List<ILayer>();

        private float[,,] gradient;

        private int inputDepth;

        private int inputHeight;

        private int inputWidth;

        int OutputDepth => this.layers.Reverse().First().OutputDepth;

        int OutputHeight => this.layers.Reverse().First().OutputHeight;

        int OutputWidth => this.layers.Reverse().First().OutputWidth;

        public NeuralNet(int inputDepth, int inputHeight, int inputWidth)
        {
            this.inputDepth = inputDepth;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
        }

        public void Add(ILayerBuilder layerBuilder)
        {
            this.layerBuilders.Add(layerBuilder);
        }

        public void Build()
        {
            foreach (var builder in this.layerBuilders)
            {
                var layer = builder.Build(this.inputDepth, this.inputHeight, this.inputWidth);
                this.inputDepth = layer.OutputDepth;
                this.inputHeight = layer.OutputHeight;
                this.inputWidth = layer.OutputWidth;
                this.layers.Add(layer);
            }

            gradient = new float[this.inputDepth, this.inputHeight, this.inputWidth];
        }

        public void Learn(IEnumerable<Tuple<float[,,], float[,,]>> samples, int epochs, int batchSize, LearningEvents events)
        {
            var count = samples.Count();
            var error = 0f;
            for(var i = 0; i < epochs; ++i)
            {
                error = 0f;
                var shuffled = samples.Shuffled();
                var batchCounter = 0;
                var samplesDone = 0;
                foreach(var sample in shuffled)
                {
                    var output = FeedForward(sample.Item1);
                    error += ErrorHelper.CalculateError(sample.Item2, output);
                    ErrorHelper.CalculateErrorGradient(sample.Item2, output, gradient);
                    var inputGradient = BackPropagate(gradient);
                    ++batchCounter;
                    if(batchCounter == batchSize)
                    {
                        batchCounter = 0;
                        UpdateParameters();
                    }

                    ++samplesDone;
                    events.InvokeEpochProgressUpdate(this, samplesDone, count);
                }

                if (batchCounter > 0)
                {
                    UpdateParameters();
                }

                error /= count;
                events.InvokeEpochDone(this, i + 1, epochs, error);
            }

            events.InvokeAllDone(this, epochs, error);
        }

        public float[,,] FeedForward(float[,,] input)
        {
            return this.layers.Aggregate(input, (current, layer) => layer.FeedForward(current));
        }

        private float[,,] BackPropagate(float[,,] gradientCostOverOutput)
        {
            return this.layers.Reverse().Aggregate(gradientCostOverOutput, (current, layer) => layer.BackPropagate(current, 0.1f));
        }

        private void UpdateParameters()
        {
            foreach (var layer in layers)
            {
                layer.UpdateParameters();
            }
        }
    }
}