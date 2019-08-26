namespace Netty.Net
{
    using System.Collections.Generic;
    using System.Linq;
    using Netty.Net.Layers;

    public class NeuralNet
    {
        private readonly IList<ILayer> layers = new List<ILayer>();

        public void Add(ILayer layer)
        {
            this.layers.Add(layer);
        }

        public float[,,] FeedForward(float[,,] input)
        {
            return this.layers.Aggregate(input, (current, layer) => layer.FeedForward(current));
        }

        public float[,,] BackPropagate(float[,,] gradientCostOverOutput, float learningFactor = 1)
        {
            return this.layers.Reverse().Aggregate(gradientCostOverOutput, (current, layer) => layer.BackPropagate(current, learningFactor));
        }
    }
}
