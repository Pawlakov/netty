namespace Netty.Net.Layers.Builders
{
    public class ActivationLayerBuilder : ILayerBuilder
    {
        public ILayer Build(int inputDepth, int inputHeight, int inputWidth)
        {
            return new ActivationLayer(inputDepth, inputHeight, inputWidth);
        }
    }
}