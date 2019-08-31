namespace Netty.Net.Layers.Builders
{
    public class PoolingLayerBuilder : ILayerBuilder
    {
        private readonly int kernelHeight;

        private readonly int kernelWidth;

        public PoolingLayerBuilder(int kernelHeight, int kernelWidth)
        {
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
        }

        public ILayer Build(int inputDepth, int inputHeight, int inputWidth)
        {
            return new PoolingLayer(inputDepth, inputHeight, inputWidth, this.kernelHeight, this.kernelWidth);
        }
    }
}