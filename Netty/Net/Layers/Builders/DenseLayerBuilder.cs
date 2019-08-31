namespace Netty.Net.Layers.Builders
{
    public class DenseLayerBuilder : ILayerBuilder
    {
        private readonly int outputDepth;

        private readonly int outputHeight;

        private readonly int outputWidth;

        public DenseLayerBuilder(int outputDepth, int outputHeight, int outputWidth)
        {
            this.outputDepth = outputDepth;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;
        }

        public ILayer Build(int inputDepth, int inputHeight, int inputWidth)
        {
            return new DenseLayer(inputDepth, inputHeight, inputWidth, this.outputDepth, this.outputHeight, this.outputWidth);
        }
    }
}
