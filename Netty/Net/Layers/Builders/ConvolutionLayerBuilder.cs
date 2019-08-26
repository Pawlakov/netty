namespace Netty.Net.Layers.Builders
{
    public class ConvolutionLayerBuilder : ILayerBuilder
    {
        private readonly int filterCount;

        private readonly int kernelHeight;

        private readonly int kernelWidth;

        private readonly int padding;

        public ConvolutionLayerBuilder(int filterCount, int kernelHeight, int kernelWidth, int padding)
        {
            this.filterCount = filterCount;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.padding = padding;
        }

        public ILayer Build(int inputDepth, int inputHeight, int inputWidth)
        {
            return new ConvolutionLayer(inputDepth, inputHeight, inputWidth, this.filterCount, this.kernelHeight, this.kernelWidth, this.padding);
        }
    }
}