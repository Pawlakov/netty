namespace Netty.Net.Layers.Builders
{
    public interface ILayerBuilder
    {
        ILayer Build(int inputDepth, int inputHeight, int inputWidth);
    }
}