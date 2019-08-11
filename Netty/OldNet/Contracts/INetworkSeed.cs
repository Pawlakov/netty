namespace ClickbaitGenerator.NeuralNet.Contracts
{
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public interface INetworkSeed
    {
        int InputHeight { get; }

        int InputWidth { get; }

        int InputDepth { get; }

        IActivationFunction ActivationFunction { get; }

        IPoolingFunction PoolingFunction { get; }

        int ConvolutionKernelRadius { get; }

        int BottleneckSize { get; }
    }
}