namespace ClickbaitGenerator.NeuralNet.Service.Functions.Contracts
{
    using System.Collections.Generic;

    public interface IPoolingFunction
    {
        float Calculate(IList<float> input, out int index);
    }
}
