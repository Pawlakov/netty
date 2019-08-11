namespace ClickbaitGenerator.NeuralNet.Service.Functions.Contracts
{
    using System.Collections.Generic;

    /// <summary>
    /// Interface for functions that are used to calculate errors for back propagation algorithm.
    /// </summary>
    public interface IErrorFunction
    {
        string Name { get; }

        List<float> CalculateErrors(List<float> expectedValues, List<float> outcomeValues);
    }
}
