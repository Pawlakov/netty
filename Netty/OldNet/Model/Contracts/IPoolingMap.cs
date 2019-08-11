namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public interface IPoolingMap : IMap
    {

        /// <summary>
        /// Calculates activation of all neurons using provided function.
        /// The function object has to be thread safe.
        /// </summary>
        /// <param name="activationFunction"></param>
        /// <param name="maxIndexes">List that will store which Neurons had the most significant activation.</param>
        void CalculateNeuronsActivation(IPoolingFunction activationFunction, List<int> maxIndexes);
    }
}
