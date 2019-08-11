namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public interface IUnpoolingMap : IMap
    {
        /// <summary>
        /// Calculates activation of all neurons using provided function.
        /// The function object has to be thread safe.
        /// </summary>
        /// <param name="maxIndexes">List that will store which Neurons had the most significant activation.</param>
        /// <param name="activationActivationFunction">Used to calculate activations of neurons in this layer, right before assigning proper weights to output connections.</param>
        void CalculateNeuronsActivation(List<int> maxIndexes, IActivationFunction activationActivationFunction);
    }
}
