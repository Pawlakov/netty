namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    using System.Collections.Generic;
    
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    /// <summary>
    /// A single map created by given filter in layer.
    /// </summary>
    public interface IFilterMap : IMap
    {
        /// <summary>
        /// Height of provided input.
        /// </summary>
        new int Height { get; }
        /// <summary>
        /// Width of provided input.
        /// </summary>
        new int Width { get; }
        /// <summary>
        /// List of all neurons contained within the layer.
        /// </summary>
        new List<INeuron> Neurons { get; }
        /// <summary>
        /// Sets given value to a single neuron of the Latent layer.
        /// </summary>
        /// <param name="value">New value for the neuron.</param>
        /// <param name="neuronID">Index of the neuron.</param>
        new void ModifyNeuron(float value, int neuronID);

        /// <summary>
        /// Connects outputs of all previous maps with inputs to this map,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousLayer">Filter maps of previous layer.</param>
        new void ConnectNeurons(List<IMap> previousLayer);

        /// <summary>
        /// Connects outputs of the previous map with inputs to this map,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousLayer">Filter maps of previous layer.</param>
        new void ConnectNeurons(IMap previousLayer);
        /// <summary>
        /// Sets the neurons to provided list. Use only if you know what you're doing or with
        /// input layer.
        /// </summary>
        /// <param name="previousLayerNeurons"></param>
        void SetNeurons(List<INeuron> previousLayerNeurons);

        /// <summary>
        /// Calculates activation of all neurons using provided function.
        /// The function object has to be thread safe.
        /// </summary>
        /// <param name="activationActivationFunction"></param>
        void CalculateNeuronsActivation(IActivationFunction activationActivationFunction);

        string ToString();
    }
}
