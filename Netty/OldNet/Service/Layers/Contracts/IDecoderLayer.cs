namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    /// <summary>
    /// Defines what should a single layer have.
    /// </summary>
    public interface IDecoderLayer : IMutatingLayer, IDeltaCalculatingLayer
    {
        int ThisLayerID { get; }
        /// <summary>
        /// Height of provided input.
        /// </summary>
        int Height { get; }
        /// <summary>
        /// Width of provided input.
        /// </summary>
        int Width { get; }
        /// <summary>
        /// Amount of filters in this layer.
        /// </summary>
        int Size { get; }
        /// <summary>
        /// List of all neurons contained within the layer.
        /// </summary>
        List<IMap> Maps { get; }
        /// <summary>
        /// Connects outputs of the previous layer with inputs to this layer,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousEncoderLayer">Layer which will this layer be connected to.</param>
        void ConnectNeurons(IDecoderLayer nextDecoderLayer);
        /// <summary>
        /// Connects outputs of the previous layer with inputs to this layer,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousLayer">Layer which will this layer be connected to.</param>
        void ConnectNeurons(IOutputLayer nextLayer);
        /// <summary>
        /// Calculates activations of all neurons in the layer. Has to be thread safe.
        /// </summary>
        void CalculateNeuronsActivation();
        string ToString();

    }
}
