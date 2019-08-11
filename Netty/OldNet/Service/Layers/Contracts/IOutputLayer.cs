namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public interface IOutputLayer
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
        /// Decoded source image maps.
        /// </summary>
        List<IMap> SourceImage { get; }

        void ConnectNeurons(List<IMap> neurons);
        /// <summary>
        /// Calculates activations of all neurons in the layer. Has to be thread safe.
        /// </summary>
        void CalculateNeuronsActivation();
        /// <summary>
        /// Provides input data for the network.
        /// </summary>
        /// <param name="data">Values ranging from 0 to 1.</param>
        List<float> GetOutput();
    }
}
