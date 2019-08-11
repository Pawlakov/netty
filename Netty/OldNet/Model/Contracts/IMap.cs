namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    using System.Collections.Generic;
    
    public interface IMap
    {
        float Bias { get; set; }
        int ThisMapID { get; }

        /// <summary>
        /// Height of provided input.
        /// </summary>
        int Height { get; }
        /// <summary>
        /// Width of provided input.
        /// </summary>
        int Width { get; }
        /// <summary>
        /// List of all neurons contained within the layer.
        /// </summary>
        List<INeuron> Neurons { get; }
        /// <summary>
        /// Sets given value to a single neuron of the Latent layer.
        /// </summary>
        /// <param name="value">New value for the neuron.</param>
        /// <param name="neuronID">Index of the neuron.</param>
        void ModifyNeuron(float value, int neuronID);

        /// <summary>
        /// Connects outputs of the previous map with inputs to this map,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousLayer">Filter maps of previous layer.</param>
        void ConnectNeurons(IMap previousLayer);
        /// <summary>
        /// Connects outputs of all previous maps with inputs to this map,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousLayer">Filter maps of previous layer.</param>
       void ConnectNeurons(List<IMap> previousLayer);
        
        string ToString();
    }
}
