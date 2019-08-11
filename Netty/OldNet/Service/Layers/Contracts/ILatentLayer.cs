namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public interface ILatentLayer : IMutatingLayer, IDeltaCalculatingLayer
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
        /// Amount of neurons that form the latent space, or bottleneck, layer.
        /// </summary>
        int LatentSpaceSize { get; }
        /// <summary>
        /// Stores the neurons that are used to flatten all maps from connected layer - input from encoder.
        /// </summary>
        List<INeuron> FlattenInputNeurons { get; }
        /// <summary>
        /// List of all neurons contained within the layer.
        /// </summary>
        List<INeuron> Neurons { get; }
        /// <summary>
        /// List on Neurons connected to the decoder.
        /// </summary>
        List<INeuron> DeflattenLayer { get; }
        /// <summary>
        /// Sets given value to a single neuron of the Latent layer.
        /// </summary>
        /// <param name="value">New value for the neuron.</param>
        /// <param name="neuronID">Index of the neuron.</param>
        void ModifyNeuron(float value, int neuronID);

        /// <summary>
        /// Connects outputs of the previous layer with inputs to this layer,
        /// using provided configuration.
        /// </summary>
        /// <param name="previousEncoderLayer">Layer which will this layer be connected to.</param>
        void ConnectNeurons(IEncoderLayer previousEncoderLayer);
        /// <summary>
        /// Connects the output neurons of this layer to provided layer. All to all type.
        /// </summary>
        /// <param name="outputEncoderLayer">Layer which will this layer be connected to.</param>
        void ConnectOutputNeurons(IDecoderLayer outputEncoderLayer);
        /// <summary>
        /// Calculates activations of all neurons in the layer.
        /// </summary>
        void CalculateNeuronsActivation();
        string ToString();
        /// <summary>
        /// Returns activations of neurons from the latent (bottleneck) layer.
        /// </summary>
        /// <returns></returns>
        List<float> GetLatentValues();
        /// <summary>
        /// Sets values of the latent (bottleneck) layer.
        /// </summary>
        /// <param name="input"></param>
        void SetLatentLayer(List<float> input);
    }
}
