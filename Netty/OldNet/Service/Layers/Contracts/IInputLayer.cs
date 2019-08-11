namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public interface IInputLayer : IMutatingLayer
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
        /// Encoded source image.
        /// </summary>
        List<IMap> SourceImage { get; }
        
        /// <summary>
        /// Provides input data for the network.
        /// </summary>
        /// <param name="data">Values ranging from 0 to 1.</param>
        void SetInput(List<float> data);

        string ToString();
    }
}
