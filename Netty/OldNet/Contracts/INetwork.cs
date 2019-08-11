namespace ClickbaitGenerator.NeuralNet.Contracts
{
    using System.Collections.Generic;
    
    public interface INetwork
    {
        void Mutate(List<float> errorGradient, float learningFactor, float inertiaFactor);

        /// <summary>
        /// Assigns provided values to input layer, then begins calculations.
        /// <param name="input">Input values for the network (encoder).</param>
        /// </summary>
        /// <returns>Result for provided input.</returns>
        List<float> PerformCalculations(List<float> input);

        /// <summary>
        /// Makes the provided image data to be processed by the encoder and returns
        /// results from the latent (bottleneck) layer.
        /// </summary>
        /// <param name="input">Input image. Each pixel equals one float times amount of channels of that pixel.</param>
        /// <returns></returns>
        List<float> PerformCalculationsEncoderOnly(List<float> input);

        /// <summary>
        /// Loads provided vector to latent(bottleneck) layer and triggers calculations
        /// in the decoder. Result is a recreated image, provided from output layer.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        List<float> PerformCalculationsDecoderOnly(List<float> input);
        
        /// <summary>
        /// Returns the amount of neurons in the input layer.
        /// </summary>
        /// <returns></returns>
        int GetInputSize();

        // Debug info

        string PrintInputLayer();

        string PrintEncoderLayers();

        string PrintLatentLayer();

        string PrintDecoderLayers();

        string PrintOutputLayer();

        string CountEverything();
    }
}