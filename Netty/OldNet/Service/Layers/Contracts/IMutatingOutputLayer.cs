namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    using System.Collections.Generic;

    /// <summary>
    /// Used with output layers that are capable of learning.
    /// </summary>
    public interface IMutatingOutputLayer
    {
        /// <summary>
        /// Calculates the deltas for this layer using provided gradient (vector) of errors.
        /// </summary>
        /// <param name="errorGradient"></param>
        void CalculateDeltas(List<float> errorGradient);
    }
}
