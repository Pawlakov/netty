namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    /// <summary>
    /// Used with layers that can calculate their deltas (used as base for mutating of the weights of the connections).
    /// </summary>
    public interface IDeltaCalculatingLayer
    {
        /// <summary>
        /// Makes the layer calculate its deltas basing on the layer after it. 
        /// </summary>
        void CalculateDeltas();
    }
}
