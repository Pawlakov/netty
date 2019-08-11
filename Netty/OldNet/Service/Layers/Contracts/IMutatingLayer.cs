namespace ClickbaitGenerator.NeuralNet.Service.Layers.Contracts
{
    /// <summary>
    /// Defines a layer that is capable of learning.
    /// </summary>
    public interface IMutatingLayer
    {
        /// <summary>
        /// Begins learning process, modifying the weights.
        /// Keep in mind to call CalculateDeltas method first.
        /// </summary>
        /// <param name="learningFactor">By how big change will the weights be modified.</param>
        /// <param name="inertiaFactor">How long it takes for changes to work.</param>
        void Mutate(float learningFactor, float inertiaFactor);
    }
}
