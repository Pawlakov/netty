namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    /// <summary>
    /// Interface for classes that can retrieve info about current training status.
    /// </summary>
    public interface ITrainingOutput
    {
        /// <summary>
        /// Writes provided data to output.
        /// </summary>
        /// <param name="data"></param>
        void Report(string data);
        /// <summary>
        /// Clears the output.
        /// </summary>
        void ClearOuptut();
    }
}
