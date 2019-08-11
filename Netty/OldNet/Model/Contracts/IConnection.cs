namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{

    /// <summary>
    /// Interface for connections between neurons.
    /// </summary>
    public interface IConnection
    {
        float lastWeightChange { get; set; }
        Ref<float> Weight { get; }
        INeuron InputNeuron { get; }
        INeuron OutputNeuron { get; }
        /// <summary>
        /// Assigns the input neuron to this connection.
        /// </summary>
        /// <param name="inputNeuron"></param>
        void AssignInputNeuron(INeuron inputNeuron);
        /// <summary>
        /// Assigns output neuron to this connection.
        /// </summary>
        /// <param name="outputNeuron"></param>
        void AssignOutputNeuron(INeuron outputNeuron);
        /// <summary>
        /// Adds the argument to weight.
        /// </summary>
        /// <param name="value"></param>
        void ModifyWeight(float value);

        string ToString();
    }
}
