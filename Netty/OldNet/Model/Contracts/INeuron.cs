namespace ClickbaitGenerator.NeuralNet.Model.Contracts
{
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Enums;

    public interface INeuron
    {
        /// <summary>
        /// Calculated during learning process. Used to modify the weights.
        /// </summary>
        float Delta { get; set; }
        /// <summary>
        /// Connections that are coming out of this neuron.
        /// </summary>
        List<IConnection> OutputConnections { get; }
        /// <summary>
        /// Connections that come into this neuron.
        /// </summary>
        List<IConnection> InputConnections { get; }
        /// <summary>
        /// The unique ID of this neuron
        /// </summary>
        long ID { get; }
        /// <summary>
        /// How much is the neuron active.
        /// </summary>
        float Activation { get; set; }
        /// <summary>
        /// Used to propagate the fitness function backwards for learning.
        /// </summary>
        /// <param name="value">Value that will be used to modify the connections.</param>
        void BackPropagate(float value);
        /// <summary>
        /// Makes the neuron calculate it's new value.
        /// </summary>
        void PerformUpdate();

        /// <summary>
        /// Adds a connection to this neuron. Make sure that you added the output neuron
        /// to the connection before adding it; this neuron will assign itself as input one.
        /// </summary>
        /// <param name="connection"></param>
        /// <param name="providedNeuronConnection">Defines whether provided with connection neuron is assigned as input or output.</param>
        void AddConnection(IConnection connection, ConnectionAssignmentType providedNeuronConnection);
        /// <summary>
        /// Multiplies each connection weight with its input neuron activation and returns the sum of results.
        /// </summary>
        /// <returns></returns>
        float CalculateInputsSum();

        int InputConnectionsCount();
        int OutputConnectionsCount();

        string ToString();
        /// <summary>
        /// Calculates sum of deltas from output connections of this neuron.
        /// </summary>
        /// <returns></returns>
        float CalculateOutputConnDeltasSum();
        /// <summary>
        /// Mutates weights of outer connections.
        /// </summary>
        void MutateOutputConnections(float learningFactor, float inertiaFactor);
    }
}
