namespace ClickbaitGenerator.NeuralNet.Helpers
{
    using ClickbaitGenerator.NeuralNet.Enums;
    using ClickbaitGenerator.NeuralNet.Model;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public static class ConnectionHelper
    {
        /// <summary>
        /// Assigns the provided neurons to the connection and adds it, properly permuted, to both of them.
        /// Version for decoder (when layer that is being connected to new layer is after the new layer).
        /// </summary>
        /// <param name="currentLayerNeuron">Neuron of the new layer.</param>
        /// <param name="connectingLayerNeuron">Neuron of the layer that is being connected to new layer.</param>
        /// <param name="connection">Connection with already preset weight.</param>
        public static void AssignToConnectionBackwards(INeuron currentLayerNeuron,
            INeuron connectingLayerNeuron, Connection connection)
        {
            connection.AssignOutputNeuron(connectingLayerNeuron);
            currentLayerNeuron.AddConnection(connection, ConnectionAssignmentType.Output);

            if (connectingLayerNeuron != null)
            {
                var mirrorConnection = new Connection(connection);
                //mirrorConnection.MirrorNeurons(connection);

                connectingLayerNeuron.AddConnection(mirrorConnection, ConnectionAssignmentType.Input);
            }
        }
        /// <summary>
        /// Assigns the provided neurons to the connection and adds it, properly permuted, to both of them.
        /// Version for encoder (when layer that is being connected to new layer is before the new layer).
        /// </summary>
        /// <param name="currentLayerNeuron">Neuron of the new layer.</param>
        /// <param name="connectingLayerNeuron">Neuron of the layer that is being connected to new layer.</param>
        /// <param name="connection">Connection with already preset weight.</param>
        public static void AssignToConnection(INeuron currentLayerNeuron,
            INeuron connectingLayerNeuron, Connection connection)
        {
            connection.AssignInputNeuron(connectingLayerNeuron);
            currentLayerNeuron.AddConnection(connection, ConnectionAssignmentType.Input);

            if (connectingLayerNeuron != null)
            {
                var mirrorConnection = new Connection(connection);
                //mirrorConnection.MirrorNeurons(connection);

                connectingLayerNeuron.AddConnection(mirrorConnection, ConnectionAssignmentType.Output);
            }
        }
    }
}
