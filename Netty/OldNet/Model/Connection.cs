namespace ClickbaitGenerator.NeuralNet.Model
{
    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public class Connection : IConnection
    {
        public INeuron InputNeuron { get; private set; }
        public INeuron OutputNeuron { get; private set; }

        public float lastWeightChange { get; set; }
        // Since there can be two connections connecting same neurons (one per neuron),
        // make it possible for their weight to be share between them.

        /// <summary>
        /// Weight of the connection.
        /// </summary>
        public Ref<float> Weight { get; private set; }

        public Connection(float weight)
        {
            this.Weight = new Ref<float>(weight);
        }

        public Connection(Connection connection)
        {
            this.Weight = connection.Weight;
            this.InputNeuron = connection.InputNeuron;
            this.OutputNeuron = connection.OutputNeuron;
        }
        /// <summary>
        /// Assigns neurons from provided connection but input to output an vice-versa.
        /// </summary>
        /// <param name="source">Source connection which the neurons will be retrieved from.</param>
        public void MirrorNeurons(Connection source)
        {
            this.InputNeuron = source.OutputNeuron;
            this.OutputNeuron = source.InputNeuron;
        }
        public void AssignInputNeuron(INeuron inputNeuron)
        {
            this.InputNeuron = inputNeuron;
        }

        public void AssignOutputNeuron(INeuron outputNeuron)
        {
            this.OutputNeuron = outputNeuron;
        }

        public void ModifyWeight(float value)
        {
            this.Weight.Value += value;
        }

        public override string ToString()
        {
            return "    Input Neuron: " + this.InputNeuron?.ID + " OutputNeuron: " + this.OutputNeuron?.ID + '\n';
        }
    }
}
