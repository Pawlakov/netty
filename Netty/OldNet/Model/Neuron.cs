namespace ClickbaitGenerator.NeuralNet.Model
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Enums;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;

    public class Neuron : INeuron
    {
        private static int _neuronID = 0;

        /// <summary>
        /// State of the neuron. Ranges from 0 to 1.
        /// </summary>
        public float Activation { get; set; }
        public long ID { get; }
        public float Delta { get; set; }
        public List<IConnection> InputConnections { get; } = new List<IConnection>();
        public List<IConnection> OutputConnections { get; } = new List<IConnection>();

        public Neuron(float activation)
        {
            this.Activation = activation;
            this.ID = _neuronID;
            _neuronID++;
        }

        public void BackPropagate(float value)
        {
            throw new NotImplementedException();
        }

        public void PerformUpdate()
        {
            throw new NotImplementedException();
        }

        public void AddConnection(IConnection connection, ConnectionAssignmentType providedNeuronConnection)
        {
            switch (providedNeuronConnection)
            {
                case ConnectionAssignmentType.Input:
                    connection.AssignOutputNeuron(this);    //you are the output neuron of this connection, assign yourself.
                    this.InputConnections.Add(connection);
                    break;
                case ConnectionAssignmentType.Output:
                    connection.AssignInputNeuron(this);    //you are the input neuron of this connection, assign yourself.
                    this.OutputConnections.Add(connection);
                    break;
            }
            
        }

        public float CalculateInputsSum()
        {
            float sum = 0.0f;
            int connectionsCount = this.InputConnections.Count;
            IConnection inputConnection;
            for (int i = 0; i < connectionsCount; i++)
            {
                inputConnection = this.InputConnections[i];
                if (inputConnection.InputNeuron != null)
                {
                    sum += inputConnection.Weight.Value * inputConnection.InputNeuron.Activation;
                }
            }

            return sum;
        }

        public int InputConnectionsCount()
        {
            return this.InputConnections.Count;
        }

        public int OutputConnectionsCount()
        {
            return this.OutputConnections.Count;
        }


        public override string ToString()
        {
            string msg = "Neuron " + this.ID + " InputConnections: \n";

            foreach (var connection in this.InputConnections)
            {
                msg += "    " + connection.ToString();
            }
            msg += "    OutputConnections: \n";

            foreach (var connection in this.OutputConnections)
            {
                msg += "    " + connection.ToString();
            }

            return msg;
        }

        public float CalculateOutputConnDeltasSum()
        {
            var outputConnectionsAmount = this.OutputConnections.Count;
            var deltasSum = 0.0f;
            for (int i = 0; i < outputConnectionsAmount; i++)
            {
                var outputConnection = this.OutputConnections[i];
                var outputNeuron = outputConnection.OutputNeuron;
                if (outputNeuron != null)
                {
                    deltasSum += outputNeuron.Delta * outputConnection.Weight.Value;
                    if (float.IsNaN(deltasSum))
                    {
                        //throw new NeuralNetworkException("THAT WAS NAN!");
                    }
                }
            }

            return deltasSum;
        }

        public void MutateOutputConnections(float learningFactor, float inertiaFactor)
        {
            var outputConnectionsCount = this.OutputConnections.Count;
            for (int i = 0; i < outputConnectionsCount; i++)
            {
                var connection = this.OutputConnections[i];
                var outputNeuron = connection.OutputNeuron;

                if (outputNeuron != null)
                {
                    var newChange = learningFactor * outputNeuron.Delta * this.Activation + inertiaFactor * connection.lastWeightChange;
                    connection.Weight.Value += newChange;
                    connection.lastWeightChange = newChange;
                }
            }
        }
    }
}
