namespace ClickbaitGenerator.NeuralNet.Model
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class OutputMap : IMap
    {
        private static int ID = 0;
        public int ThisMapID{get; private set; }

        public float Bias { get; set; }
        public int Height { get; }
        public int Width { get; }
        public List<INeuron> Neurons { get; private set; }
        public void ModifyNeuron(float value, int neuronID)
        {
            throw new NotImplementedException();
        }

        public OutputMap(int width, int height)
        {
            this.Width = width;
            this.Height = height;

            this.Bias = CustomRandom.NextFloat();

            this.ThisMapID = ID;
            ID++;
        }
        public void ConnectNeurons(IMap previousLayer)
        {
            var tempList = new List<IMap>();
            tempList.Add(previousLayer);
            this.ConnectNeurons(tempList);
        }

        public void ConnectNeurons(List<IMap> previousLayer)
        {
            var random = new Random();
            int layerNeuronSize = this.Width * this.Height;
            int previousLayerSize = previousLayer.Count;
            for (int i = 0; i < layerNeuronSize; i++)
            {
                var newNeuron = new Neuron((float)random.NextDouble());

                for (int j = 0; j < previousLayerSize; j++)
                {
                    var connection = new Connection(CustomRandom.NextFloat());
                    ConnectionHelper.AssignToConnectionBackwards(newNeuron, previousLayer[j].Neurons[i], connection);
                    //connection.AssignInputNeuron(previousLayer[j].Neurons[i]);
                    //newNeuron.AddConnection(connection, connectSourceAs);
                    this.Neurons.Add(newNeuron);
                }
            }
        }

        public void CalculateNeuronsActivation(IActivationFunction activationActivationFunction)
        {
            var activationSum = 0.0f;
            INeuron currentNeuron;
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                currentNeuron = this.Neurons[i];
                activationSum = currentNeuron.CalculateInputsSum();
                currentNeuron.Activation = activationActivationFunction.Calculate(activationSum + this.Bias);
            }
        }

        public void SetNeurons(List<INeuron> neurons)
        {
            this.Neurons = neurons;
        }

        public override string ToString()
        {
            string msg = "Output map: " + this.ThisMapID + "\n";

            foreach (var neuron in this.Neurons)
            {
                msg += "    " + neuron;
            }
            

            return msg;
        }
    }
}
