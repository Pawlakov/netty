namespace ClickbaitGenerator.NeuralNet.Model
{
    using System;
    using System.Collections.Generic;

    using ClickbaitGenerator.NeuralNet.Helpers;
    using ClickbaitGenerator.NeuralNet.Model.Contracts;
    using ClickbaitGenerator.NeuralNet.Service.Functions.Contracts;

    public class UnpoolingMap : IUnpoolingMap
    {
        private static int ID = 0;
        public int ThisMapID{get; private set;}

        public float Bias { get; set; }
        public int Height { get; }
        public int Width { get; }
        private int _divisor { get; set; }
        /// <summary>
        /// How many neurons are contained in unpooling filter (or window).
        /// </summary>
        private int _unpoolingWindowSize;
        public List<INeuron> Neurons { get; } = new List<INeuron>();

        public UnpoolingMap(int width, int height, int divisor)
        {
            this.Width = width;
            this.Height = height;
            this._divisor = divisor;
            this._unpoolingWindowSize = divisor * divisor;

            this.Bias = CustomRandom.NextFloat();

            Random random = new Random();
            int size = width * height;
            for (int i = 0; i < size; i++)
            {
                this.Neurons.Add(new Neuron((float)random.NextDouble()));
            }

            this.ThisMapID = ID;
            ID++;
        }
        public void ModifyNeuron(float value, int neuronID)
        {
            throw new NotImplementedException();
        }

        public void ConnectNeurons(IMap previousLayer)
        {
            this.Neurons.Clear();
            Random random = new Random();
            var thisMapSize = this.Height * this.Width;
            var sourceNeurons = previousLayer.Neurons;
            var sourceMapWidth = previousLayer.Width;
            var sourceMapHeight = previousLayer.Height;

            for (int i = 0; i < thisMapSize; i++)
            {
                var neuron = new Neuron((float)random.NextDouble());
                this.Neurons.Add(neuron);
            }

            for (int j = 0; j < sourceNeurons.Count; j++)
            {
                var row = (int)Math.Floor((float)j / sourceMapWidth);
                var column = j - row * sourceMapWidth;
                var unpoolingColumn = (int) Math.Floor(column/(float)this._divisor);
                var unpoolingRow = (int) Math.Floor(row/(float)this._divisor);
                var unpoolingIndex = unpoolingColumn + unpoolingRow * this.Width;

                ConnectionHelper.AssignToConnectionBackwards(this.Neurons[unpoolingIndex], sourceNeurons[j], new Connection(CustomRandom.NextFloat()));
            }

            /*var sourceWidth = previousLayer.Width;  //Placeholder for source width
            var calcWidth = _divisor * sourceWidth;              //Width accordingly to size of previous layer*_divisor. Can be higher than truly needed if pooling was bigger than pooled convolution layer.
            var calcHeight = _divisor * previousLayer.Height;    //Height accordingly to size of previous layer*_divisor. Can be higher than truly needed if pooling was bigger than pooled convolution layer.
            var sourceNeurons = previousLayer.Neurons;
            int destSize = previousLayer.Width * previousLayer.Height;                  //Total amount of neurons to create. Can be higher than truly needed if pooling was bigger than pooled convolution layer.

            for (int i = 0; i < destSize; i++)
            {
                var column = (int)Math.Floor((float)i/_divisor) % sourceWidth;   //Unpooling result is bigger than source - make sure to not exceed the width of source map.
                var row = i / (calcWidth * _divisor);                            //Same goes for rows. More than one row of the result can reference single row from source.
                var newConnection = new Connection(CustomRandom.NextFloat());
                var index = column + row * sourceWidth;                             //Calculate index of source neuron.
                //newConnection.AssignInputNeuron(sourceNeurons[index]);
                var newNeuron = new Neuron((float)random.NextDouble());

                ConnectionHelper.AssignToConnectionBackwards(newNeuron, sourceNeurons[index], newConnection);
                //newNeuron.AddConnection(newConnection, connectSourceAs);
                Neurons.Add(newNeuron);
            }

            //Now, cut away any neurons that are exceeding the original size (of responsive encoder layer).
                //By height
            if(Height < calcHeight)
            {
                var removeFrom = calcWidth * Height;
                Neurons.RemoveRange(removeFrom, Neurons.Count - removeFrom);
            }
                //By width
            if(Width < calcWidth)
            {
                var rowExcess = calcWidth - Width;
                for (int i = Height; i > 0; i--)
                {
                    var startRemoveFrom = i * calcWidth - rowExcess;
                    Neurons.RemoveRange(startRemoveFrom, rowExcess);
                }
            }*/
        }

        public void ConnectNeurons(List<IMap> previousLayer)
        {
            this.ConnectNeurons(previousLayer[0]);
        }

        public override string ToString()
        {
            string msg = "Unpooling map: " + this.ThisMapID + "\n";

            foreach (var neuron in this.Neurons)
            {
                msg += "    " + neuron;
            }

            return msg;
        }

        public void CalculateNeuronsActivation(List<int> maxIndexes, IActivationFunction activationActivationFunction)
        {
            //Calculate activation of this map's neurons.
            var neuronsCount = this.Neurons.Count;
            int i;
            for (i = 0; i < neuronsCount; i++)
            {
                var currentNeuron = this.Neurons[i];
                var thisNeuronActivation = activationActivationFunction.Calculate(currentNeuron.CalculateInputsSum() + this.Bias);
                currentNeuron.Activation = thisNeuronActivation;
            }
            //Set proper outgoing connections to 1.0f (only one such connection may exist, the rest shall be set to 0.
            //for (int i = 0; i < neuronsCount; i++)
            //{
            //    Neurons[i].Activation = 0;      //Reset all activations.
            //}
            int j = 0;
            var maxIndexesCount = maxIndexes.Count;
            for (i = 0; i < maxIndexesCount; i++)//The max index count and neurons list have same size.
            {
                //i*unpoolingWindowSize - maxIndexes stores relative index to given window. Position of the window beginning has to be added to it.
                INeuron maxNeuron = this.Neurons[i];     //Get the neuron which is at the same index as max from  pooling layer
                var outputConnections = maxNeuron.OutputConnections;
                var outputConnectionsCount = outputConnections.Count;
                for (j = 0; j < outputConnectionsCount; j++)
                {
                    //outputConnections[j].Weight.Value = 0.0f;  //Reset weights of all output connections.
                    outputConnections[j].Weight.Value = 1.0f;   //Scatter the value to all output neurons.
                }

                //maxNeuron.OutputConnections[maxIndexes[i]].Weight.Value = 1.0f;    //Assign to connection with max value weight that allows for reading the neuron activation value, without any changes.
            }
        }
    }
}
